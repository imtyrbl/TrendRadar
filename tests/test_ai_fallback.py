"""AI 渠道回落/兜底逻辑的离线回归测试（不发任何真实网络请求）"""

import io
import os
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

# 避免 LiteLLM 导入时联网拉取模型价格表
with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}):
    from trendradar.ai.analyzer import AIAnalyzer
    from trendradar.ai.filter import AIFilter
    from trendradar.ai.jev_client import JevClient
    from trendradar.context import AppContext
    from trendradar.core.loader import resolve_ai_fallback_config


JEV_CONFIG = {
    "MODEL": "jev-latest",
    "API_KEY": "test-key",
    "API_BASE": "https://jev.example.invalid",
    "TIMEOUT": 300,
}


class JevProbeTests(unittest.TestCase):
    def setUp(self):
        self.client = JevClient(JEV_CONFIG)

    def test_probe_passes_when_answer_present(self):
        with patch.object(JevClient, "decide", return_value={"answers": {"probe": {"choice": "test"}}}) as decide:
            ok, reason = self.client.probe(15)
        self.assertTrue(ok)
        self.assertEqual(reason, "")
        # 探活必须用自己的短超时，而不是批量分类的长超时
        self.assertEqual(decide.call_args.kwargs["timeout"], 15)

    def test_probe_fails_on_upstream_error(self):
        error = RuntimeError("Jev 请求失败 HTTP 502: upstream")
        with patch.object(JevClient, "decide", side_effect=error):
            ok, reason = self.client.probe(15)
        self.assertFalse(ok)
        self.assertIn("502", reason)

    def test_probe_fails_when_answers_missing(self):
        with patch.object(JevClient, "decide", return_value={"error": "x"}):
            ok, _ = self.client.probe(15)
        self.assertFalse(ok)

    def test_probe_fails_fast_on_incomplete_config(self):
        client = JevClient({**JEV_CONFIG, "API_BASE": ""})
        with patch.object(JevClient, "decide") as decide:
            ok, _ = client.probe(15)
        self.assertFalse(ok)
        decide.assert_not_called()


class JevFallbackToChatTests(unittest.TestCase):
    def _make_filter(self):
        filter_config = {
            "PROVIDER": "jev",
            "BATCH_SIZE": 50,
            "JEV_MODEL": "jev-latest",
            "JEV_API_KEY": "test-key",
            "JEV_API_BASE": "https://jev.example.invalid",
            "JEV_PROBE_TIMEOUT": 10,
            "JEV_MAX_ITEMS_PER_BATCH": 15,
        }
        with redirect_stdout(io.StringIO()):
            return AIFilter({"MODEL": "test/model", "API_KEY": "k"}, filter_config, lambda: None)

    def test_probe_failure_switches_whole_run_to_chat(self):
        ai_filter = self._make_filter()
        titles = [{"id": i, "title": f"t{i}", "source": "s"} for i in range(120)]
        tags = [{"id": 1, "tag": "AI", "description": ""}]

        with patch.object(JevClient, "probe", return_value=(False, "HTTP 502")), \
                redirect_stdout(io.StringIO()):
            ai_filter.ensure_jev_available()

        self.assertEqual(ai_filter.provider, "chat")
        # 回落后按 chat 的 batch_size 分批
        self.assertEqual([len(b) for b in ai_filter.plan_batches(titles, tags)], [50, 50, 20])
        with patch.object(AIFilter, "_classify_batch_chat", return_value=[]) as chat, \
                patch.object(AIFilter, "_classify_batch_jev") as jev:
            ai_filter.classify_batch(titles[:3], tags)
        chat.assert_called_once()
        jev.assert_not_called()

    def test_probe_success_keeps_jev(self):
        ai_filter = self._make_filter()
        with patch.object(JevClient, "probe", return_value=(True, "")), \
                redirect_stdout(io.StringIO()):
            ai_filter.ensure_jev_available()
        self.assertEqual(ai_filter.provider, "jev")


class FallbackConfigTests(unittest.TestCase):
    def _config(self, fallback):
        return {
            "AI": {"MODEL": "openai/main", "API_KEY": "main-key", "API_BASE": "https://main", "FALLBACK_MODELS": ["x/y"]},
            "AI_ANALYSIS_MODEL": {"model": "", "timeout": 120, "fallback": fallback},
        }

    def test_disabled_without_model(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AI_ANALYSIS_FALLBACK_MODEL", None)
            self.assertIsNone(resolve_ai_fallback_config(
                self._config({"model": "", "timeout": 180}), "AI_ANALYSIS_MODEL", "AI_ANALYSIS"))

    def test_inherits_primary_and_overrides(self):
        with patch.dict(os.environ, {"AI_ANALYSIS_FALLBACK_API_KEY": "backup-key"}):
            result = resolve_ai_fallback_config(
                self._config({"model": "openai/backup", "timeout": 180}), "AI_ANALYSIS_MODEL", "AI_ANALYSIS")
        self.assertEqual(result["MODEL"], "openai/backup")
        self.assertEqual(result["API_KEY"], "backup-key")
        self.assertEqual(result["API_BASE"], "https://main")
        self.assertEqual(result["TIMEOUT"], 180)
        self.assertEqual(result["FALLBACK_MODELS"], [])
        self.assertNotIn("FALLBACK", result)

    def test_model_from_env_only(self):
        with patch.dict(os.environ, {"AI_ANALYSIS_FALLBACK_MODEL": "openai/env-backup"}):
            result = resolve_ai_fallback_config(
                self._config({"model": ""}), "AI_ANALYSIS_MODEL", "AI_ANALYSIS")
        self.assertEqual(result["MODEL"], "openai/env-backup")


def _response(content, finish_reason="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content), finish_reason=finish_reason)])


class AnalyzerFallbackTests(unittest.TestCase):
    def _make_analyzer(self, with_fallback=True):
        primary = {"MODEL": "openai/main", "API_KEY": "k", "NUM_RETRIES": 0}
        fallback = {"MODEL": "openai/backup", "API_KEY": "k2", "NUM_RETRIES": 0} if with_fallback else None
        with patch("trendradar.ai.analyzer.load_prompt_template", return_value=("", "{news_content}")):
            return AIAnalyzer(primary, {}, lambda: None, fallback_config=fallback)

    def _run(self, analyzer, side_effect):
        with patch("trendradar.ai.client.completion", side_effect=side_effect) as completion, \
                redirect_stdout(io.StringIO()):
            result = analyzer._call_ai("prompt")
        return result, [c.kwargs["model"] for c in completion.call_args_list]

    def test_primary_ok_no_fallback(self):
        result, models = self._run(self._make_analyzer(), [_response("{}")])
        self.assertEqual(result, "{}")
        self.assertEqual(models, ["openai/main"])

    def test_timeout_uses_fallback(self):
        result, models = self._run(self._make_analyzer(), [TimeoutError("timed out"), _response("{}")])
        self.assertEqual(result, "{}")
        self.assertEqual(models, ["openai/main", "openai/backup"])

    def test_truncated_by_thinking_uses_fallback(self):
        analyzer = self._make_analyzer()
        result, models = self._run(analyzer, [_response('{"core', "length"), _response("{}")])
        self.assertEqual(result, "{}")
        self.assertEqual(models, ["openai/main", "openai/backup"])
        # JSON 修复重试要跟着产出响应的备用渠道走
        self.assertIs(analyzer._active_client, analyzer.fallback_client)

    def test_empty_content_uses_fallback(self):
        _, models = self._run(self._make_analyzer(), [_response(""), _response("{}")])
        self.assertEqual(models, ["openai/main", "openai/backup"])

    def test_without_fallback_error_propagates(self):
        with self.assertRaises(TimeoutError):
            self._run(self._make_analyzer(with_fallback=False), [TimeoutError("timed out")])


class PerChannelMinScoreTests(unittest.TestCase):
    def test_threshold_follows_result_channel(self):
        ctx = AppContext({
            "AI_FILTER": {"MIN_SCORE": 0.7, "JEV_MIN_SCORE": 0, "MIN_PUSH_SCORE": 0.5},
            "RSS": {"FRESHNESS_FILTER": {"ENABLED": False}},
        })
        items = [
            # chat 结果（push_score=None）：0.6 < chat 阈值 0.7，应被过滤
            {"title": "chat-low", "relevance_score": 0.6, "push_score": None},
            {"title": "chat-high", "relevance_score": 0.8, "push_score": None},
            # jev 结果：相关度 0.3 不受 chat 阈值影响，只看 push 阈值
            {"title": "jev-pass", "relevance_score": 0.3, "push_score": 0.9},
            {"title": "jev-push-low", "relevance_score": 0.9, "push_score": 0.2},
        ]
        result = SimpleNamespace(tags=[{"tag": "AI", "position": 1, "items": [
            {**i, "source_type": "hotlist", "first_time": "10-00", "last_time": "10-00"} for i in items
        ]}])
        with redirect_stdout(io.StringIO()):
            hotlist, _, _ = ctx.convert_ai_filter_to_report_data(result)
        titles = [t["title"] for t in hotlist[0]["titles"]]
        self.assertEqual(titles, ["chat-high", "jev-pass"])


class StandaloneSummaryFilterTests(unittest.TestCase):
    def test_summaries_for_unsent_sources_are_dropped(self):
        with patch("trendradar.ai.analyzer.load_prompt_template", return_value=("", "{standalone_content}")):
            analyzer = AIAnalyzer(
                {"MODEL": "openai/main", "API_KEY": "k"},
                {"INCLUDE_STANDALONE": True}, lambda: SimpleNamespace(strftime=lambda _: "now"),
            )
        standalone = {
            "platforms": [
                {"id": "zhihu", "name": "知乎", "items": [{"title": "t"}]},
                {"id": "wallstreetcn-hot", "name": "华尔街见闻", "items": []},
            ],
            "rss_feeds": [],
        }
        # 模型照抄示例输出了未送入的 Hacker News，以及无数据的华尔街见闻
        response = '{"standalone_summaries": {"[知乎]": "a", "Hacker News": "本期无该来源数据", "华尔街见闻": "c"}}'
        with patch.object(AIAnalyzer, "_call_ai", return_value=response), redirect_stdout(io.StringIO()):
            result = analyzer.analyze(
                [{"word": "AI", "titles": [{"title": "x"}]}], standalone_data=standalone)
        self.assertEqual(result.standalone_summaries, {"[知乎]": "a"})


if __name__ == "__main__":
    unittest.main()
