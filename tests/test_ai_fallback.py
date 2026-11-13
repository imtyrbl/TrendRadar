"""AI 渠道回落/兜底逻辑的离线回归测试（不发任何真实网络请求）"""

import io
import json
import os
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

# 避免 LiteLLM 导入时联网拉取模型价格表
with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}):
    from trendradar.ai.analyzer import AIAnalyzer
    from trendradar.ai.filter import AIFilter
    from trendradar.ai.filter_pipeline import AIFilterPipeline
    from trendradar.ai.gemini_decisions_client import (
        GeminiDecisionAPIError,
        GeminiDecisionClient,
        TokenRateLimiter,
    )
    from trendradar.ai.jev_client import JevClient
    from trendradar.context import AppContext
    from trendradar.core.loader import _load_ai_filter_config, resolve_ai_fallback_config


JEV_CONFIG = {
    "MODEL": "jev-latest",
    "API_KEY": "test-key",
    "API_BASE": "https://jev.example.invalid",
    "TIMEOUT": 300,
}


class GeminiDecisionTests(unittest.TestCase):
    TAGS = [
        {"id": 1, "tag": "AI模型与产品", "description": "关注人工智能模型和产品能力演进", "priority": 1},
        {"id": 2, "tag": "宏观经济", "description": "关注利率、就业和全球流动性", "priority": 2},
    ]
    TITLES = [
        {"id": 101, "title": "谷歌发布新一代 Gemini 模型", "source": "科技媒体"},
        {"id": 102, "title": "标题震惊！点开领取免费会员", "source": "营销号"},
    ]
    INTERESTS = "标题质量要求：\n- 不要标题党\n- 不要营销软文"

    def _make_filter(self, **overrides):
        config = {
            "PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-key",
            "GEMINI_MAX_ITEMS_PER_BATCH": 50,
            "GEMINI_TOKENS_PER_MINUTE": 35000,
            **overrides,
        }
        with redirect_stdout(io.StringIO()):
            return AIFilter({"MODEL": "test/model", "API_KEY": "test-key"}, config, lambda: None)

    def test_build_questions_carry_only_short_keys(self):
        ai_filter = self._make_filter()
        questions = ai_filter._gemini_build_questions(self.TITLES, self.TAGS)
        self.assertEqual(len(questions), 4)
        # 标签名与描述都只放共享 input，题目里的选项只有短键
        self.assertEqual(questions["n101"]["criteria"], {"T01": "", "T02": "", "other": ""})
        self.assertEqual(questions["n101"]["type"], "choice")
        self.assertEqual(questions["push101"]["type"], "noul")

    def test_shared_input_contains_rules_and_each_news_once(self):
        ai_filter = self._make_filter()
        input_text = ai_filter._gemini_build_input(self.TITLES, self.TAGS, self.INTERESTS)
        self.assertIn("关注人工智能模型和产品能力演进", input_text)
        self.assertIn("不要标题党", input_text)
        self.assertEqual(input_text.count("101: [科技媒体] 谷歌发布新一代 Gemini 模型"), 1)
        self.assertEqual(input_text.count("102: [营销号] 标题震惊！点开领取免费会员"), 1)

    def test_gemini_parser_preserves_two_independent_scores(self):
        response = {"answers": {
            "n101": {"choice": "T01", "confidence": 0.95},
            "push101": {"noul": 0.91},
            "n102": {"choice": "other", "confidence": 0.88},
            "push102": {"noul": 0.02},
        }}
        results = AIFilter._parse_gemini_response(response, self.TITLES, self.TAGS)
        self.assertEqual(results, [{
            "news_item_id": 101,
            "tag_id": 1,
            "relevance_score": 0.95,
            "push_score": 0.91,
        }])

    def test_missing_answer_fails_whole_batch(self):
        response = {"answers": {"n101": {"choice": "1"}}}
        self.assertIsNone(AIFilter._parse_gemini_response(response, self.TITLES, self.TAGS))

    def test_token_bucket_enforces_budget_and_reconciles_actual_usage(self):
        limiter = TokenRateLimiter(1000)
        with patch("trendradar.ai.gemini_decisions_client.time.sleep"):
            first = limiter.wait_for_capacity(100)
            limiter.reconcile(first, 40)
            self.assertAlmostEqual(limiter._available_tokens, 260, delta=0.01)
            with self.assertRaises(ValueError):
                limiter.wait_for_capacity(1000)

    def test_limiter_waits_for_exact_shared_tpm_budget(self):
        limiter = TokenRateLimiter(1000)
        waits = []

        def advance_time(seconds):
            waits.append(seconds)
            limiter._updated_at -= seconds

        with patch("trendradar.ai.gemini_decisions_client.time.sleep", side_effect=advance_time):
            limiter.wait_for_capacity(200)
            limiter.wait_for_capacity(200)
        self.assertEqual(len(waits), 1)
        self.assertAlmostEqual(waits[0], 200 * 60 / (1000 * 0.7), delta=0.001)

    def test_gemini_rest_client_sends_expected_endpoint_and_parses_usage(self):
        response_data = {
            "answers": {
                "category": {"type": "choice", "choice": "ai", "confidence": 0.99},
            },
            "usage": {"total_input_tokens": 169, "total_output_tokens": 0},
        }
        client = GeminiDecisionClient({"API_KEY": "unit-test-key", "TOKENS_PER_MINUTE": 1000})
        with patch(
            "trendradar.ai.gemini_decisions_client.urllib.request.urlopen",
            return_value=io.BytesIO(json.dumps(response_data).encode("utf-8")),
        ) as urlopen:
            result = client.decide("一个共享标题", {
                "category": {
                    "type": "choice",
                    "instructions": "标题属于哪个方向？",
                    "criteria": {"ai": "人工智能", "other": "其它"},
                },
            })
        self.assertEqual(result["usage"]["total_input_tokens"], 169)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://generativelanguage.googleapis.com/v1beta/decisions")
        self.assertEqual(request.get_header("X-goog-api-key"), "unit-test-key")
        self.assertEqual(json.loads(request.data)["model"], "gemini-decision-1")

    def test_plan_gemini_batches_splits_by_low_tpm_budget(self):
        # TPM=2000 → burst=600 tokens：装得下约 2 条新闻、装不下 3 条，必然切成多批
        ai_filter = self._make_filter(GEMINI_TOKENS_PER_MINUTE=2000, GEMINI_MAX_ITEMS_PER_BATCH=3)
        titles = [{"id": 100 + i, "title": f"第{i}条新闻标题内容", "source": "科技媒体"} for i in range(10)]
        tags = self.TAGS

        batches = ai_filter.plan_batches(titles, tags, self.INTERESTS)

        self.assertGreater(len(batches), 1)
        self.assertTrue(all(len(b) <= 3 for b in batches))
        self.assertEqual(sum(len(b) for b in batches), 10)
        seen_ids = [t["id"] for b in batches for t in b]
        self.assertEqual(seen_ids, [t["id"] for t in titles])
        burst = ai_filter.gemini_client.rate_limiter.burst_capacity
        self.assertEqual(burst, 600)
        for batch in batches:
            questions = ai_filter._gemini_build_questions(batch, tags)
            # Decisions API 单请求最多 1000 个问题（每条新闻 2 问）
            self.assertLessEqual(len(questions), 1000)
            estimated = ai_filter.gemini_client.estimate_request_tokens(
                ai_filter._gemini_build_input(batch, tags, self.INTERESTS), questions,
            )
            # 与 plan_gemini_batches 相同的安全系数后仍不得突破突发容量
            self.assertLessEqual(int(estimated * 1.25 + 0.999999), burst)

    def test_gemini_429_retries_once_then_falls_back_to_chat(self):
        ai_filter = self._make_filter()
        titles = [{"id": 100 + i, "title": f"第{i}条新闻标题内容", "source": "科技媒体"} for i in range(6)]
        pipeline = AIFilterPipeline(
            {"AI": {"MODEL": "test/model", "API_KEY": "k"}, "AI_FILTER": {}, "RSS": {"ENABLED": False}},
            Mock(),
            lambda: None,
        )
        error_429 = GeminiDecisionAPIError(429, "resource exhausted", retry_after=60.0)
        chat_result = [{"news_item_id": 100, "tag_id": 1, "relevance_score": 0.9}]

        with patch.object(ai_filter, "plan_batches", side_effect=[[titles[:2], titles[2:4], titles[4:]], []]),                 patch.object(GeminiDecisionClient, "decide", side_effect=error_429) as decide,                 patch.object(ai_filter, "_classify_batch_chat", return_value=chat_result) as chat,                 redirect_stdout(io.StringIO()):
            total_results, succeeded_news_ids, _ = pipeline._classify_batches(
                ai_filter, titles, [], self.TAGS, self.INTERESTS, {"BATCH_INTERVAL": 0},
            )

        # 首批：原请求 + 重试一次，都 429 → 切 chat；之后不再打 Gemini
        self.assertEqual(decide.call_count, 2)
        self.assertEqual(ai_filter.provider, "chat")
        # 三批都由 chat 完成，没有新闻被留到下轮
        self.assertEqual(chat.call_count, 3)
        self.assertEqual(sorted(succeeded_news_ids), [t["id"] for t in titles])
        self.assertEqual(len(total_results), 3)

    def test_gemini_429_recovers_on_retry(self):
        ai_filter = self._make_filter()
        ok = {"answers": {
            "n101": {"choice": "T01", "confidence": 0.95}, "push101": {"noul": 0.91},
            "n102": {"choice": "other", "confidence": 0.88}, "push102": {"noul": 0.02},
        }}
        with patch.object(GeminiDecisionClient, "decide",
                          side_effect=[GeminiDecisionAPIError(429, "x", retry_after=1.0), ok]),                 redirect_stdout(io.StringIO()):
            results = ai_filter.classify_batch(self.TITLES, self.TAGS, self.INTERESTS)
        self.assertEqual(ai_filter.provider, "gemini")
        self.assertEqual([r["news_item_id"] for r in results], [101])

    def test_jev_and_gemini_thresholds_load_separately(self):
        config = _load_ai_filter_config({"ai_filter": {
            "jev": {"min_score": 0.1, "min_push_score": 0.5},
            "gemini": {"min_score": 0.2, "min_push_score": 0.4},
        }})
        self.assertEqual((config["JEV_MIN_SCORE"], config["MIN_PUSH_SCORE"]), (0.1, 0.5))
        self.assertEqual((config["GEMINI_MIN_SCORE"], config["GEMINI_MIN_PUSH_SCORE"]), (0.2, 0.4))

    def test_provider_and_quota_configuration_load(self):
        with patch.dict(os.environ, {"AI_FILTER_GEMINI_API_KEY": "env-key", "AI_FILTER_GEMINI_TOKENS_PER_MINUTE": "42000"}):
            config = _load_ai_filter_config({"ai_filter": {"provider": "gemini"}})
        self.assertEqual(config["PROVIDER"], "gemini")
        self.assertEqual(config["GEMINI_API_KEY"], "env-key")
        self.assertEqual(config["GEMINI_TOKENS_PER_MINUTE"], 42000)

    def test_default_provider_remains_chat(self):
        config = _load_ai_filter_config({"ai_filter": {}})
        self.assertEqual(config["PROVIDER"], "chat")


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

    def test_gemini_provider_uses_its_own_push_threshold(self):
        base = {"MIN_SCORE": 0.7, "JEV_MIN_SCORE": 0, "MIN_PUSH_SCORE": 0.5,
                "GEMINI_MIN_SCORE": 0, "GEMINI_MIN_PUSH_SCORE": 0.4}
        item = {"title": "push-0.45", "relevance_score": 0.9, "push_score": 0.45,
                "source_type": "hotlist", "first_time": "10-00", "last_time": "10-00"}
        result = SimpleNamespace(tags=[{"tag": "AI", "position": 1, "items": [item]}])
        kept = {}
        for provider in ("jev", "gemini"):
            ctx = AppContext({"AI_FILTER": {**base, "PROVIDER": provider},
                              "RSS": {"FRESHNESS_FILTER": {"ENABLED": False}}})
            with redirect_stdout(io.StringIO()):
                hotlist, _, _ = ctx.convert_ai_filter_to_report_data(result)
            kept[provider] = sum(len(s["titles"]) for s in hotlist)
        self.assertEqual(kept, {"jev": 0, "gemini": 1})


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
