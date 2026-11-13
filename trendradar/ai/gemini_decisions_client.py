# coding=utf-8
"""Gemini Decisions API 的标准库客户端与 TPM 速率控制。"""

import json
import math
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Optional, Tuple

from trendradar.ai.jev_client import estimate_tokens

API_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/decisions"
MODEL_NAME = "gemini-decision-1"
TOKEN_ESTIMATE_SAFETY_FACTOR = 1.25


class GeminiDecisionAPIError(RuntimeError):
    """Google Decisions API 错误，保留限流退避信息但不包含请求密钥。"""

    def __init__(self, status: int, detail: str, retry_after: Optional[float] = None):
        self.status = status
        self.retry_after = retry_after
        super().__init__(f"Gemini Decisions API HTTP {status}: {detail}")


class TokenRateLimiter:
    """用保守 token 估算平滑分配每分钟输入额度。"""

    def __init__(self, tokens_per_minute: int):
        if tokens_per_minute <= 0:
            raise ValueError("Gemini TPM 上限必须大于 0")
        self.tokens_per_minute = tokens_per_minute
        # 任意 60 秒内最多消耗「30% 突发 + 70% 补充」的配置预算。
        self.burst_capacity = max(1, int(tokens_per_minute * 0.3))
        self.refill_tokens_per_minute = tokens_per_minute * 0.7
        self._available_tokens = float(self.burst_capacity)
        self._updated_at = time.monotonic()
        self._blocked_until = 0.0

    def wait_for_capacity(self, estimated_tokens: int) -> Dict[str, Any]:
        reserved_tokens = math.ceil(estimated_tokens * TOKEN_ESTIMATE_SAFETY_FACTOR)
        if reserved_tokens > self.burst_capacity:
            raise ValueError(
                f"单个 Gemini 请求估算需要 {reserved_tokens} 个预算 tokens，"
                f"超过单请求突发容量 {self.burst_capacity}；请降低每批新闻数"
            )

        while True:
            now = time.monotonic()
            elapsed = now - self._updated_at
            self._available_tokens = min(
                float(self.burst_capacity),
                self._available_tokens + elapsed * self.refill_tokens_per_minute / 60,
            )
            self._updated_at = now

            blocked_wait = max(0.0, self._blocked_until - now)
            token_wait = (
                max(0.0, reserved_tokens - self._available_tokens)
                * 60 / self.refill_tokens_per_minute
            )
            wait_seconds = max(blocked_wait, token_wait)
            if wait_seconds <= 0:
                self._available_tokens -= reserved_tokens
                return {"reserved_tokens": reserved_tokens}

            print(f"[AI筛选][Gemini] TPM 速率控制，等待 {wait_seconds:.1f} 秒")
            time.sleep(wait_seconds)

    def reconcile(self, reservation: Dict[str, Any], actual_tokens: int) -> None:
        """用 Google 返回的真实 usage 修正后续请求的 token 预算。"""
        if actual_tokens < 0:
            return
        now = time.monotonic()
        elapsed = now - self._updated_at
        self._available_tokens = min(
            float(self.burst_capacity),
            self._available_tokens + elapsed * self.refill_tokens_per_minute / 60,
        )
        self._updated_at = now
        self._available_tokens = min(
            float(self.burst_capacity),
            self._available_tokens + reservation["reserved_tokens"] - actual_tokens,
        )

    def defer(self, seconds: float) -> None:
        if seconds > 0:
            self._blocked_until = max(self._blocked_until, time.monotonic() + seconds)


class GeminiDecisionClient:
    """使用 Gemini Decisions REST API，不引入额外 SDK 依赖。"""

    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get("API_KEY", "")
        self.timeout = config.get("TIMEOUT", 60)
        self.tokens_per_minute = config.get("TOKENS_PER_MINUTE", 35000)
        self.rate_limiter = TokenRateLimiter(self.tokens_per_minute)

    def estimate_request_tokens(self, input_text: str, questions: Dict[str, Dict]) -> int:
        """估算完整序列化请求大小；真实用量以 Google usage 响应为准。"""
        payload = {
            "model": MODEL_NAME,
            "input": input_text,
            "questions": questions,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return estimate_tokens(body.decode("utf-8"))

    def decide(
        self,
        input_text: str,
        questions: Dict[str, Dict],
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        question_counts = len(questions)
        if question_counts > 1000:
            raise ValueError("Gemini Decisions API 每个请求最多接受 1000 个问题")
        payload = {
            "model": MODEL_NAME,
            "input": input_text,
            "questions": questions,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        estimated_tokens = self.estimate_request_tokens(input_text, questions)
        reservation = self.rate_limiter.wait_for_capacity(estimated_tokens)
        request = urllib.request.Request(
            API_ENDPOINT,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "x-goog-api-key": self.api_key,
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            retry_after = _parse_retry_after(error.headers.get("Retry-After"))
            if error.code == 429 and retry_after is None:
                retry_after = 60.0
            if retry_after is not None:
                self.rate_limiter.defer(retry_after)
            detail = error.read().decode("utf-8", "replace")[:300]
            raise GeminiDecisionAPIError(error.code, detail, retry_after) from error

        usage = result.get("usage", {}) if isinstance(result, dict) else {}
        actual_tokens = usage.get("total_input_tokens") if isinstance(usage, dict) else None
        try:
            actual_tokens = int(actual_tokens)
        except (TypeError, ValueError):
            actual_tokens = None
        if actual_tokens is not None:
            self.rate_limiter.reconcile(reservation, actual_tokens)

        return result

    def probe(self, timeout: float) -> Tuple[bool, str]:
        """发送一个最小决策，验证 key 与 Decisions API 均可用。"""
        ok, error = self.validate_config()
        if not ok:
            return False, error

        try:
            response = self.decide(
                "Google 于 2026 年发布 Gemini Decision 1。",
                {
                    "probe": {
                        "type": "noul",
                        "instructions": "这句话是否提到了 Gemini Decision 1？",
                    }
                },
                timeout=timeout,
            )
        except Exception as error:
            return False, f"{type(error).__name__}: {error}"

        answers = response.get("answers") if isinstance(response, dict) else None
        if not isinstance(answers, dict) or "probe" not in answers:
            return False, "响应缺少 answers.probe"
        return True, ""

    def validate_config(self) -> Tuple[bool, str]:
        if not self.api_key:
            return False, "未配置 Gemini API Key（AI_FILTER_GEMINI_API_KEY）"
        return True, ""


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None
