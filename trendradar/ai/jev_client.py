# coding=utf-8
"""
Jev 决策模型客户端

Jev（TypeSafe System One）不是 LLM：不生成文本、不输出 token，直接把概率算到答案字段上。
请求体是 state + questions，与 OpenAI Chat Completions 协议不兼容，
因此不复用 AIClient（它基于 LiteLLM 的 messages 接口，硬套只会触发 UnsupportedParamsError）。
"""

import json
import urllib.error
import urllib.request
from typing import Any, Dict

# Cloudflare 会按 User-Agent 拦截非浏览器请求（error code: 1010），必须显式设置
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# 官方硬限制：state + 最长单个问题 ≤ 32k，state + 全部问题 ≤ 64k
# 取略小的值留安全余量（估算公式实测偏低约 10%）
JEV_SINGLE_LIMIT = 30000
JEV_TOTAL_LIMIT = 62000


def estimate_tokens(text: str) -> int:
    """保守估算 token 数：CJK 按 1 字 1 token，其余按 3 字符 1 token"""
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    other = len(text) - cjk
    return cjk + other // 3 + 1


def normalize_endpoint(url: str) -> str:
    """把配置里的 Jev 地址补全成完整接口地址

    允许三种填法，都归一到 https://host/v1/systemone：
      https://host                 （只填域名）
      https://host/v1              （填到 v1）
      https://host/v1/systemone    （完整地址，原样返回）
    """
    url = (url or "").strip().rstrip("/")
    if not url:
        return ""
    if url.endswith("/systemone"):
        return url
    if url.endswith("/v1"):
        return f"{url}/systemone"
    return f"{url}/v1/systemone"


class JevClient:
    """Jev 决策模型客户端（标准库直连，不经过 LiteLLM）"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 配置字典
                - MODEL: 模型标识（如 jev-latest）
                - API_KEY: API 密钥
                - API_BASE: 接口地址（Jev 是 /v1/systemone 而非 /v1/chat/completions）
                - TIMEOUT: 请求超时时间（秒）
        """
        self.model = config.get("MODEL", "jev-latest")
        self.api_key = config.get("API_KEY", "")
        self.api_base = normalize_endpoint(config.get("API_BASE", ""))
        self.timeout = config.get("TIMEOUT", 300)

    def decide(self, state: str, questions: Dict[str, Dict]) -> Dict[str, Any]:
        """
        发起一次决策请求

        Args:
            state: 所有问题共享的上下文文本
            questions: {问题键名: {"type": "choice"|"score"|"noul", ...}}

        Returns:
            完整响应字典，答案在 response["answers"] 下

        Raises:
            Exception: 网络或接口失败时抛出（调用方据此判定批次失败，不标记已分析）
        """
        payload = {
            "model": self.model,
            "state": state,
            "questions": questions,
        }

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": BROWSER_UA,
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(self.api_base, data=body, method="POST", headers=headers)

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"Jev 请求失败 HTTP {e.code}: {detail}") from e

    def validate_config(self) -> tuple[bool, str]:
        """验证配置是否有效"""
        if not self.model:
            return False, "未配置 Jev 模型（model）"
        if not self.api_base:
            return False, "未配置 Jev 接口地址（api_base）"
        if not self.api_key:
            return False, "未配置 Jev API Key，请在 config.yaml 或环境变量 AI_FILTER_JEV_API_KEY 中设置"
        return True, ""


def split_into_batches(
    items: list,
    build_state,
    build_questions,
    max_items: int,
    single_limit: int = JEV_SINGLE_LIMIT,
    total_limit: int = JEV_TOTAL_LIMIT,
) -> list:
    """
    按 token 估算动态分批

    官方限制是 token 而非条数，且随标签数变化，因此不能只按条数切。
    先按 max_items 粗切，再校验估算 token，超限继续缩批。

    Args:
        items: 待分批的元素列表
        build_state: (batch_items) -> str，构造该批的 state
        build_questions: (batch_items) -> Dict，构造该批的问题
        max_items: 每批元素数上限（token 不超限时的封顶值）

    Returns:
        分批后的元素列表的列表
    """
    batches = []
    current = []

    for item in items:
        trial = current + [item]
        state = build_state(trial)
        questions = build_questions(trial)

        total = estimate_tokens(state) + sum(
            estimate_tokens(json.dumps(q, ensure_ascii=False)) for q in questions.values()
        )
        longest = estimate_tokens(state) + max(
            (estimate_tokens(json.dumps(q, ensure_ascii=False)) for q in questions.values()),
            default=0,
        )

        if current and (len(trial) > max_items or total > total_limit or longest > single_limit):
            batches.append(current)
            current = [item]
        else:
            current = trial

    if current:
        batches.append(current)

    return batches
