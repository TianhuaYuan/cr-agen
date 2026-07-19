"""LLM Chat 客户端单例（从 1 号项目 rag/clients.py 抽出的极简版）。

只保留 Chat 客户端：Worker 与 Supervisor 复用它调 LLM。
设计要点：单例 + 依赖注入友好——测试时可向调用方传入 fake client，免真实 API key。
"""
import logging

import httpx
from openai import AsyncOpenAI

from .config import settings

logger = logging.getLogger(__name__)

# 单次 LLM 调用超时（秒）：单一来源 config.LLM_TIMEOUT（默认 120）。
# 客户端默认即与 Worker 权威超时一致，避免两层超时语义打架。
_chat_client: AsyncOpenAI | None = None


def get_chat_client() -> AsyncOpenAI:
    """返回（惰性创建）Chat 客户端单例。

    注意：显式传入 ``http_client=httpx.AsyncClient(trust_env=False)`` 禁用
    环境变量代理（HTTP_PROXY/HTTPS_PROXY）。原因：开发机常驻系统代理（如
    Clash/V2Ray）可能未运行或拒绝转发到 LLM API 域名，导致 ``Connection error``。
    直连更稳定，LLM API 本身走 HTTPS 已经安全。
    """
    global _chat_client
    if _chat_client is None:
        _chat_client = AsyncOpenAI(
            api_key=settings.CHAT_API_KEY,
            base_url=settings.CHAT_BASE_URL,
            timeout=settings.LLM_TIMEOUT,
            http_client=httpx.AsyncClient(trust_env=False),
        )
    return _chat_client


def reset_chat_client() -> None:
    """清空单例（测试用，便于隔离）。"""
    global _chat_client
    _chat_client = None
