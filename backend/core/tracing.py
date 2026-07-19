"""Langfuse 链路追踪封装（Phase 14: Task 14.1）。

设计要点：
1. **抽象 Span / Tracer 接口**：统一 NoOp 和 Langfuse 两个 backend
2. **NoOp 默认降级**：未配置或 SDK 不可用时返回 NoOpTracer，所有方法 no-op
3. **Langfuse 可选依赖**：try import langfuse，失败则 _LANGFUSE_AVAILABLE=False
4. **Span 支持上下文管理**：`with tracer.start_span("name") as span: ...` 自动 end

使用方式：
    from backend.core.tracing import get_tracer
    tracer = get_tracer()
    with tracer.start_span("llm_call", metadata={"model": "gpt-4o"}) as span:
        # 业务逻辑
        span.update({"tokens": 150})
        # 退出 with 时自动 end

未配置 LANGFUSE_PUBLIC_KEY 时，get_tracer() 返回 NoOpTracer，
所有 span 操作都是 no-op，对业务零侵入。
"""
import logging
from typing import Any

from .config import settings

logger = logging.getLogger(__name__)

# ── Langfuse SDK 可选依赖检测 ──
try:
    from langfuse import Langfuse as _LangfuseClientClass  # type: ignore
    _LANGFUSE_AVAILABLE = True
except ImportError:
    _LangfuseClientClass = None  # type: ignore
    _LANGFUSE_AVAILABLE = False


class Span:
    """统一 Span 接口，支持 with 上下文管理。

    NoOp 模式下只是个数据容器；Langfuse 模式下包装底层 langfuse span。
    """

    def __init__(self, name: str, metadata: dict | None = None,
                 _backend: Any = None):
        self.name = name
        self.metadata: dict = dict(metadata) if metadata else {}
        self._ended = False
        self._backend = _backend  # langfuse span（如果有）

    def update(self, metadata: dict | None = None) -> None:
        """合并 metadata 到 span（不覆盖已有键的语义由调用方控制）。"""
        if metadata:
            self.metadata.update(metadata)
        if self._backend is not None and hasattr(self._backend, "update"):
            try:
                self._backend.update(metadata=metadata) if metadata else None
            except Exception as exc:
                logger.debug("Span.update backend failed: %s", exc)

    def end(self, metadata: dict | None = None) -> None:
        """结束 span，可选合并最终 metadata。"""
        if self._ended:
            return
        self.update(metadata)
        self._ended = True
        if self._backend is not None and hasattr(self._backend, "end"):
            try:
                self._backend.end()
            except Exception as exc:
                logger.debug("Span.end backend failed: %s", exc)

    def __enter__(self) -> "Span":
        return self

    def __exit__(self, *args) -> None:
        self.end()


class NoOpTracer:
    """NoOp tracer：所有方法 no-op，对业务零侵入。

    未配置 Langfuse 或 SDK 不可用时使用。
    """

    def start_span(self, name: str, metadata: dict | None = None) -> Span:
        return Span(name, metadata=metadata)


class LangfuseTracer:
    """Langfuse tracer：包装 langfuse client，真实追踪。

    需要 langfuse SDK 已安装 + LANGFUSE_PUBLIC_KEY / SECRET_KEY 已配置。
    """

    def __init__(self, client: Any):
        self.client = client

    def start_span(self, name: str, metadata: dict | None = None) -> Span:
        """创建 Langfuse trace，返回包装后的 Span。

        注意：Langfuse 的 trace() 返回一个 trace 对象，我们包装成统一 Span。
        若 backend 调用失败，降级为 NoOp Span（不抛异常，保证业务不中断）。
        """
        backend_span = None
        try:
            if hasattr(self.client, "trace"):
                backend_span = self.client.trace(name=name, metadata=metadata)
        except Exception as exc:
            logger.debug("Langfuse trace failed, fallback to NoOp span: %s", exc)
        return Span(name, metadata=metadata, _backend=backend_span)


# ── 模块级单例 ──
_tracer: Any = None


def get_tracer() -> Any:
    """返回 tracer 单例（惰性创建）。

    决策逻辑：
    1. settings.LANGFUSE_PUBLIC_KEY 未配置 → NoOpTracer
    2. 配置了但 _LANGFUSE_AVAILABLE=False（SDK 未装）→ NoOpTracer + warning 日志
    3. 配置 + SDK 可用 → LangfuseTracer
    """
    global _tracer
    if _tracer is not None:
        return _tracer

    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        _tracer = NoOpTracer()
        return _tracer

    if not _LANGFUSE_AVAILABLE or _LangfuseClientClass is None:
        logger.warning(
            "Langfuse 配置存在但 SDK 未安装，降级为 NoOp tracer。"
            "请 pip install langfuse 后重启服务。"
        )
        _tracer = NoOpTracer()
        return _tracer

    try:
        client = _LangfuseClientClass(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
        _tracer = LangfuseTracer(client)
        logger.info("Langfuse tracer 已启用（host=%s）", settings.LANGFUSE_HOST)
    except Exception as exc:
        logger.warning("Langfuse 客户端初始化失败，降级 NoOp: %s", exc)
        _tracer = NoOpTracer()

    return _tracer


def reset_tracer() -> None:
    """清空单例（测试用，便于隔离）。"""
    global _tracer
    _tracer = None
