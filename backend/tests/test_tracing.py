"""Langfuse tracing 模块测试（Phase 14: Task 14.1）。

TDD Red → Green：
- 先写测试（Red）：backend.core.tracing 不存在 → import 失败。
- 再写实现（Green）：让测试通过。

测试策略：
- NoOp tracer：所有方法 no-op，但 Span 支持 with 上下文管理
- get_tracer 工厂：未配置 → NoOpTracer；配置 + SDK 可用 → LangfuseTracer
- Span 行为：update / end / metadata 合并
"""
import pytest

from backend.core.tracing import NoOpTracer, Span, get_tracer, reset_tracer


@pytest.fixture(autouse=True)
def _reset_tracer_each_test():
    """每个测试前 reset tracer 单例，避免跨测试泄漏。"""
    reset_tracer()
    yield
    reset_tracer()


class TestNoOpTracer:
    """NoOpTracer 行为测试。"""

    def test_noop_tracer_returns_span(self):
        """start_span 返回 Span 实例。"""
        tracer = NoOpTracer()
        span = tracer.start_span("test_span")
        assert isinstance(span, Span)
        assert span.name == "test_span"

    def test_noop_span_supports_context_manager(self):
        """Span 支持 with 语法，退出时自动 end。"""
        tracer = NoOpTracer()
        with tracer.start_span("test_span") as span:
            assert not span._ended
        assert span._ended

    def test_noop_span_metadata_initial(self):
        """start_span 接受 metadata 并存到 span。"""
        tracer = NoOpTracer()
        span = tracer.start_span("test_span", metadata={"code_len": 100})
        assert span.metadata == {"code_len": 100}

    def test_noop_span_update_merges_metadata(self):
        """span.update 合并 metadata，不覆盖原值。"""
        tracer = NoOpTracer()
        span = tracer.start_span("test_span", metadata={"a": 1})
        span.update({"b": 2})
        assert span.metadata == {"a": 1, "b": 2}

    def test_noop_span_end_with_metadata(self):
        """span.end 接受 metadata，合并后标记 ended。"""
        tracer = NoOpTracer()
        span = tracer.start_span("test_span", metadata={"a": 1})
        span.end({"b": 2})
        assert span._ended
        assert span.metadata == {"a": 1, "b": 2}


class TestGetTracer:
    """get_tracer 工厂函数测试。"""

    def test_get_tracer_default_noop(self):
        """未配置 LANGFUSE_PUBLIC_KEY 时返回 NoOpTracer。"""
        tracer = get_tracer()
        # 默认无配置 → NoOpTracer
        assert isinstance(tracer, NoOpTracer)

    def test_get_tracer_no_langfuse_sdk_returns_noop(self, monkeypatch):
        """配置了 LANGFUSE_PUBLIC_KEY 但 SDK 未安装时返回 NoOpTracer。"""
        from backend.core import tracing

        # 模拟有配置
        monkeypatch.setattr(tracing.settings, "LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setattr(tracing.settings, "LANGFUSE_SECRET_KEY", "sk-test")
        # 模拟 langfuse SDK 不可用
        monkeypatch.setattr(tracing, "_LANGFUSE_AVAILABLE", False)

        tracer = tracing.get_tracer()
        assert isinstance(tracer, NoOpTracer)

    def test_get_tracer_with_langfuse_sdk_returns_langfuse_tracer(self, monkeypatch):
        """配置 + SDK 可用时返回 LangfuseTracer。"""
        from backend.core import tracing

        monkeypatch.setattr(tracing.settings, "LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setattr(tracing.settings, "LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.setattr(tracing.settings, "LANGFUSE_HOST", "http://localhost:3000")
        monkeypatch.setattr(tracing, "_LANGFUSE_AVAILABLE", True)

        # Mock Langfuse 客户端构造
        class _FakeLangfuseClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        monkeypatch.setattr(tracing, "_LangfuseClientClass", _FakeLangfuseClient)

        tracer = tracing.get_tracer()
        assert isinstance(tracer, tracing.LangfuseTracer)
        assert tracer.client.kwargs["public_key"] == "pk-test"
        assert tracer.client.kwargs["secret_key"] == "sk-test"
        assert tracer.client.kwargs["host"] == "http://localhost:3000"


class TestLangfuseTracerInterface:
    """LangfuseTracer 接口测试（用 mock client）。"""

    def test_langfuse_tracer_start_span_returns_span(self, monkeypatch):
        """LangfuseTracer.start_span 返回 Span 实例。"""
        from backend.core import tracing

        class _FakeSpan:
            def __init__(self):
                self.ended = False
                self.metadata = None
            def update(self, metadata=None):
                self.metadata = metadata
            def end(self):
                self.ended = True

        class _FakeClient:
            def trace(self, name=None, metadata=None):
                self.last_name = name
                self.last_metadata = metadata
                return _FakeSpan()

        tracer = tracing.LangfuseTracer(_FakeClient())
        span = tracer.start_span("llm_call", metadata={"model": "gpt-4o-mini"})
        assert isinstance(span, Span)
        assert span.name == "llm_call"
        assert span.metadata == {"model": "gpt-4o-mini"}
