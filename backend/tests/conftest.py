"""pytest 共享 fixtures（Task 1.4）。

- ``async_session``：每个测试一个独立的内存 SQLite 引擎 + 建表，测试间零污染。
  不碰 ``backend.core.database`` 的模块级引擎（那是运行时用的文件库）。
- ``mock_llm``：把 ``get_chat_client`` 替换成假客户端，Worker/Supervisor 测试免真实 API key。
"""
import sys
from types import SimpleNamespace

import pytest
import pytest_asyncio


def _install_fake_fastmcp():
    """如果环境里没有完整的 fastmcp（仅安装了 slim 版本），就伪造假模块绕过 import。

    为什么有这个：沙箱/CI 环境可能只装了 fastmcp-slim（缺少 server 支持），
    但 backend/main.py 是模块级 import（from backend.mcp.server import mcp），
    只要 import main.py 就会触发 fastmcp.server import 失败。
    这里在 sys.modules 里注册假的 fastmcp 模块，提供 FastMCP 的最小实现。

    策略：先卸载已存在的 fastmcp（可能是 slim 版本，缺少 server），再安装假的。
    """
    for key in list(sys.modules.keys()):
        if key.startswith("fastmcp"):
            del sys.modules[key]

    try:
        from fastmcp import FastMCP  # noqa: F401
        return  # 真实 fastmcp 完整可用
    except ImportError:
        pass  # 没有完整的，继续装假的

    from types import SimpleNamespace as _SN

    class _FakeFastMCP:
        def __init__(self, name: str = "fake"):
            self.name = name
            self._tools = {}
            self._resources = {}

        def tool(self):
            def decorator(fn):
                self._tools[fn.__name__] = fn
                return fn
            return decorator

        def resource(self, uri: str):
            def decorator(fn):
                self._resources[uri] = fn
                return fn
            return decorator

        def http_app(self, **kwargs):
            from starlette.applications import Starlette
            from starlette.responses import JSONResponse
            from starlette.routing import Route

            async def _sse(request):
                return JSONResponse({"fake": True})

            return Starlette(routes=[Route("/sse", _sse)])

    fake_module = _SN(FastMCP=_FakeFastMCP)
    sys.modules["fastmcp"] = fake_module
    sys.modules["fastmcp.server"] = fake_module
    sys.modules["fastmcp.tools"] = fake_module


_install_fake_fastmcp()
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from backend.core.database import Base


class _FakeChatClient:
    """极简假 Chat 客户端：只实现 Worker 用到的 ``chat.completions.create``。"""

    def __init__(self, response_text: str = "{}"):
        self._text = response_text

    @property
    def chat(self) -> SimpleNamespace:
        return SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    async def _create(self, *args, **kwargs) -> SimpleNamespace:
        # 返回结构对齐 openai SDK：resp.choices[0].message.content
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=self._text))
            ]
        )


@pytest_asyncio.fixture
async def async_session():
    """独立内存库 + 建表，测试结束销毁引擎。"""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,  # 内存库必须单连接，否则各连接看不到对方建的表
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        yield session

    await engine.dispose()


@pytest.fixture
def mock_llm(monkeypatch):
    """把 get_chat_client 换成假客户端，返回可控的 response_text。"""
    client = _FakeChatClient(response_text='{"findings": []}')
    monkeypatch.setattr(
        "backend.core.llm.get_chat_client", lambda: client
    )
    return client


@pytest.fixture
def fake_llm_factory():
    """工厂 fixture：返回可定制 response_text 的假客户端构造器。

    decompose / worker 测试需要让 LLM 返回特定 JSON（成功形态或非法形态），
    用工厂造一个再 monkeypatch 进去，比改全局 mock_llm 灵活。
    """

    def _make(response_text: str) -> _FakeChatClient:
        return _FakeChatClient(response_text=response_text)

    return _make
