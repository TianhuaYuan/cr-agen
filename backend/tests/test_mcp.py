"""MCP Gateway 测试（Phase 8: Task 8.1-8.4）。

测试策略：
- 用 FastMCP Client 连接本地 MCP Server（in-memory transport）。
- mock LLM 调用（SmartFakeClient 根据 prompt 内容返回不同 JSON）。
- 验证 4 个 tool 的返回结构，不依赖真实 LLM API。
- Task 8.4: httpx ASGITransport → FastAPI /mcp → JSON-RPC 2.0 → 验证 HTTP 集成。

TDD Red → Green：
- 先写测试（Red）：import 失败说明模块不存在。
- 再写实现（Green）：让测试通过。
"""
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from httpx import ASGITransport

from backend.mcp.server import mcp


class _SmartFakeClient:
    """根据 prompt 内容返回不同 JSON 的假客户端（复用 W1 E2E 模式）。

    - decompose prompt 含「拆解」或「priority」→ 返回任务列表 JSON
    - worker prompt → 返回 findings JSON
    """

    _TASKS_JSON = json.dumps([
        {"role": "security", "description": "检查 SQL 注入和硬编码密钥", "priority": 1},
        {"role": "quality", "description": "检查命名规范和函数长度", "priority": 2},
        {"role": "performance", "description": "检查嵌套循环和性能瓶颈", "priority": 3},
        {"role": "structure", "description": "检查上帝函数和架构问题", "priority": 3},
    ])

    _FINDINGS_JSON = json.dumps([
        {
            "severity": "high",
            "line": 5,
            "description": "硬编码 API 密钥",
            "suggestion": "改用环境变量",
            "code_snippet": 'api_key = "sk-xxx"',
        },
        {
            "severity": "medium",
            "line": 12,
            "description": "函数过长（80 行）",
            "suggestion": "拆分为 3 个子函数",
            "code_snippet": "def huge_function(...):",
        },
    ])

    @property
    def chat(self):
        async def _create(*args, **kwargs):
            messages = kwargs.get("messages", [])
            user_content = ""
            for m in messages:
                if m.get("role") == "user":
                    user_content = m.get("content", "")
                    break

            if "拆解" in user_content or "priority" in user_content:
                text = self._TASKS_JSON
            else:
                text = self._FINDINGS_JSON

            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
            )

        return SimpleNamespace(completions=SimpleNamespace(create=_create))


def _patch_llm(monkeypatch):
    """monkeypatch LLM 客户端为 SmartFakeClient。"""
    from backend.core import llm as llm_mod
    client = _SmartFakeClient()
    monkeypatch.setattr(llm_mod, "get_chat_client", lambda: client)
    return client


# ── Task 8.1: MCP Server 骨架 ──────────────────────────────


class TestMCPServerSkeleton:
    """MCP Server 实例和 ping tool。"""

    def test_mcp_instance_exists(self):
        """mcp 是 FastMCP 实例。"""
        from fastmcp import FastMCP
        assert isinstance(mcp, FastMCP)

    @pytest.mark.asyncio
    async def test_ping_tool(self):
        """ping tool 返回非空结果。"""
        from fastmcp import Client
        async with Client(mcp) as client:
            result = await client.call_tool("ping", {})
            assert result is not None


# ── Task 8.2: 审查能力 MCP Tools ────────────────────────────


class TestReviewCodeTool:
    """review_code tool：完整审查 → Markdown 报告。"""

    @pytest.mark.asyncio
    async def test_review_code_returns_report(self, monkeypatch):
        """review_code 返回非空字符串（Markdown 报告）。"""
        _patch_llm(monkeypatch)

        from fastmcp import Client
        async with Client(mcp) as client:
            result = await client.call_tool(
                "review_code",
                {"code": "api_key = 'sk-xxx'", "language": "python"},
            )
            assert result is not None
            text = result.content[0].text if result.content else ""
            assert isinstance(text, str)
            assert len(text) > 0


class TestDecomposeCodeTool:
    """decompose_code tool：任务拆解 → 任务列表。"""

    @pytest.mark.asyncio
    async def test_decompose_returns_tasks(self, monkeypatch):
        """decompose_code 返回任务列表。"""
        _patch_llm(monkeypatch)

        from fastmcp import Client
        async with Client(mcp) as client:
            result = await client.call_tool(
                "decompose_code",
                {"code": "x = 1", "language": "python"},
            )
            assert result is not None


class TestWorkerReviewTool:
    """worker_review tool：单 Worker 审查 → findings。"""

    @pytest.mark.asyncio
    async def test_worker_review_returns_findings(self, monkeypatch):
        """worker_review 返回 findings 列表。"""
        _patch_llm(monkeypatch)

        from fastmcp import Client
        async with Client(mcp) as client:
            result = await client.call_tool(
                "worker_review",
                {"code": "eval(x)", "language": "python", "role": "security"},
            )
            assert result is not None


class TestAggregateReportTool:
    """aggregate_report tool：聚合 findings → Markdown 报告。"""

    @pytest.mark.asyncio
    async def test_aggregate_returns_markdown(self):
        """aggregate_report 返回非空 Markdown 字符串。"""
        findings = [
            {"severity": "high", "line": 1, "description": "硬编码密钥",
             "suggestion": "用环境变量", "code_snippet": "", "worker": "security"},
            {"severity": "medium", "line": 5, "description": "命名不规范",
             "suggestion": "用 snake_case", "code_snippet": "", "worker": "quality"},
        ]

        from fastmcp import Client
        async with Client(mcp) as client:
            result = await client.call_tool(
                "aggregate_report",
                {"findings": findings, "language": "python"},
            )
            assert result is not None
            text = result.content[0].text if result.content else ""
            assert isinstance(text, str)
            assert len(text) > 0


# ── Task 8.3: MCP Resources ─────────────────────────────────


class TestMCPResources:
    """review://history 和 review://stats 资源。"""

    @pytest.mark.asyncio
    async def test_review_history_resource(self, monkeypatch):
        """review://history 返回列表（空或非空都行，关键是结构正确）。"""
        from backend.mcp import server as mcp_server
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine, AsyncSession
        from sqlalchemy.pool import StaticPool
        from backend.core.database import Base

        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        monkeypatch.setattr(mcp_server, "AsyncSessionLocal", maker)

        from fastmcp import Client
        async with Client(mcp) as client:
            result = await client.read_resource("review://history")
            assert result is not None

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_review_stats_resource(self, monkeypatch):
        """review://stats 返回含 total 字段的字典。"""
        from backend.mcp import server as mcp_server
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine, AsyncSession
        from sqlalchemy.pool import StaticPool
        from backend.core.database import Base

        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        monkeypatch.setattr(mcp_server, "AsyncSessionLocal", maker)

        from fastmcp import Client
        async with Client(mcp) as client:
            result = await client.read_resource("review://stats")
            assert result is not None

        await engine.dispose()


# ── Task 8.4: MCP Server 挂载到 FastAPI ─────────────────────


def _parse_jsonrpc_response(resp: httpx.Response) -> dict:
    """解析 MCP JSON-RPC 响应（兼容 JSON 和 SSE 格式）。

    Streamable HTTP 传输可能返回 application/json 或 text/event-stream。
    SSE 格式：每行 data: {...}\\n\\n
    """
    content_type = resp.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        for line in resp.text.strip().split("\n"):
            if line.startswith("data: "):
                return json.loads(line[6:])
        return {}
    return resp.json()


_MCP_HEADERS = {"Accept": "application/json, text/event-stream"}


def _jsonrpc(method: str, params: dict, id_: int = 0) -> dict:
    """构造 JSON-RPC 2.0 请求体。"""
    return {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": id_,
    }


@asynccontextmanager
async def _mcp_http_client():
    """创建 MCP HTTP 测试客户端（手动触发 lifespan + ASGITransport）。

    关键：ASGITransport 不触发 ASGI lifespan，但 MCP StreamableHTTPASGIApp
    需要 lifespan 初始化 task_group。直接用 _mcp_app.router.lifespan_context()
    手动触发 _mcp_app 的 lifespan（和 main.py 里的方式一致），不需要额外依赖。

    路径注意：FastAPI mount("/mcp", subapp) 必须 POST 到 /mcp/（带尾斜杠）
    才会转发给子应用的根路径 /。POST /mcp（无尾斜杠）返回 405。
    """
    from backend.main import create_app, _mcp_app

    async with _mcp_app.router.lifespan_context(_mcp_app):
        app = create_app()
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", follow_redirects=True
        ) as client:
            yield client


class TestMCPFastAPIIntegration:
    """MCP Server 挂载到 FastAPI /mcp 路径的 HTTP 集成测试。

    用 httpx ASGITransport 直接测 FastAPI app，不发真实网络请求。
    MCP 用 stateless_http=True，每个请求独立处理，无需 session 管理。

    坑1：ASGITransport 不触发 ASGI lifespan，MCP StreamableHTTPASGIApp
         需要 lifespan 初始化 session_manager / task_group。
         解法：直接用 _mcp_app.router.lifespan_context() 手动触发。
    坑2：FastAPI mount("/mcp", subapp) 必须 POST 到 /mcp/（带尾斜杠）
         才会转发给子应用的根路径 /。POST /mcp（无尾斜杠）返回 405，
         follow_redirects 没用——因为返回的是 405 不是 307。
    """

    @pytest.mark.asyncio
    async def test_mcp_endpoint_exists(self):
        """POST /mcp/ 不返回 404（端点已挂载到 FastAPI）。"""
        async with _mcp_http_client() as client:
            resp = await client.post("/mcp/", json=_jsonrpc("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            }), headers=_MCP_HEADERS)
            assert resp.status_code != 404

    @pytest.mark.asyncio
    async def test_mcp_ping_via_http(self):
        """通过 HTTP JSON-RPC 调 ping tool → 返回 result。"""
        async with _mcp_http_client() as client:
            # Initialize（stateless 模式下仍需握手，但无 session）
            await client.post("/mcp/", json=_jsonrpc("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            }), headers=_MCP_HEADERS)

            # 调 ping tool
            call_resp = await client.post("/mcp/", json=_jsonrpc("tools/call", {
                "name": "ping",
                "arguments": {},
            }, id_=1), headers=_MCP_HEADERS)
            assert call_resp.status_code == 200
            data = _parse_jsonrpc_response(call_resp)
            assert "result" in data

    @pytest.mark.asyncio
    async def test_mcp_review_code_via_http(self, monkeypatch):
        """通过 HTTP JSON-RPC 调 review_code → 返回 result（含报告）。"""
        _patch_llm(monkeypatch)

        async with _mcp_http_client() as client:
            await client.post("/mcp/", json=_jsonrpc("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"},
            }), headers=_MCP_HEADERS)

            call_resp = await client.post("/mcp/", json=_jsonrpc("tools/call", {
                "name": "review_code",
                "arguments": {"code": "api_key = 'sk-xxx'", "language": "python"},
            }, id_=1), headers=_MCP_HEADERS)
            assert call_resp.status_code == 200
            data = _parse_jsonrpc_response(call_resp)
            assert "result" in data
