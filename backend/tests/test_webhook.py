"""GitHub Webhook 测试（Phase 9: Task 9.3）。

TDD Red → Green：
- 先写测试（Red）：backend.api.webhooks 不存在 → import/404 失败。
- 再写实现（Green）：让测试通过。

测试策略：
- 注入 GitHubClient（monkeypatch github 模块构造器）模拟 GitHub API，不触网。
- 注入 SmartFakeClient 模拟 LLM，graph 走完整链路。
- 验证 X-Hub-Signature-256 签名校验 + pull_request opened 事件 → 返回审查报告。
"""
import hashlib
import hmac
import json
from types import SimpleNamespace

import httpx
import pytest
from httpx import ASGITransport

from backend.integrations.github import GitHubClient
from backend.tests.test_github import _FakeHTTPClient, _SAMPLE_PATCH


class _SmartFakeClient:
    """根据 prompt 内容返回不同 JSON 的假客户端（复用 W1 E2E / MCP / CLI 模式）。"""

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
    client = _SmartFakeClient()
    monkeypatch.setattr("backend.core.llm.get_chat_client", lambda: client)
    return client


def _sign(secret: str, body: bytes) -> str:
    """计算 GitHub webhook 签名 sha256=<hmac>。"""
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


_PAYLOAD = {
    "action": "opened",
    "pull_request": {"html_url": "https://github.com/octocat/hello-world/pull/1"},
}


def _make_app_with_mock_gh(monkeypatch, webhook_secret=""):
    """构造 app，并把 GitHubClient 构造器替换为注入 fake http 的实例。"""
    _patch_llm(monkeypatch)
    import backend.integrations.github as gh_mod

    fake_gh = GitHubClient(http_client=_FakeHTTPClient(_SAMPLE_PATCH))
    monkeypatch.setattr(gh_mod, "GitHubClient", lambda *a, **k: fake_gh)

    if webhook_secret:
        from backend.core.config import settings
        monkeypatch.setattr(settings, "GITHUB_WEBHOOK_SECRET", webhook_secret)

    from backend.main import create_app

    return create_app()


@pytest.mark.asyncio
async def test_webhook_reviews_pr(monkeypatch):
    """pull_request opened + 合法签名 → 拉 diff → 审查 → 返回报告。"""
    secret = "test-secret"
    monkeypatch.setenv("CR_AGENT_GITHUB_WEBHOOK_SECRET", secret)
    app = _make_app_with_mock_gh(monkeypatch, webhook_secret=secret)

    transport = ASGITransport(app=app)
    body = json.dumps(_PAYLOAD).encode()
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/webhooks/github",
            content=body,
            headers={
                "X-Hub-Signature-256": _sign(secret, body),
                "X-GitHub-Event": "pull_request",
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "reviewed"
    assert "report" in data and len(data["report"]) > 0


@pytest.mark.asyncio
async def test_webhook_invalid_signature(monkeypatch):
    """非法签名 → 401。"""
    monkeypatch.setenv("CR_AGENT_GITHUB_WEBHOOK_SECRET", "test-secret")
    app = _make_app_with_mock_gh(monkeypatch, webhook_secret="test-secret")

    transport = ASGITransport(app=app)
    body = json.dumps(_PAYLOAD).encode()
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/webhooks/github",
            content=body,
            headers={
                "X-Hub-Signature-256": "sha256=deadbeef",
                "X-GitHub-Event": "pull_request",
            },
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_webhook_ignores_non_pr_event(monkeypatch):
    """非 pull_request 事件 → 忽略（status=ignored）。"""
    secret = "test-secret"
    monkeypatch.setenv("CR_AGENT_GITHUB_WEBHOOK_SECRET", secret)
    app = _make_app_with_mock_gh(monkeypatch, webhook_secret=secret)

    transport = ASGITransport(app=app)
    body = json.dumps(_PAYLOAD).encode()
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/webhooks/github",
            content=body,
            headers={
                "X-Hub-Signature-256": _sign(secret, body),
                "X-GitHub-Event": "push",
            },
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ignored"


def test_verify_signature_logic():
    """_verify_signature 单测：合法/非法/无签名/空 secret 跳过。"""
    from backend.api.webhooks import _verify_signature

    secret, body = "s", b"payload"
    sig = _sign(secret, body)
    assert _verify_signature(secret, body, sig) is True
    assert _verify_signature(secret, body, "sha256=wrong") is False
    assert _verify_signature(secret, body, None) is False
    # 未配置 secret 则跳过验证（开发环境）
    assert _verify_signature("", body, None) is True


@pytest.mark.asyncio
async def test_webhook_rejects_oversized_body(monkeypatch):
    """DoS 加固：未授权超大 payload 应在验签前拒绝（413），而非读完整个 body 才因签名错误返回 401。

    修复前：handler 先 `await request.body()` 无上限缓冲，再验签 → 攻击者可耗尽内存/CPU。
    """
    monkeypatch.setenv("CR_AGENT_GITHUB_WEBHOOK_SECRET", "test-secret")
    app = _make_app_with_mock_gh(monkeypatch)

    transport = ASGITransport(app=app)
    # 2MB 真实 body（超过 1MB 上限），带非法签名（模拟未授权攻击者）
    big_body = b"x" * (2 * 1024 * 1024)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/webhooks/github",
            content=big_body,
            headers={
                "X-Hub-Signature-256": "sha256=deadbeef",
                "X-GitHub-Event": "pull_request",
            },
        )
    assert resp.status_code == 413


@pytest.mark.asyncio
async def test_webhook_rejects_when_secret_required_but_missing(monkeypatch):
    """生产环境 WEBHOOK_SECRET_REQUIRED=True 但 secret 未配置 → 未授权请求应被拒（401）。

    修复前：``if not secret: return True`` 使空 secret 即免鉴权，任何请求都放行（免签 + 可烧 token）。
    """
    from backend.core.config import settings

    monkeypatch.setattr(settings, "WEBHOOK_SECRET_REQUIRED", True)
    monkeypatch.setenv("CR_AGENT_GITHUB_WEBHOOK_SECRET", "")  # 确保 secret 为空
    app = _make_app_with_mock_gh(monkeypatch)

    transport = ASGITransport(app=app)
    body = json.dumps(_PAYLOAD).encode()
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/webhooks/github",
            content=body,
            headers={"X-GitHub-Event": "pull_request"},  # 无签名头
        )
    assert resp.status_code == 401
