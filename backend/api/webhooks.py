"""GitHub Webhook 端点（Phase 9: Task 9.3）。

POST /api/v1/webhooks/github：
- 验证 X-Hub-Signature-256（HMAC-SHA256，防伪造请求）
- 处理 pull_request opened 事件 → 拉 diff → 审查 → 返回报告
- 其他事件/动作返回 ignored（不报错，便于 GitHub 重放测试）

设计选型：
- 模块引用 `github_pkg.GitHubClient`（非 from-import），使测试 monkeypatch 生效
  （W1 踩过的坑：from-import 绑定局部名，monkeypatch 模块属性不生效）。
- 签名验证用 `hmac.compare_digest` 防时序攻击；secret 为空则跳过（开发环境）。
- 仅处理 opened（PR 新建/重开），避免 push 到分支时的重复审查噪音。
"""
import hashlib
import hmac
import json
import logging
import os

from fastapi import APIRouter, Header, HTTPException, Request

from backend.core.common import fetch_pr_code
from backend.core.config import settings
from backend.integrations import github as github_pkg
from backend.services.supervisor.graph import build_supervisor_graph

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webhooks", tags=["webhooks"])

# Webhook body 上限（字节）：防未授权攻击者发送超大 payload 被无上限缓冲耗尽内存/CPU（DoS 加固）。
# 验签前即拒绝超大体型，攻击面最小化。GitHub PR diff 通常远小于此值。
_MAX_WEBHOOK_BODY = 1_000_000


async def _read_body_limited(request: Request, max_bytes: int = _MAX_WEBHOOK_BODY) -> bytes:
    """限流读取 request body：声明大小或实际流超过上限 → 413 拒绝（验签前）。

    相比直接 ``await request.body()``，这里在缓冲前先按 Content-Length 拒超大体型，
    并流式累加期间再次兜底，避免未授权大 payload 被完整读入内存。
    """
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes:
        raise HTTPException(status_code=413, detail="payload too large")
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail="payload too large")
        chunks.append(chunk)
    return b"".join(chunks)


def _verify_signature(secret: str, body: bytes, signature: str | None) -> bool:
    """验证 GitHub webhook 签名。

    - secret 非空：必须签名匹配才通过（hmac.compare_digest 防时序）。
    - secret 为空：开发态默认跳过验签（便利）；但若 ``WEBHOOK_SECRET_REQUIRED=True``
      （生产环境应启用），空 secret 视为配置缺失 → 拒绝请求，避免免鉴权暴露外部触发面。
    """
    if not secret:
        if settings.WEBHOOK_SECRET_REQUIRED:
            logger.warning(
                "Webhook secret 未配置但 WEBHOOK_SECRET_REQUIRED=True，拒绝未授权请求"
            )
            return False
        return True
    if not signature or not signature.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature[len("sha256="):], expected)


@router.post("/github")
async def github_webhook(
    request: Request,
    x_hub_signature_256: str | None = Header(default=None),
    x_github_event: str | None = Header(default=None),
):
    body = await _read_body_limited(request)
    secret = settings.GITHUB_WEBHOOK_SECRET
    if not _verify_signature(secret, body, x_hub_signature_256):
        raise HTTPException(status_code=401, detail="invalid signature")

    event = x_github_event
    if event != "pull_request":
        return {"status": "ignored", "reason": f"event '{event}' not handled"}

    payload = json.loads(body)
    action = payload.get("action")
    if action != "opened":
        return {"status": "ignored", "reason": f"action '{action}' not handled"}

    pr = payload.get("pull_request") or {}
    pr_url = pr.get("html_url")
    if not pr_url:
        raise HTTPException(status_code=400, detail="missing pull_request.html_url")

    code, lang = await fetch_pr_code(pr_url)

    graph = build_supervisor_graph()
    result = await graph.ainvoke({"code": code, "language": lang})
    report = result.get("report", "")
    return {"status": "reviewed", "pr_url": pr_url, "report": report}
