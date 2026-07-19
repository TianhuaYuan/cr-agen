"""CLI --pr 参数测试（Phase 9: Task 9.2）。

TDD Red → Green：
- 先写测试（Red）：backend.cli.main.cmd_review_pr 不存在 → AttributeError。
- 再写实现（Green）：让测试通过。

测试策略：
- 注入 _FakeHTTPClient 模拟 GitHub API（来自 test_github.py）。
- 注入 _SmartFakeClient 模拟 LLM（来自 test_mcp.py 模式），让 graph 走完整链路。
- 验证 --pr 模式端到端：拉 patch → 解析 → graph 审查 → 返回报告。
"""
import json
from types import SimpleNamespace

import pytest

from backend.integrations.github import GitHubClient
from backend.tests.test_github import _FakeHTTPClient, _SAMPLE_PATCH


class _SmartFakeClient:
    """根据 prompt 内容返回不同 JSON 的假客户端（复用 W1 E2E / MCP 模式）。"""

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
    client = _SmartFakeClient()
    monkeypatch.setattr(
        "backend.core.llm.get_chat_client", lambda: client
    )
    return client


_PR_URL = "https://github.com/octocat/hello-world/pull/1"


def test_cli_review_pr_end_to_end(monkeypatch):
    """--pr 模式：GitHubClient 拉 patch → 解析 → graph 审查 → 返回非空报告。"""
    _patch_llm(monkeypatch)
    fake_gh = GitHubClient(http_client=_FakeHTTPClient(_SAMPLE_PATCH))
    from backend.cli.main import cmd_review_pr

    report = cmd_review_pr(_PR_URL, gh_client=fake_gh)
    assert isinstance(report, str)
    assert len(report) > 0
    # 报告应是 Markdown（含标题级内容）
    assert "#" in report


def test_cli_review_pr_uses_pr_code(monkeypatch):
    """PR 代码片段应进入审查结果（SmartFakeClient 的硬编码密钥 finding 命中）。"""
    _patch_llm(monkeypatch)
    fake_gh = GitHubClient(http_client=_FakeHTTPClient(_SAMPLE_PATCH))
    from backend.cli.main import cmd_review_pr

    report = cmd_review_pr(_PR_URL, gh_client=fake_gh)
    # SmartFakeClient 返回「硬编码 API 密钥」finding，报告应含对应描述
    assert "硬编码" in report or "密钥" in report or "API" in report


def test_main_review_pr_argparse(monkeypatch, capsys):
    """main() 能解析 --pr 参数并输出报告（argparse 互斥组路径）。"""
    _patch_llm(monkeypatch)
    fake_gh = GitHubClient(http_client=_FakeHTTPClient(_SAMPLE_PATCH))
    import backend.integrations.github as gh_mod

    # 让 cmd_review_pr 内部 new GitHubClient() 返回我们的 fake
    monkeypatch.setattr(gh_mod, "GitHubClient", lambda *a, **k: fake_gh)

    from backend.cli.main import main

    main(["review", "--pr", _PR_URL])
    captured = capsys.readouterr()
    assert len(captured.out) > 0
    assert "#" in captured.out


def test_main_review_requires_file_or_pr():
    """review 子命令必须传 --file 或 --pr 之一（互斥组 required）。"""
    from backend.cli.main import main

    with pytest.raises(SystemExit):
        main(["review"])


# ── --file 安全校验（P0：路径穿越 + 密钥文件读）──────────────────────

def test_resolve_review_file_rejects_outside_project():
    """--file 逃出项目根（如 ../../etc/passwd）→ 必须拒绝（防任意文件读）。"""
    from backend.cli.main import _resolve_review_file

    with pytest.raises(ValueError):
        _resolve_review_file("../../etc/passwd")


def test_resolve_review_file_rejects_secret_files():
    """--file 指向密钥文件（.env / 私钥 / 证书）→ 必须拒绝（防密钥送进 LLM）。"""
    from backend.cli.main import _resolve_review_file

    for name in ("backend/.env", ".env.prod", "id_rsa", "key.pem", "tls.key"):
        with pytest.raises(ValueError):
            _resolve_review_file(name)


def test_resolve_review_file_allows_sample():
    """--file 指向项目内正常代码样本（backend/tests/samples/sample_bad_python.py）→ 返回绝对路径。"""
    from backend.cli.main import _resolve_review_file

    path = _resolve_review_file("backend/tests/samples/sample_bad_python.py")
    assert path.is_absolute()
    assert path.name == "sample_bad_python.py"
