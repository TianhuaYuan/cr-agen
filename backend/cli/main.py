"""CLI 入口（Task 6.2 + Phase 9 Task 9.2）。

python -m backend.cli review --file backend/tests/samples/sample_bad_python.py
python -m backend.cli review --pr https://github.com/owner/repo/pull/1

stdout 输出完整 Markdown 报告，面试演示入口。

--pr 模式（Task 9.2）：从 GitHub PR 拉取 .patch → 解析为代码 → 走与 --file
相同的 supervisor 审查链路。--file 与 --pr 互斥（argparse mutually exclusive group）。
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

from backend.core.languages import EXT_TO_LANGUAGE
from backend.integrations import github as github_pkg
from backend.integrations.github import GitHubClient  # 仅类型注解
from backend.services.supervisor.graph import build_supervisor_graph

# 拒绝读取的密钥/敏感文件名（避免把 .env / 私钥 / 证书等送进 LLM prompt 造成泄露）
_SECRET_FILENAMES = {
    ".env", "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
    ".git-credentials", "credentials", "secrets",
}
_SECRET_SUFFIXES = (".pem", ".key", ".p12", ".keystore", ".jks")


def _resolve_review_file(file: str) -> Path:
    """解析 --file 为绝对路径并做安全校验，返回校验通过的 Path。

    - 必须位于项目仓库根内（防 ``../../etc/passwd`` 之类路径穿越读系统文件）；
    - 拒绝密钥类文件名（.env / 私钥 / 证书等），避免把密钥送进 LLM。
    不合法则抛 ``ValueError``，由 cmd_review 捕获后退出。
    """
    # cli/main.py → backend → 仓库根
    repo_root = Path(__file__).resolve().parents[2]
    path = Path(file).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError:
        raise ValueError(f"--file 必须位于项目目录内，拒绝读取项目外文件：{file}")
    name = path.name
    if (name in _SECRET_FILENAMES
            or name.startswith(".env.")
            or name.endswith(_SECRET_SUFFIXES)):
        raise ValueError(f"--file 拒绝读取密钥/敏感文件：{name}")
    return path


def _detect_language(file_path: str) -> str:
    """从文件扩展名推断语言。"""
    return EXT_TO_LANGUAGE.get(Path(file_path).suffix.lower(), "python")


def cmd_review(file: str, language: str | None = None) -> str:
    """审查本地代码文件，返回 Markdown 报告。"""
    try:
        path = _resolve_review_file(file)
    except ValueError as e:
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)

    if not path.is_file():
        print(f"错误：文件不存在 - {file}", file=sys.stderr)
        sys.exit(1)

    code = path.read_text(encoding="utf-8")
    lang = language or _detect_language(str(path))

    graph = build_supervisor_graph()
    result = asyncio.run(graph.ainvoke({"code": code, "language": lang}))
    return result.get("report", "# 审查完成\n\n未生成报告。")


def cmd_review_pr(pr_url: str, gh_client: GitHubClient | None = None) -> str:
    """从 GitHub PR 拉取代码并审查，返回 Markdown 报告。

    Args:
        pr_url: GitHub PR URL（如 https://github.com/owner/repo/pull/1）。
        gh_client: 可注入的 GitHubClient（测试用），不传则新建（读 GITHUB_TOKEN）。
    """
    if gh_client is None:
        gh_client = github_pkg.GitHubClient(token=os.getenv("GITHUB_TOKEN"))

    owner, repo, number = gh_client.parse_pr_url(pr_url)
    patch = asyncio.run(gh_client.get_pr_patch(owner, repo, number))
    code = gh_client.parse_patch_to_code(patch)
    lang = gh_client.detect_language(patch)

    graph = build_supervisor_graph()
    result = asyncio.run(graph.ainvoke({"code": code, "language": lang}))
    return result.get("report", "# 审查完成\n\n未生成报告。")


def main(argv: list[str] | None = None) -> None:
    """CLI 主入口。"""
    parser = argparse.ArgumentParser(
        prog="cr-agent",
        description="多 Agent 代码审查协作平台",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    review_parser = sub.add_parser("review", help="审查代码文件或 GitHub PR")
    src = review_parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="本地代码文件路径")
    src.add_argument(
        "--pr",
        help="GitHub PR URL（如 https://github.com/owner/repo/pull/1）",
    )
    review_parser.add_argument(
        "--language",
        default=None,
        help="语言（python/javascript/typescript/go/java）；--file 默认从扩展名推断，--pr 从 PR 文件推断",
    )

    args = parser.parse_args(argv)

    if args.command == "review":
        if args.file:
            report = cmd_review(args.file, args.language)
        else:
            report = cmd_review_pr(args.pr)
        print(report)


if __name__ == "__main__":
    main()
