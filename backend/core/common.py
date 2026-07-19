"""公共工具函数（抽取重复逻辑）。

- extract_json_array: 从 LLM 文本提取 JSON 数组（decompose.py / workers/base.py 共用）
- fetch_pr_code: 从 PR URL 获取代码（reviews.py / webhooks.py 共用）
"""
import json
import re
from typing import Tuple

from backend.integrations import github as github_pkg


def extract_json_array(text: str) -> list:
    """从 LLM 文本提取 JSON 数组。

    LLM 常把 JSON 包在 ```json ... ``` 或夹杂解释文字里，不能直接 json.loads 整个文本。
    用正则抓第一个 '[' 到最后一个 ']' 的子串，最稳。抓不到或解析失败抛 ValueError。
    """
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError("响应中找不到 JSON 数组")
    data = json.loads(match.group(0))
    if not isinstance(data, list):
        raise ValueError("LLM 返回的不是数组")
    return data


async def fetch_pr_code(pr_url: str) -> Tuple[str, str]:
    """从 GitHub PR URL 获取代码和语言。

    返回 (code, language) 元组。失败抛 ValueError 或 RuntimeError。
    """
    gh = github_pkg.GitHubClient()
    owner, repo, number = gh.parse_pr_url(pr_url)
    patch = await gh.get_pr_patch(owner, repo, number)
    code = gh.parse_patch_to_code(patch)
    if not code.strip():
        raise ValueError("PR 没有可审查的代码变更")
    language = gh.detect_language(patch)
    return code, language
