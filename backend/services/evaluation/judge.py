"""LLM-as-Judge 评测（Phase 10: Task 10.2）。

两种评分：
1. judge_rule_based：确定性基线（关键词重叠 + 来源标记检测），无需 LLM，快。
2. judge_with_llm：LLM 作为裁判，输入 code+expected+actual_report，输出三维度 0-1 分数。

三维度（与 1 号项目 ai-resume-analyzer 的 LLM-as-Judge 对齐，便于简历绑定）：
- completeness（完整性 40%）：actual 覆盖 expected 的程度
- accuracy（准确性 40%）：actual 是否正确无幻觉
- source_traceability（来源可信度 20%）：actual 是否标注来源（行号/代码片段）
composite = 0.4*completeness + 0.4*accuracy + 0.2*source_traceability
"""
import json
import re
from dataclasses import dataclass

from backend.core.config import settings
from backend.core import llm as llm_mod


@dataclass
class Judgment:
    completeness: float
    accuracy: float
    source_traceability: float
    composite: float
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "completeness": self.completeness,
            "accuracy": self.accuracy,
            "source_traceability": self.source_traceability,
            "composite": self.composite,
            "rationale": self.rationale,
        }


def _composite(comp: float, acc: float, src: float) -> float:
    return round(0.4 * comp + 0.4 * acc + 0.2 * src, 4)


_PROMPT = """你是代码审查质量裁判。请评估「实际审查结果」相对「期望发现」的质量。

# 原始代码
{code}

# 期望发现（ground truth）
{expected}

# 实际审查报告
{actual}

请严格从三维度打分（每维度 0.0-1.0，最多两位小数）：
- completeness: 实际报告覆盖了期望发现的多少（漏报扣分）
- accuracy: 实际报告是否正确（幻觉、错误严重度、张冠李戴都扣分）
- source_traceability: 实际报告是否标注了来源（行号/代码片段，便于追溯）

仅输出 JSON，不要其他文字：
{{"completeness":<float>, "accuracy":<float>, "source_traceability":<float>, "rationale":"<简短理由>"}}"""


def _parse_judge_json(text: str) -> dict:
    """从 LLM 输出解析 judge JSON（容错：去 ```json 包裹，正则兜底）。"""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        nums = re.findall(r"(\d+\.?\d*)", text)
        return {
            "completeness": float(nums[0]) if nums else 0.0,
            "accuracy": float(nums[1]) if len(nums) > 1 else 0.0,
            "source_traceability": float(nums[2]) if len(nums) > 2 else 0.0,
            "rationale": "parse-fallback",
        }


def judge_rule_based(expected_findings: list[dict], actual_report: str) -> Judgment:
    """确定性基线评分（无 LLM）。

    - completeness：期望 description 前 8 字在报告中出现的比例
    - accuracy：基线用 completeness 近似（无 LLM 时无法判幻觉）
    - source_traceability：报告含「行」或代码块标记
    """
    if not expected_findings:
        src = 1.0 if ("行" in actual_report or "```" in actual_report) else 0.5
        return Judgment(1.0, 1.0, src, _composite(1.0, 1.0, src), "rule-based:no-expected")

    comp_hits = 0
    for ef in expected_findings:
        desc = ef.get("description", "")
        if desc and desc[:8] in actual_report:
            comp_hits += 1
    comp = comp_hits / len(expected_findings)
    acc = comp  # 基线近似
    src = 1.0 if ("行" in actual_report or "```" in actual_report) else 0.0
    return Judgment(comp, acc, src, _composite(comp, acc, src), "rule-based heuristic")


async def judge_with_llm(
    code: str,
    expected_findings: list[dict],
    actual_report: str,
    client=None,
) -> Judgment:
    """LLM 作为裁判评分。client 可注入（测试用）。"""
    client = client or llm_mod.get_chat_client()
    prompt = _PROMPT.format(
        code=code,
        expected=json.dumps(expected_findings, ensure_ascii=False, indent=2),
        actual=actual_report,
    )
    resp = await client.chat.completions.create(
        model=settings.CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        timeout=settings.LLM_TIMEOUT,
    )
    text = resp.choices[0].message.content
    data = _parse_judge_json(text)
    comp = float(data.get("completeness", 0.0))
    acc = float(data.get("accuracy", 0.0))
    src = float(data.get("source_traceability", 0.0))
    return Judgment(comp, acc, src, _composite(comp, acc, src), data.get("rationale", ""))
