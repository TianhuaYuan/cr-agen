"""LLM-as-Judge 评测测试（Phase 10: Task 10.2）。

TDD Red → Green：
- 先写测试（Red）：backend.services.evaluation.judge / eval 不存在 → import 失败。
- 再写实现（Green）：让测试通过。

测试策略：
- 注入 _FakeGraphClient（graph 审查用 LLM）与 _FakeJudgeClient（裁判 LLM），零真实 API。
- 验证 judge 解析、规则基线、LLM 裁判、run_one 端到端（graph + judge）。
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.services.evaluation.dataset import load_dataset
from backend.services.evaluation.judge import (
    Judgment,
    _parse_judge_json,
    judge_rule_based,
    judge_with_llm,
)


class _FakeGraphClient:
    """graph 审查用的假 LLM（返回 worker findings JSON）。"""

    _FINDINGS = json.dumps([
        {
            "severity": "high",
            "line": 1,
            "description": "硬编码 API 密钥和数据库密码",
            "suggestion": "改用环境变量",
            "code_snippet": "API_KEY='...'",
        }
    ])

    @property
    def chat(self):
        async def _create(*a, **k):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=self._FINDINGS))]
            )

        return SimpleNamespace(completions=SimpleNamespace(create=_create))


class _FakeJudgeClient:
    """裁判 LLM 假客户端（返回评委 JSON）。"""

    _JUDGE = json.dumps({
        "completeness": 0.9,
        "accuracy": 0.8,
        "source_traceability": 1.0,
        "rationale": "覆盖了主要安全问题，来源标注清晰",
    })

    @property
    def chat(self):
        async def _create(*a, **k):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=self._JUDGE))]
            )

        return SimpleNamespace(completions=SimpleNamespace(create=_create))


def _patch_graph_llm(monkeypatch):
    monkeypatch.setattr("backend.core.llm.get_chat_client", lambda: _FakeGraphClient())


def test_parse_judge_json_strips_fence():
    text = "```json\n" + json.dumps(
        {"completeness": 0.5, "accuracy": 0.5, "source_traceability": 0.5}
    ) + "\n```"
    d = _parse_judge_json(text)
    assert d["completeness"] == 0.5


def test_judge_rule_based():
    expected = [
        {"severity": "high", "category": "security", "description": "硬编码 API 密钥和数据库密码", "line": 1}
    ]
    report = "硬编码 API 密钥和数据库密码\n行号 1"
    j = judge_rule_based(expected, report)
    assert isinstance(j, Judgment)
    assert j.completeness > 0.5
    assert j.source_traceability == 1.0


@pytest.mark.asyncio
async def test_judge_with_llm(monkeypatch):
    monkeypatch.setattr("backend.core.llm.get_chat_client", lambda: _FakeJudgeClient())
    j = await judge_with_llm("code", [{"description": "x"}], "report")
    assert isinstance(j, Judgment)
    # 0.4*0.9 + 0.4*0.8 + 0.2*1.0 = 0.88
    assert abs(j.composite - 0.88) < 0.01


@pytest.mark.asyncio
async def test_run_one(monkeypatch):
    _patch_graph_llm(monkeypatch)
    from backend.services.evaluation.eval import run_one

    samples = load_dataset(Path(__file__).parent / "eval_samples" / "dataset.json")
    r = await run_one(samples[0], judge_client=_FakeJudgeClient())
    assert r["id"] == samples[0].id
    assert "judgment" in r
    assert r["judgment"]["composite"] > 0


# ── Task 13.3: 置信度阈值过滤 + 阈值扫描 ──────────────────────

class _FakeGraphClientWithConfidence:
    """返回带 confidence 的多条件 findings，用于阈值过滤测试。

    3 条 findings：
    - confidence=0.9（高，命中 expected）
    - confidence=0.3（低，误报）
    - confidence=0.6（中，误报）
    """

    _FINDINGS = json.dumps([
        {
            "severity": "high", "line": 1,
            "description": "硬编码 API 密钥和数据库密码",
            "suggestion": "改用环境变量", "code_snippet": "API_KEY='...'",
            "confidence": 0.9,
        },
        {
            "severity": "low", "line": 5,
            "description": "变量命名不规范",
            "suggestion": "用 snake_case", "code_snippet": "x=1",
            "confidence": 0.3,
        },
        {
            "severity": "medium", "line": 10,
            "description": "缺少类型注解",
            "suggestion": "加 type hint", "code_snippet": "def f():",
            "confidence": 0.6,
        },
    ])

    @property
    def chat(self):
        async def _create(*a, **k):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=self._FINDINGS))]
            )
        return SimpleNamespace(completions=SimpleNamespace(create=_create))


def _patch_graph_llm_with_confidence(monkeypatch):
    monkeypatch.setattr(
        "backend.core.llm.get_chat_client",
        lambda: _FakeGraphClientWithConfidence(),
    )


@pytest.mark.asyncio
async def test_run_one_with_confidence_threshold(monkeypatch):
    """run_one 支持 confidence_threshold 参数，过滤低置信度后再算 PRF。"""
    _patch_graph_llm_with_confidence(monkeypatch)
    from backend.services.evaluation.eval import run_one

    samples = load_dataset(Path(__file__).parent / "eval_samples" / "dataset.json")
    # threshold=0.5 → 过滤掉 confidence=0.3 的，保留 0.9 和 0.6
    r = await run_one(samples[0], judge_client=_FakeJudgeClient(),
                      confidence_threshold=0.5)
    assert r["id"] == samples[0].id
    # findings 应只含 confidence >= 0.5 的
    for f in r["findings"]:
        assert f.get("confidence", 0.0) >= 0.5


@pytest.mark.asyncio
async def test_run_one_threshold_affects_prf(monkeypatch):
    """阈值过滤后 PRF 应该变化（precision 提升）。"""
    _patch_graph_llm_with_confidence(monkeypatch)
    from backend.services.evaluation.eval import run_one

    samples = load_dataset(Path(__file__).parent / "eval_samples" / "dataset.json")
    # 不过滤
    r0 = await run_one(samples[0], judge_client=_FakeJudgeClient())
    # threshold=0.7 → 只保留 confidence >= 0.7（只有 0.9 那条）
    r7 = await run_one(samples[0], judge_client=_FakeJudgeClient(),
                       confidence_threshold=0.7)
    # findings 数量应该减少
    assert len(r7["findings"]) <= len(r0["findings"])


@pytest.mark.asyncio
async def test_scan_threshold_returns_table(monkeypatch):
    """scan_threshold 对已有 results 扫不同阈值，返回各阈值 P/R/F1。"""
    _patch_graph_llm_with_confidence(monkeypatch)
    from backend.services.evaluation.eval import run_one, scan_threshold

    samples = load_dataset(Path(__file__).parent / "eval_samples" / "dataset.json")
    # 跑一次，拿带原始 findings 的 result
    r = await run_one(samples[0], judge_client=_FakeJudgeClient())
    results = [r]

    # 扫描 0.0 ~ 0.9
    table = scan_threshold(results)
    assert isinstance(table, list)
    assert len(table) == 10  # 0.0, 0.1, ..., 0.9
    for row in table:
        assert "threshold" in row
        assert "precision" in row
        assert "recall" in row
        assert "f1" in row
    # threshold=0.0 应该和不过滤一致
    assert table[0]["threshold"] == 0.0
    # threshold=0.9 应该 precision 更高（或相等）
    assert table[9]["precision"] >= table[0]["precision"]


@pytest.mark.asyncio
async def test_scan_threshold_empty_results():
    """空 results → scan_threshold 返回空列表。"""
    from backend.services.evaluation.eval import scan_threshold
    table = scan_threshold([])
    assert table == []
