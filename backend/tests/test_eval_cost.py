"""成本控制测试（W3: Task 11.3）。

TDD Red → Green：
- 验证 TokenMeter 累加、MeteredClient 拦截 .chat.completions.create 记录 usage、
  其余属性代理真实 client、estimate_cost 单价计算、run_one(meter=) 真实累加 token。
"""
import json
from types import SimpleNamespace

import pytest

from backend.services.evaluation.cost import TokenMeter, MeteredClient, estimate_cost


class _Usage:
    def __init__(self, p: int, c: int):
        self.prompt_tokens = p
        self.completion_tokens = c
        self.total_tokens = p + c


class _FakeClient:
    """返回带 usage 的假 client，并暴露非 .chat 属性用于代理测试。"""

    def __init__(self, p: int = 10, c: int = 5):
        self._usage = _Usage(p, c)
        self.proxied_models = False

    @property
    def chat(self):
        async def _create(*a, **k):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))],
                usage=self._usage,
            )

        return SimpleNamespace(completions=SimpleNamespace(create=_create))

    @property
    def models(self):
        # 非 .chat 路径的属性，MeteredClient 应原样代理
        self.proxied_models = True
        return SimpleNamespace(list=lambda: [])


def test_token_meter_add():
    m = TokenMeter()
    m.add(_Usage(10, 5))
    m.add(_Usage(20, 10))
    assert m.prompt_tokens == 30
    assert m.completion_tokens == 15
    assert m.total_tokens == 45
    assert m.call_count == 2
    assert m.to_dict()["call_count"] == 2


def test_token_meter_add_tolerates_missing_usage():
    m = TokenMeter()
    m.add(None)  # 真实调用有时无 usage 字段
    assert m.call_count == 0 and m.total_tokens == 0


@pytest.mark.asyncio
async def test_metered_client_records_usage():
    m = TokenMeter()
    client = MeteredClient(_FakeClient(10, 5), m)
    resp = await client.chat.completions.create(model="x")
    assert m.call_count == 1
    assert m.total_tokens == 15
    assert resp.choices[0].message.content == "[]"


def test_metered_client_proxies_other_attrs():
    m = TokenMeter()
    client = MeteredClient(_FakeClient(), m)
    models = client.models  # 应代理到真实 client
    assert isinstance(models, SimpleNamespace)
    assert client._real.proxied_models is True  # 确实访问了真实 client


def test_estimate_cost():
    m = TokenMeter()
    m.add(_Usage(1000, 500))
    # prompt 0.01/1k, completion 0.02/1k → 0.01 + 0.01 = 0.02
    cost = estimate_cost(m, price_per_1k_prompt=0.01, price_per_1k_completion=0.02)
    assert abs(cost - 0.02) < 1e-9


class _FakeGraphClient:
    _FINDINGS = json.dumps([
        {"severity": "high", "line": 1, "description": "硬编码 API 密钥", "suggestion": "x", "code_snippet": "y"},
    ])

    @property
    def chat(self):
        async def _create(*a, **k):
            usage = SimpleNamespace(prompt_tokens=12, completion_tokens=3, total_tokens=15)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=self._FINDINGS))],
                usage=usage,
            )

        return SimpleNamespace(completions=SimpleNamespace(create=_create))


class _FakeJudgeClient:
    _JUDGE = json.dumps({"completeness": 0.9, "accuracy": 0.8, "source_traceability": 1.0, "rationale": "ok"})

    @property
    def chat(self):
        async def _create(*a, **k):
            usage = SimpleNamespace(prompt_tokens=8, completion_tokens=2, total_tokens=10)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=self._JUDGE))],
                usage=usage,
            )

        return SimpleNamespace(completions=SimpleNamespace(create=_create))


def _samples():
    from backend.services.evaluation.dataset import Sample, ExpectedFinding
    return [Sample(
        id="sec-001", language="python", category="security", code="API_KEY='x'",
        expected_findings=[ExpectedFinding("high", "security", "硬编码 API 密钥", 1)],
    )]


@pytest.mark.asyncio
async def test_run_one_meters_graph_tokens(monkeypatch):
    monkeypatch.setattr("backend.core.llm.get_chat_client", lambda: _FakeGraphClient())
    meter = TokenMeter()
    # 只测 graph 用量（judge 用显式 client，不经 meter）
    await run_one_meter_helper(meter)
    assert meter.call_count >= 5  # decompose + 4 Worker
    assert meter.total_tokens > 0


async def run_one_meter_helper(meter):
    from backend.services.evaluation.eval import run_one
    from backend.services.evaluation.dataset import Sample, ExpectedFinding
    s = Sample(id="sec-001", language="python", category="security", code="API_KEY='x'",
               expected_findings=[ExpectedFinding("high", "security", "硬编码 API 密钥", 1)])
    await run_one(s, judge_client=_FakeJudgeClient(), meter=meter)


class _FakeGraphClientWithUsage:
    """带 usage 的 graph 假 client，用于验证 run_all + meter 全链路。"""
    _TASKS = json.dumps([
        {"role": "security", "description": "审查安全"},
        {"role": "quality", "description": "审查质量"},
        {"role": "performance", "description": "审查性能"},
        {"role": "structure", "description": "审查结构"},
    ])
    _FINDINGS = json.dumps([
        {"severity": "high", "line": 1, "description": "硬编码密钥", "suggestion": "x", "code_snippet": "y"},
    ])
    _USAGE = SimpleNamespace(prompt_tokens=12, completion_tokens=3, total_tokens=15)

    @property
    def chat(self):
        async def _create(*a, **k):
            msgs = k.get("messages") or (a[0] if a else None)
            text = ""
            if msgs:
                text = msgs[-1].get("content", "") if isinstance(msgs, list) else ""
            content = self._TASKS if "任务" in text or "task" in text.lower() else self._FINDINGS
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=self._USAGE,
            )
        return SimpleNamespace(completions=SimpleNamespace(create=_create))


@pytest.mark.asyncio
async def test_run_all_returns_tokens_with_meter(monkeypatch):
    monkeypatch.setattr("backend.core.llm.get_chat_client", lambda: _FakeGraphClientWithUsage())
    from backend.services.evaluation.eval import run_all
    from backend.services.evaluation.cost import TokenMeter

    meter = TokenMeter()
    from pathlib import Path
    dataset_path = Path(__file__).resolve().parent / "eval_samples" / "dataset.json"
    summary = await run_all(dataset_path, limit=1, meter=meter)
    assert summary["tokens"] is not None
    assert summary["tokens"]["call_count"] >= 5  # decompose + 4 Worker
    assert summary["tokens"]["total_tokens"] > 0
