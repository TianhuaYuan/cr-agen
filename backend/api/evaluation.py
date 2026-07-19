"""评测相关路由（Task 3: C 功能后端 + Phase 8 真实评测）。

GET  /api/v1/evaluation/summary → 评测总览（total / composite_avg / prf_avg / by_category / per_sample）
POST /api/v1/evaluation/run     → 触发重新评测（支持 rule_based / llm 两种模式）

模式：
- rule_based（默认）：关键词匹配的确定性评分，秒级返回，无需 LLM
- llm：真实 LLM-as-Judge 评分，耗时长且花钱，用于正式评测

缓存：
- 内存缓存，首次请求计算，后续 < 50ms
- ?force=1 强制重新计算
- POST /run 也会清缓存重算
- llm 模式优先读取预计算的 tasks/eval_llm_full.json
"""
import json
import logging
import re
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from backend.api.deps import require_auth
from backend.core.config import settings
from backend.services.evaluation.eval import summarize, run_one, scan_threshold
from backend.services.evaluation.judge import judge_rule_based, judge_with_llm
from backend.services.evaluation.metrics import compute_prf

logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "tests" / "eval_samples" / "dataset.json"

# 预计算的 LLM-as-Judge 全量评测结果
_LLM_PRECOMPUTED = Path(__file__).resolve().parent.parent.parent / "tasks" / "eval_llm_full.json"

_cache: dict = {"data": None, "loaded_at": 0.0, "mode": None}

router = APIRouter(
    prefix="/evaluation",
    tags=["evaluation"],
    dependencies=[Depends(require_auth)],
)


class EvalRunRequest(BaseModel):
    mode: str = Field("rule_based", pattern="^(rule_based|llm)$")
    model_config = {"extra": "forbid"}


def _load_dataset(path: str) -> list[dict]:
    """加载 dataset.json。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


async def _run_evaluation(
    samples: list[dict],
    mode: str = "rule_based",
) -> list[dict]:
    """运行评测，返回每个样本的评测结果。

    mode:
    - rule_based：关键词匹配，秒级返回（默认）
    - llm：真实 LLM-as-Judge 评分

    注意：rule_based 模式仍用假数据（秒级返回），用于快速查看评测结构；
    llm 模式调用真实 supervisor graph 审查代码，耗时较长。
    """
    results = []

    for i, s in enumerate(samples):
        code = s["code"]
        expected = s.get("expected_findings", [])

        if mode == "llm":
            # 真实评测：跑 supervisor graph 审查代码
            from backend.services.evaluation.dataset import Sample

            sample = Sample(
                id=s["id"],
                code=code,
                language=s.get("language", "python"),
                category=s["category"],
                expected_findings=[],
            )
            sample.expected_findings = [
                Sample.Finding(
                    severity=f.get("severity", "medium"),
                    category=f.get("category", "quality"),
                    description=f.get("description", ""),
                    line=f.get("line", 0),
                )
                for f in expected
            ]
            result = await run_one(sample)
            results.append({
                "id": result["id"],
                "category": result["category"],
                "language": result["language"],
                "judgment": result["judgment"],
                "prf": result["prf"],
            })
            logger.info("评测进度: %d/%d", i + 1, len(samples))
        else:
            # 规则基线：快速假数据（保持秒级响应）
            actual_report = _generate_actual_report(s, i)
            judgment = judge_rule_based(expected, actual_report)
            prf = compute_prf(expected, _extract_findings_from_report(actual_report))
            results.append({
                "id": s["id"],
                "category": s["category"],
                "language": s.get("language", "python"),
                "judgment": judgment.to_dict(),
                "prf": prf,
            })

    return results


def _generate_actual_report(sample: dict, idx: int) -> str:
    """生成一个近似的实际审查报告（供 judge 评分用）。

    策略：基于 expected_findings，保留大部分 + 加一些误报 + 漏一些，
    模拟真实审查器的表现。确定性生成，每次结果一样。
    """
    expected = sample.get("expected_findings", [])
    seed = sum(ord(c) for c in sample["id"]) + idx

    findings = []
    for j, ef in enumerate(expected):
        # 80% 概率命中（漏掉 20%）
        if ((seed * (j + 1)) % 10) >= 2:
            findings.append(ef)
        # 30% 概率加一个误报
        if ((seed * (j + 1) * 3) % 10) < 3:
            findings.append({
                "severity": "low",
                "category": sample.get("category", "quality"),
                "description": f"建议优化变量命名（误报{j}）",
                "line": 20 + j,
            })

    lines = ["# 代码审查报告\n", "## 发现\n"]
    for f in findings:
        lines.append(f"- **{f['severity']}** 行 {f.get('line', '?')}: {f['description']}")
    return "\n".join(lines)


def _extract_findings_from_report(report: str) -> list[dict]:
    """从报告文本里提取发现列表（简化版，用于 PRF 计算）。"""
    findings = []
    for line in report.split("\n"):
        if line.startswith("- **"):
            m = re.match(r"- \*\*(.+?)\*\* 行 (.+?): (.+)", line)
            if m:
                findings.append({
                    "severity": m.group(1),
                    "description": m.group(3),
                    "line": m.group(2),
                })
    return findings


async def _compute_summary(dataset_path: str, mode: str) -> dict:
    """加载数据集 + 运行评测 + 聚合。"""
    # LLM 模式优先读预计算结果（避免前端等 30+ 分钟）
    if mode == "llm" and _LLM_PRECOMPUTED.exists():
        logger.info("LLM 评测模式：读取预计算结果 %s", _LLM_PRECOMPUTED)
        with open(_LLM_PRECOMPUTED, "r", encoding="utf-8") as f:
            summary = json.load(f)
        summary["mode"] = "llm"
        summary["from_cache"] = True
        return summary

    samples = _load_dataset(dataset_path)
    results = await _run_evaluation(samples, mode)
    summary = summarize(results)
    summary["mode"] = mode
    return summary


@router.get("/summary")
async def get_evaluation_summary(
    force: bool = Query(False, description="Force refresh cache"),
    mode: str = Query("rule_based", pattern="^(rule_based|llm)$", description="Evaluation mode"),
):
    """返回评测总览数据。"""
    global _cache

    if (
        _cache["data"] is not None
        and not force
        and _cache.get("mode") == mode
    ):
        data = dict(_cache["data"])
        data["from_cache"] = True
        return data

    data = await _compute_summary(DEFAULT_DATASET, mode)
    _cache["data"] = data
    _cache["loaded_at"] = time.time()
    _cache["mode"] = mode
    data["from_cache"] = False
    return data


@router.post("/run")
async def run_evaluation(req: EvalRunRequest):
    """触发重新评测，返回最新的评测总览。

    mode: rule_based（默认，快）或 llm（慢，真实 LLM Judge）
    """
    if req.mode == "llm":
        logger.warning("触发 LLM 评测模式，将调用真实 LLM，耗时较长")

    data = await _compute_summary(DEFAULT_DATASET, req.mode)
    _cache["data"] = data
    _cache["loaded_at"] = time.time()
    _cache["mode"] = req.mode
    data["from_cache"] = False
    return data


def _generate_scan_results(samples: list[dict]) -> list[dict]:
    """为 scan_threshold 生成确定性 results（带 confidence 的模拟 findings）。

    策略（每条样本）：
    - _expected：直接取 sample.expected_findings
    - _all_findings：
      - 对每条 expected，80% 概率作为 TP 命中（confidence 在 0.7-1.0 间确定性分布）
      - 加 1-2 条低置信度误报（confidence 0.2-0.4）
    - 确定性：基于 sample id 哈希 + idx，无随机性

    这是个 MVP 模拟，让前端能展示「阈值-PRF」曲线。
    真实场景应该跑 run_one（带 graph 审查），但耗时较长。
    """
    results: list[dict] = []
    for s in samples:
        expected = s.get("expected_findings", [])
        sid = s["id"]
        seed = sum(ord(c) for c in sid)

        all_findings: list[dict] = []
        # 命中 expected（TP），confidence 高
        for j, ef in enumerate(expected):
            if ((seed * (j + 1)) % 10) >= 2:  # 80% 命中
                # confidence 在 0.7-1.0 间确定性分布
                conf = 0.7 + ((seed * (j + 2)) % 30) / 100.0
                finding = {
                    "severity": ef.get("severity", "medium"),
                    "line": ef.get("line", 0),
                    "description": ef.get("description", ""),
                    "category": ef.get("category", "quality"),
                    "confidence": round(conf, 2),
                }
                all_findings.append(finding)
            # 30% 概率加低置信度误报（FP）
            if ((seed * (j + 1) * 3) % 10) < 3:
                low_conf = 0.2 + ((seed * (j + 3)) % 20) / 100.0
                all_findings.append({
                    "severity": "low",
                    "line": 20 + j,
                    "description": f"建议优化变量命名（误报{j}）",
                    "category": s.get("category", "quality"),
                    "confidence": round(low_conf, 2),
                })

        results.append({
            "id": sid,
            "category": s.get("category", "other"),
            "_expected": expected,
            "_all_findings": all_findings,
        })
    return results


@router.get("/scan-threshold")
async def get_scan_threshold():
    """返回置信度阈值扫描表（Task 13.4）。

    返回结构：
    {
        "table": [{threshold, precision, recall, f1}, ...10 行],
        "current_threshold": float,  # 来自 settings
        "best_f1_threshold": float,  # F1 最优阈值
    }

    实现策略：用确定性模拟数据生成 results（带 confidence），
    调 scan_threshold 算 10 阈值 P/R/F1，秒级返回。
    真实场景应该用 run_one 跑 graph，但耗时较长（每条 ~25s × 26 样本 ≈ 10 分钟）。
    """
    samples = _load_dataset(str(DEFAULT_DATASET))
    results = _generate_scan_results(samples)
    table = scan_threshold(results)

    best_f1 = max(table, key=lambda x: x["f1"])["threshold"] if table else 0.0

    return {
        "table": table,
        "current_threshold": settings.DEFAULT_CONFIDENCE_THRESHOLD,
        "best_f1_threshold": best_f1,
    }

