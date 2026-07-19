"""评测主流程（Phase 10: Task 10.2，W3: Task 11.1 扩展）。

run_one：单条样本 → supervisor graph 审查 → LLM-as-Judge 评分 → 返回原始 findings。
run_samples / run_all：遍历样本 → 汇总（总体 + 分类聚合）。
summarize / summarize_by_category：聚合计算。
render_report：渲染 Markdown 评测报告。

CLI：
  python -m backend.services.evaluation.eval --limit 5
  python -m backend.services.evaluation.eval --all --report reports/eval_report.md
"""
import argparse
import asyncio
import json
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from backend.services.aggregator.merge import aggregate_findings, split_by_confidence
from backend.services.evaluation.dataset import Sample, load_dataset
from backend.services.evaluation.judge import judge_with_llm
from backend.services.evaluation.metrics import compute_prf
from backend.services.supervisor.graph import build_supervisor_graph


async def run_one(sample: Sample, judge_client=None, meter=None,
                  confidence_threshold: float = 0.0) -> dict:
    """单条样本评测：graph 审查 → judge 评分。

    返回 dict 含：id / category / language / judgment / findings（graph 原始 findings）。
    meter 为可选 TokenMeter（W3 Task 11.3 接入），传入则用 MeteredClient 包 graph client。
    confidence_threshold > 0 时，先过滤低置信度 finding 再算 PRF（Task 13.3）。
    """
    graph_client = None
    original_getter = None
    if meter is not None:
        from backend.services.evaluation.cost import MeteredClient
        from backend.core import llm as llm_mod

        original_getter = llm_mod.get_chat_client
        real = original_getter()
        graph_client = MeteredClient(real, meter)
        llm_mod.get_chat_client = lambda: graph_client

    try:
        graph = build_supervisor_graph()
        result = await graph.ainvoke({"code": sample.code, "language": sample.language})
    finally:
        if original_getter is not None:
            from backend.core import llm as llm_mod

            llm_mod.get_chat_client = original_getter  # 还原原始函数，避免污染

    report = result.get("report", "")
    # 用聚合后的去重 findings 作为「实际发现」（与报告一致），PRF 比对更准
    all_findings = aggregate_findings(result.get("worker_results", []))
    # Task 13.3: 置信度阈值过滤（split_by_confidence 内部已处理 threshold<=0 → 不过滤）
    findings, _low = split_by_confidence(all_findings, confidence_threshold)
    expected = [f.__dict__ for f in sample.expected_findings]
    prf = compute_prf(expected, findings)
    judgment = await judge_with_llm(sample.code, expected, report, client=judge_client)
    return {
        "id": sample.id,
        "category": sample.category,
        "language": sample.language,
        "judgment": judgment.to_dict(),
        "findings": findings,
        "prf": prf,
        "report": report,
        "_expected": expected,  # Task 13.3: scan_threshold 需要，重新算 PRF
        "_all_findings": all_findings,  # 未过滤的完整 findings，scan_threshold 用
    }


async def run_samples(samples: list[Sample], judge_client=None, meter=None, limit: int | None = None) -> list[dict]:
    """遍历样本评测，返回每条结果（单条失败不影响整体）。"""
    if limit:
        samples = samples[:limit]
    results: list[dict] = []
    for s in samples:
        try:
            results.append(await run_one(s, judge_client=judge_client, meter=meter))
        except Exception as e:  # 单条失败不影响整体
            results.append({"id": s.id, "category": s.category, "error": str(e)})
    return results


async def run_all(dataset_path: str | Path, limit: int | None = None, meter=None) -> dict:
    """遍历数据集评测，返回汇总（含每条分数 + 总体 + 分类聚合）。"""
    samples = load_dataset(dataset_path)
    results = await run_samples(samples, limit=limit, meter=meter)
    summary = summarize(results)
    summary["tokens"] = meter.to_dict() if meter is not None else None
    return summary


def scan_threshold(results: list[dict]) -> list[dict]:
    """对已有 results 扫不同 confidence 阈值，返回各阈值 P/R/F1 表格。

    不重跑 graph——复用 results 里的 _all_findings（带 confidence，未过滤），
    用不同阈值过滤后，配合 _expected 重新算真实 PRF。
    O(阈值数 × findings 数)，非常快。

    Args:
        results: run_one / run_samples 的返回列表，需含 _all_findings + _expected

    Returns:
        [{"threshold": 0.0, "precision": ..., "recall": ..., "f1": ...}, ...]
        10 行，threshold 从 0.0 到 0.9 步长 0.1
    """
    if not results:
        return []

    # 只取有 _all_findings + _expected 的结果
    valid = [r for r in results if "_all_findings" in r and "_expected" in r]
    if not valid:
        return []

    table: list[dict] = []
    for i in range(10):
        threshold = i / 10.0
        prf_list = []
        for r in valid:
            high, _low = split_by_confidence(r["_all_findings"], threshold)
            prf = compute_prf(r["_expected"], high)
            prf_list.append(prf)

        table.append({
            "threshold": threshold,
            "precision": round(mean(p["precision"] for p in prf_list), 4),
            "recall": round(mean(p["recall"] for p in prf_list), 4),
            "f1": round(mean(p["f1"] for p in prf_list), 4),
        })
    return table


def summarize(results: list[dict]) -> dict:
    """从每条结果聚合：总体 composite_avg / prf_avg / by_category。"""
    scored = [r for r in results if "judgment" in r]
    composite_avg = mean(r["judgment"]["composite"] for r in scored) if scored else 0.0
    prf_list = [r["prf"] for r in scored if "prf" in r]
    prf_avg = _avg_prf(prf_list) if prf_list else None
    return {
        "total": len(results),
        "composite_avg": round(composite_avg, 4),
        "prf_avg": prf_avg,
        "by_category": summarize_by_category(scored),
        "per_sample": results,
    }


def summarize_by_category(results: list[dict]) -> dict:
    """按 category 聚合：count + 各维度均值 + prf 均值。"""
    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(r["category"], []).append(r)

    out: dict[str, dict] = {}
    for cat, rs in groups.items():
        out[cat] = {
            "count": len(rs),
            "composite_avg": round(mean(x["judgment"]["composite"] for x in rs), 4),
            "completeness_avg": round(mean(x["judgment"]["completeness"] for x in rs), 4),
            "accuracy_avg": round(mean(x["judgment"]["accuracy"] for x in rs), 4),
            "source_avg": round(mean(x["judgment"]["source_traceability"] for x in rs), 4),
            "prf": _avg_prf([x["prf"] for x in rs if "prf" in x]),
        }
    return out


def _avg_prf(prf_list: list[dict]) -> dict | None:
    if not prf_list:
        return None
    return {
        "precision": round(mean(x.get("precision", 0.0) for x in prf_list), 4),
        "recall": round(mean(x.get("recall", 0.0) for x in prf_list), 4),
        "f1": round(mean(x.get("f1", 0.0) for x in prf_list), 4),
    }


def render_report(summary: dict) -> str:
    """渲染 Markdown 评测报告。"""
    lines: list[str] = []
    lines.append("# cr-agent 评测报告")
    lines.append("")
    lines.append("## 总览")
    lines.append("")
    lines.append(f"- 样本总数：**{summary['total']}**")
    lines.append(f"- 综合得分 composite_avg：**{summary['composite_avg']}**")
    if summary.get("prf_avg"):
        p = summary["prf_avg"]
        lines.append(f"- 硬指标 PRF：precision={p['precision']} / recall={p['recall']} / f1={p['f1']}")
    if summary.get("tokens"):
        t = summary["tokens"]
        lines.append(
            f"- Token 用量：prompt={t['prompt_tokens']} / completion={t['completion_tokens']} "
            f"/ total={t['total_tokens']} / calls={t['call_count']}"
        )
    lines.append("")

    lines.append("## 分类明细")
    lines.append("")
    lines.append("| 类别 | 样本数 | composite | completeness | accuracy | source | PRF-f1 |")
    lines.append("|------|--------|-----------|--------------|----------|--------|--------|")
    for cat, m in summary["by_category"].items():
        prf_f1 = m["prf"]["f1"] if m.get("prf") else "-"
        lines.append(
            f"| {cat} | {m['count']} | {m['composite_avg']} | {m['completeness_avg']} "
            f"| {m['accuracy_avg']} | {m['source_avg']} | {prf_f1} |"
        )
    lines.append("")

    lines.append("## 每条样本")
    lines.append("")
    for r in summary["per_sample"]:
        if "error" in r:
            lines.append(f"### {r['id']} ❌ 错误：{r['error']}")
            lines.append("")
            continue
        j = r["judgment"]
        prf = r.get("prf")
        prf_str = f" / PRF-f1={prf['f1']}" if prf else ""
        lines.append(
            f"### {r['id']}（{r['category']}）— composite={j['composite']}{prf_str}"
        )
        lines.append("")
        lines.append(f"- completeness={j['completeness']} / accuracy={j['accuracy']} / source={j['source_traceability']}")
        if j.get("rationale"):
            lines.append(f"- 裁判理由：{j['rationale']}")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="cr-agent LLM-as-Judge 评测")
    default_dataset = (
        Path(__file__).resolve().parent.parent.parent
        / "tests"
        / "eval_samples"
        / "dataset.json"
    )
    parser.add_argument("--dataset", default=str(default_dataset))
    parser.add_argument("--limit", type=int, default=None, help="只评测前 N 条")
    parser.add_argument("--out", default="reports/eval_report.json", help="JSON 输出路径")
    parser.add_argument("--report", default=None, help="Markdown 报告输出路径")
    parser.add_argument("--tokens", action="store_true", help="计量 token 用量（graph + judge 全量）")
    parser.add_argument("--scan-threshold", action="store_true",
                        help="扫描 0.0~0.9 置信度阈值，输出各阈值 P/R/F1 表格")
    args = parser.parse_args()

    meter = None
    if args.tokens:
        from backend.services.evaluation.cost import TokenMeter

        meter = TokenMeter()

    if args.scan_threshold:
        # 阈值扫描模式：跑一次 graph，扫 10 个阈值，不跑真 judge（省 token）
        # 用 _RuleJudgeClient 返回固定评分，仅为走通 run_one 流程拿到 findings

        class _RuleJudgeClient:
            """阈值扫描用的假 judge client：不调 LLM，返回固定评分。"""
            @property
            def chat(self):
                async def _create(*a, **k):
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(
                            content=json.dumps({
                                "completeness": 0.5, "accuracy": 0.5,
                                "source_traceability": 0.5, "rationale": "scan",
                            })
                        ))]
                    )
                return SimpleNamespace(completions=SimpleNamespace(create=_create))

        samples = load_dataset(args.dataset)
        if args.limit:
            samples = samples[:args.limit]
        results = asyncio.run(run_samples(
            samples, judge_client=_RuleJudgeClient(), meter=meter
        ))
        table = scan_threshold(results)
        print("\n=== 置信度阈值扫描 ===")
        print(f"{'阈值':>6} | {'precision':>10} | {'recall':>10} | {'f1':>10}")
        print("-" * 48)
        for row in table:
            print(f"{row['threshold']:>6.1f} | {row['precision']:>10.4f} | "
                  f"{row['recall']:>10.4f} | {row['f1']:>10.4f}")
        # 找 F1 最优
        best = max(table, key=lambda x: x["f1"])
        print(f"\n★ F1 最优阈值：{best['threshold']:.1f} "
              f"(P={best['precision']:.4f} / R={best['recall']:.4f} / F1={best['f1']:.4f})")
        print(f"\n建议写入 settings.DEFAULT_CONFIDENCE_THRESHOLD = {best['threshold']:.1f}")
        return

    summary = asyncio.run(run_all(args.dataset, args.limit, meter=meter))
    Path(args.out).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.report:
        Path(args.report).write_text(render_report(summary), encoding="utf-8")
    print(f"评测完成：{summary['total']} 条，composite_avg={summary['composite_avg']}")


if __name__ == "__main__":
    main()
