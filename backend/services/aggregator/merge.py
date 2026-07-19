"""Aggregator — 合并去重 + 排序。

接收 4 个 Worker 的 findings 列表，做两件事：
1. **去重**：两个 Worker 在同一行发现了同一个问题（line + description 相同，大小写不敏感）
   → 合并为一条，取更高 severity，来源列表追加。
2. **排序**：按 severity 降序（high > medium > low > info），同 severity 按行号升序。

设计要点：
- 去重 key = (line, description_lower_stripped)。line 为 None 时不去重（None 行号 = LLM 没给行号，无法判定是否同一位置）。
- 合并时 severity 取更严重的（high > medium > low > info），suggestion 拼接，sources 收集所有来源 worker。
- 输入 findings 列表可能来自 4 个 Worker 并行产出（operator.add 累加后的结果），顺序不确定，所以排序是必须的。
"""
import logging

logger = logging.getLogger(__name__)

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}


def _severity_rank(sev: str) -> int:
    """severity → 排序权重，越小越靠前。未知 severity 归入 info。"""
    return _SEVERITY_ORDER.get(sev, 3)


def _line_key(line) -> tuple:
    """把 line 统一成可比较的键，避免 int/str/None 混排触发 TypeError。

    - 数值行号（int/float）→ 最前，按数值升序
    - None（LLM 没给行号）→ 其次
    - 字符串行号（如 "N/A"，structure Worker 常返回）→ 最后，按字符串排
    返回 (rank, value) 元组，保证任意两种 line 类型都能比较。
    """
    if isinstance(line, (int, float)):
        return (0, line)
    if line is None:
        return (1, 0)
    return (2, str(line))


def _merge_two(a: dict, b: dict) -> dict:
    """合并两条同 key finding：取更高 severity，拼接 suggestion，收集 sources。"""
    sources = set(a.get("sources", [a.get("worker", "?")]))
    sources.update(b.get("sources", [b.get("worker", "?")]))
    # severity 取更严重的
    if _severity_rank(a["severity"]) <= _severity_rank(b["severity"]):
        base = a
        other = b
    else:
        base = b
        other = a
    # suggestion 拼接（去重）
    suggestions = []
    for s in [base.get("suggestion", ""), other.get("suggestion", "")]:
        if s and s not in suggestions:
            suggestions.append(s)
    return {
        **base,
        "suggestion": " / ".join(suggestions) if suggestions else "",
        "sources": sorted(sources),
    }


def split_by_confidence(findings: list[dict], threshold: float = 0.0) -> tuple[list[dict], list[dict]]:
    """按 confidence 阈值把 findings 拆成 (high, low) 两个列表。

    - confidence >= threshold → high（高置信度，正式报告）
    - confidence < threshold → low（低置信度，提示区）
    - 缺失 confidence 字段 → 视为 0.0（最不可信）

    不修改输入列表，返回新的列表。
    """
    if threshold <= 0.0:
        return list(findings), []
    high: list[dict] = []
    low: list[dict] = []
    for f in findings:
        conf = f.get("confidence", 0.0)
        # 非数值 confidence 兜底为 0.0
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.0
        if conf >= threshold:
            high.append(f)
        else:
            low.append(f)
    return high, low


def aggregate_findings(findings: list[dict],
                       confidence_threshold: float = 0.0) -> list[dict]:
    """去重 + 排序。返回新的列表，不修改输入。

    confidence_threshold > 0 时，先过滤掉 confidence < threshold 的 finding
    （缺失 confidence 视为 0.0），再对剩余做去重 + 排序。
    低置信度的 finding 不会出现在返回值里（如需获取，用 split_by_confidence）。
    """
    if not findings:
        return []

    # Task 13.2: 置信度阈值过滤
    if confidence_threshold > 0.0:
        findings, _low = split_by_confidence(findings, confidence_threshold)
        if not findings:
            return []

    # 去重：(line, description_lower) 为 key
    deduped: dict[tuple, dict] = {}
    no_line: list[dict] = []  # line=None 的不参与去重，直接保留

    for f in findings:
        line = f.get("line")
        if line is None:
            no_line.append(f)
            continue
        key = (line, f.get("description", "").strip().lower())
        if key in deduped:
            deduped[key] = _merge_two(deduped[key], f)
        else:
            deduped[key] = {**f, "sources": [f.get("worker", "?")]}

    result = list(deduped.values()) + no_line

    # 排序：severity 升序权重（high 先）→ line 升序（数值优先，None/str 殿后）
    result.sort(key=lambda f: (_severity_rank(f.get("severity", "info")),
                               _line_key(f.get("line"))))
    return result
