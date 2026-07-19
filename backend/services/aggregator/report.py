"""Report 生成 — 将 Aggregator 输出的 findings 渲染为结构化 Markdown。

报告结构：
1. 标题（# 代码审查报告）
2. 摘要（语言、问题总数、各严重度数量、参与的 Worker）
3. 按 Worker 维度分组的表格（安全/质量/性能/架构）
4. 异常/警告区（如果有 errors）
"""
import logging

logger = logging.getLogger(__name__)

_ROLE_LABELS = {
    "security": "安全审查",
    "quality": "代码质量",
    "performance": "性能",
    "structure": "架构",
}

_ROLE_ORDER = ["security", "quality", "performance", "structure"]

# 每个审查维度的专属图标（按 role 索引，区别于按 severity 索引的 _SEVERITY_BADGE）
_ROLE_ICON = {
    "security": "🔒",
    "quality": "✨",
    "performance": "⚡",
    "structure": "🏗️",
}

_SEVERITY_BADGE = {
    "high": "🔴 高危",
    "medium": "🟡 中危",
    "low": "🟢 低危",
    "info": "ℹ️ 提示",
}

_SEVERITY_COUNT_KEYS = {"high": "high", "medium": "medium", "low": "low", "info": "info"}


def generate_report(findings: list[dict], language: str = "python",
                    errors: list[str] | None = None,
                    low_confidence: list[dict] | None = None) -> str:
    """渲染 Markdown 审查报告。

    Task 13.2 新增 low_confidence 参数：低置信度 finding 单独展示在报告末尾的
    「低置信度提示」区，与正式发现区分开，避免噪音干扰但又保留信息。
    """
    errors = errors or []
    low_confidence = low_confidence or []
    lines: list[str] = []

    # ── 标题 ──
    lines.append("# 🔍 代码审查报告\n")

    # ── 摘要 ──
    total = len(findings)
    counts = {k: 0 for k in _SEVERITY_COUNT_KEYS}
    for f in findings:
        sev = f.get("severity", "info")
        if sev in counts:
            counts[sev] += 1

    lines.append("## 📋 摘要\n")
    lines.append(f"- **语言**: {language}")
    lines.append(f"- **问题总数**: {total}（高危: {counts['high']}, "
                 f"中危: {counts['medium']}, 低危: {counts['low']}, 提示: {counts['info']}）")
    if low_confidence:
        lines.append(f"- **低置信度提示**: {len(low_confidence)} 条（已折叠，详见报告末尾）")
    workers_present = sorted(set(f.get("worker", "?") for f in findings))
    if workers_present:
        lines.append(f"- **审查维度**: {', '.join(workers_present)}")
    lines.append("")

    # ── 分维度表格 ──
    if findings:
        for role in _ROLE_ORDER:
            # 每个 finding 只在 primary worker 的 section 出现（worker 字段）
            # sources 列表在表格里展示所有来源
            role_findings = [f for f in findings if f.get("worker") == role]
            if not role_findings:
                continue
            label = _ROLE_LABELS.get(role, role)
            icon = _ROLE_ICON.get(role, "📋")
            lines.append(f"## {icon} {label}\n")
            lines.append("| 行号 | 严重度 | 问题 | 建议 | 来源 |")
            lines.append("|------|--------|------|------|------|")
            for f in role_findings:
                line = f.get("line", "—")
                sev = _SEVERITY_BADGE.get(f.get("severity", "info"), f.get("severity", ""))
                desc = f.get("description", "")
                sug = f.get("suggestion", "")
                sources = ", ".join(f.get("sources", [f.get("worker", "?")]))
                lines.append(f"| {line} | {sev} | {desc} | {sug} | {sources} |")
            lines.append("")
    else:
        lines.append("## ✅ 未发现问题\n")
        lines.append("本次审查未发现明显问题。\n")

    # ── 低置信度提示区（Task 13.2）──
    if low_confidence:
        lines.append("## 💭 低置信度提示\n")
        lines.append(f"> 以下 {len(low_confidence)} 条发现置信度较低，可能为误报，仅供参考。\n")
        lines.append("| 行号 | 严重度 | 问题 | 建议 | 来源 | 置信度 |")
        lines.append("|------|--------|------|------|------|--------|")
        for f in low_confidence:
            line = f.get("line", "—")
            sev = _SEVERITY_BADGE.get(f.get("severity", "info"), f.get("severity", ""))
            desc = f.get("description", "")
            sug = f.get("suggestion", "")
            sources = ", ".join(f.get("sources", [f.get("worker", "?")]))
            conf = f.get("confidence", "?")
            lines.append(f"| {line} | {sev} | {desc} | {sug} | {sources} | {conf} |")
        lines.append("")

    # ── 异常区 ──
    if errors:
        lines.append("## ⚠️ 审查警告\n")
        for err in errors:
            lines.append(f"- {err}")
        lines.append("")

    return "\n".join(lines)
