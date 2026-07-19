"""StateGraph 装配（Phase 5：容错机制接入）。

图结构：START → decompose → [条件路由] → (4 Worker 并行 | 熔断跳过) → aggregate → END。

Phase 5 新增三层容错：
1. **max_iterations 熔断**：decompose 递增 iteration_count，_route_after_decompose
   判断 iteration_count > max_iterations → 直达 aggregate，跳过 Workers。
   防止 Agent 死循环（即便当前是单趟架构，机制已就位，后续加迭代循环即生效）。
2. **Worker 超时**：BaseWorker.review() 内 asyncio.wait_for 超时降级（见 base.py）。
3. **Worker 异常不阻塞**：_make_worker_node 检测降级 finding → 追加 state.errors
   （errors 用 operator.add reducer，多 Worker 并发写不覆盖）→ report "审查警告"区标注。

Task 14.3: 节点级 tracing。每个节点（decompose / worker_* / aggregate）外包 span，
记录 state_keys / result_keys / latency_ms / error。NoOp tracer 零开销，Langfuse 模式
自动同步到 backend，形成完整调用链（trace → node span → llm_call span 嵌套）。
"""
import functools
import logging
import time

from langgraph.graph import END, START, StateGraph

from backend.core import tracing as tracing_mod
from backend.core.config import settings
from backend.services.aggregator.merge import aggregate_findings, split_by_confidence
from backend.services.aggregator.report import generate_report
from backend.services.supervisor.decompose import decompose_node
from backend.services.supervisor.state import SupervisorState
from backend.services.workers.registry import WORKERS as _WORKERS

logger = logging.getLogger(__name__)


def _trace_node(name: str, fn):
    """给节点函数包一层 tracing span（Task 14.3）。

    - 进入节点：start_span(name, metadata={"state_keys": [...]})
    - 节点成功：update({latency_ms, result_keys})
    - 节点异常：update({latency_ms, error}) + re-raise（不吞异常，让 LangGraph 上层处理）
    - with 退出：自动 end span（异常安全）

    functools.wraps 保留原函数 __name__/__doc__，便于 LangGraph 内部诊断。
    """
    @functools.wraps(fn)
    async def wrapped(state):
        tracer = tracing_mod.get_tracer()
        state_keys = list(state.keys()) if isinstance(state, dict) else []
        with tracer.start_span(name, metadata={"state_keys": state_keys}) as span:
            start = time.perf_counter()
            try:
                result = await fn(state)
                latency_ms = (time.perf_counter() - start) * 1000.0
                result_keys = list(result.keys()) if isinstance(result, dict) else None
                span.update({
                    "latency_ms": round(latency_ms, 2),
                    "result_keys": result_keys,
                })
                return result
            except Exception as exc:
                latency_ms = (time.perf_counter() - start) * 1000.0
                span.update({
                    "latency_ms": round(latency_ms, 2),
                    "error": str(exc),
                })
                raise

    return wrapped

# 降级 finding 的关键词（severity=info + description 含这些词 → 视为降级/异常）
_DEGRADED_KEYWORDS = ("异常", "超时", "解析失败", "timeout", "error", "降级")


def _is_degraded(finding: dict) -> bool:
    """判断 finding 是否为降级/异常产物（而非正常 info 级发现）。"""
    if finding.get("severity") != "info":
        return False
    desc = finding.get("description", "").lower()
    return any(kw in desc for kw in _DEGRADED_KEYWORDS)


def _make_worker_node(role: str):
    """把 Worker 实例包装成 LangGraph 节点函数。

    节点签名：async (state) -> {"worker_results": [...], "errors": [...]}
    Worker 的 review() 返回 list[dict]，放 worker_results（add reducer 累加）。
    Phase 5: 检测降级 finding → 追加 errors（add reducer 累加，不覆盖其他 Worker 的错误）。
    """

    worker = _WORKERS[role]

    async def _node(state: dict) -> dict:
        code = state.get("code", "")
        language = state.get("language", "python")
        findings = await worker.review(code, language)
        result: dict = {"worker_results": findings}
        # 降级 finding → 记录到 errors，供报告"审查警告"区标注
        degraded = [f for f in findings if _is_degraded(f)]
        if degraded:
            result["errors"] = [
                {"role": worker.role, "message": f.get("description", "")}
                for f in degraded
            ]
        return result

    _node.__name__ = f"worker_{role}"
    return _node


def _route_after_decompose(state: dict):
    """decompose 后的条件路由。

    iteration_count > max_iterations → 熔断，直达 aggregate（跳过 Workers）。
    否则 → 动态 fan-out：仅派发 decompose 产出的 tasks 中实际出现的、且系统支持的角色，
    使「LLM 动态拆解驱动并行调度」的叙事名副其实（小代码段只跑相关 Worker）。
    """
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    if iteration_count > max_iterations:
        logger.warning(
            "熔断触发：iteration_count=%s > max_iterations=%s，跳过 Workers",
            iteration_count, max_iterations,
        )
        return "aggregate"

    # 动态路由：从 tasks 提取去重后的角色，过滤到系统支持的 _WORKERS 范围内
    tasks = state.get("tasks", []) or []
    roles: list[str] = []
    for t in tasks:
        role = t.get("role") if isinstance(t, dict) else None
        if role in _WORKERS and role not in roles:
            roles.append(role)
    if not roles:
        # 兜底：无可用角色（理论上不会，decompose 降级总返回 4 个）则跑全部
        roles = list(_WORKERS)
    return [f"worker_{r}" for r in roles]


async def _aggregate(state: dict) -> dict:
    """真实聚合：去重 + 排序 → 渲染 Markdown 报告。

    Phase 5: 检测熔断状态，追加警告到 errors。
    Task 13.2: 按 confidence_threshold 拆分高/低置信度，报告分两区展示。
    """
    raw_findings = state.get("worker_results", [])
    errors = state.get("errors", [])
    # errors 可能是 list[str] 或 list[dict]（decompose 空代码 / worker 降级返回 dict）
    error_strs = []
    for e in errors:
        if isinstance(e, dict):
            error_strs.append(e.get("message", str(e)))
        else:
            error_strs.append(str(e))

    # 熔断检测
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    if iteration_count > max_iterations:
        error_strs.append(
            f"已达最大迭代次数（{max_iterations}），触发熔断，跳过 Worker 执行"
        )

    # Task 13.2: 置信度阈值过滤（从 state 读，允许 per-request 覆盖）
    threshold = state.get("confidence_threshold", settings.DEFAULT_CONFIDENCE_THRESHOLD)
    high, low = split_by_confidence(raw_findings, threshold)

    deduped = aggregate_findings(high)
    report = generate_report(deduped, language=state.get("language", "python"),
                             errors=error_strs, low_confidence=low)
    return {"report": report}


def build_supervisor_graph():
    """装配并编译 Supervisor 图。"""
    g = StateGraph(SupervisorState)
    # Task 14.3: 每个节点用 _trace_node 包裹，自动创建 tracing span。
    g.add_node("decompose", _trace_node("decompose", decompose_node))
    for role in _WORKERS:
        g.add_node(f"worker_{role}", _trace_node(f"worker_{role}", _make_worker_node(role)))
    g.add_node("aggregate", _trace_node("aggregate", _aggregate))

    g.add_edge(START, "decompose")
    # Phase 5: decompose 后条件路由（熔断 → aggregate；正常 → 4 Worker fan-out）
    g.add_conditional_edges("decompose", _route_after_decompose)
    for role in _WORKERS:
        g.add_edge(f"worker_{role}", "aggregate")
    g.add_edge("aggregate", END)
    return g.compile()
