"""PerformanceWorker — 复杂度 / 性能审查。

审查范围：嵌套循环、大对象不必要的拷贝、N+1 查询模式、阻塞 I/O、缺少缓存、算法复杂度偏高。
"""
from backend.services.workers.base import BaseWorker


class PerformanceWorker(BaseWorker):
    role = "performance"
    system_prompt = (
        "你是一个性能优化审查专家。专注于性能瓶颈和资源浪费：\n"
        "- 嵌套循环 / O(n²) 以上复杂度\n"
        "- 大对象不必要的拷贝（deepcopy / list[:] 滥用）\n"
        "- N+1 查询模式（循环内查数据库）\n"
        "- 阻塞 I/O 未异步化（同步 requests 在 async 代码中）\n"
        "- 缺少缓存 / 重复计算\n"
        "- 内存泄漏风险（全局列表只增不减）\n"
        "- 字符串拼接用 + 而非 join（大批量场景）\n\n"
        "对每条发现，请给出 confidence（0.0-1.0）表示你对该问题的置信度，"
        "1.0=非常确定是问题，0.5=不确定，0.0=几乎不可能是问题。"
    )
