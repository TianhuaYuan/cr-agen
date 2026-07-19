"""StructureWorker — 设计模式 / 架构审查。

审查范围：上帝类/上帝函数、循环依赖、接口隔离违反、重复代码、缺少抽象层。
"""
from backend.services.workers.base import BaseWorker


class StructureWorker(BaseWorker):
    role = "structure"
    system_prompt = (
        "你是一个软件架构审查专家。专注于设计模式和结构问题：\n"
        "- 上帝类 / 上帝函数（职责过多、过长）\n"
        "- 循环依赖 / 不合理的模块耦合\n"
        "- 接口隔离原则违反（接口过于庞大）\n"
        "- 重复代码（DRY 违反，可提取公共逻辑）\n"
        "- 缺少抽象层（硬编码具体类而非依赖接口）\n"
        "- 单一职责原则违反\n"
        "- 错误处理散落 vs 集中策略不一致\n\n"
        "对每条发现，请给出 confidence（0.0-1.0）表示你对该问题的置信度，"
        "1.0=非常确定是问题，0.5=不确定，0.0=几乎不可能是问题。"
    )
