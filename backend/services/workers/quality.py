"""QualityWorker — 代码规范 / 风格审查。

审查范围：命名规范、注释质量、函数长度、缩进一致性、PEP8/ESLint 常见违规。
"""
from backend.services.workers.base import BaseWorker


class QualityWorker(BaseWorker):
    role = "quality"
    system_prompt = (
        "你是一个代码质量审查专家。专注于代码规范和风格问题：\n"
        "- 命名规范（变量/函数/类命名是否清晰、是否遵循语言惯例）\n"
        "- 注释质量（是否有必要的注释、注释是否过时）\n"
        "- 函数长度与复杂度（是否过长、参数是否过多）\n"
        "- 缩进与格式一致性\n"
        "- PEP8 / ESLint / Google Style 等常见违规项\n\n"
        "对每条发现，请给出 confidence（0.0-1.0）表示你对该问题的置信度，"
        "1.0=非常确定是问题，0.5=不确定，0.0=几乎不可能是问题。"
    )
