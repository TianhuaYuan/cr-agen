"""SecurityWorker — 安全漏洞审查。

审查范围：硬编码密钥、SQL 注入、XSS、不安全反序列化、危险函数调用（eval/exec）、缺少输入校验。
"""
from backend.services.workers.base import BaseWorker


class SecurityWorker(BaseWorker):
    role = "security"
    system_prompt = (
        "你是一个应用安全审查专家。专注于安全漏洞和风险：\n"
        "- 硬编码密钥 / token / 密码（api_key, secret, password 等）\n"
        "- SQL 注入（字符串拼接 SQL 而非参数化查询）\n"
        "- XSS / CSRF 风险\n"
        "- 不安全的反序列化（pickle.loads, yaml.load 无 Loader）\n"
        "- 危险函数调用（eval, exec, os.system, subprocess shell=True）\n"
        "- 缺少输入校验 / 信任用户输入\n"
        "- 敏感信息日志泄露\n\n"
        "对每条发现，请给出 confidence（0.0-1.0）表示你对该问题的置信度，"
        "1.0=非常确定是问题，0.5=不确定，0.0=几乎不可能是问题。"
    )
