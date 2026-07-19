"""sample_good_python.py —— 干净的 Python 代码（对照组）。

规范：类型注解、小函数、异常处理、环境变量配置、列表推导。
"""
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, str]:
    """从文件加载配置，异常安全。"""
    try:
        lines = config_path.read_text(encoding="utf-8").splitlines()
        return {
            k.strip(): v.strip()
            for line in lines
            if "=" in line and not line.startswith("#")
            for k, v in [line.split("=", 1)]
        }
    except FileNotFoundError:
        logger.warning("配置文件不存在: %s", config_path)
        return {}


def filter_active_users(users: list[dict]) -> list[dict]:
    """过滤活跃用户，用列表推导而非循环 append。"""
    return [u for u in users if u.get("active", False)]


def calculate_stats(numbers: list[float]) -> dict[str, float]:
    """计算统计指标，函数单一职责。"""
    if not numbers:
        return {"count": 0, "sum": 0.0, "avg": 0.0}

    total = sum(numbers)
    return {
        "count": len(numbers),
        "sum": total,
        "avg": total / len(numbers),
    }


def main() -> None:
    """主入口：读取环境变量 → 加载配置 → 处理数据。"""
    api_key = os.environ.get("API_KEY", "")
    if not api_key:
        logger.warning("API_KEY 未设置，使用默认配置")

    config = load_config(Path("config.ini"))
    logger.info("加载配置: %d 项", len(config))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
