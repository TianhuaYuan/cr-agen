"""cr-agent 全局配置（从 1 号项目瘦身而来）。

只保留 cr-agent 需要的部分：LLM Chat 客户端 + 数据库 + 日志。
环境变量统一前缀 ``CR_AGENT_``，避免与系统/其他项目冲突。

``CHAT_*`` 默认空字符串：保证测试环境可直接 import（不强制要求真实密钥）；
真正启动服务时由 ``validate_required_settings()`` 在 lifespan 阶段 fail-fast 校验。
"""
import logging
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# APP_ENV 控制加载哪个 .env 文件：dev → .env.dev, test → .env.test, prod → .env.prod
_APP_ENV = os.getenv("APP_ENV", "dev")

# .env 文件放 backend/ 下；用绝对路径避免 CWD 依赖
_BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # ── LLM Chat 客户端（复用 1 号项目 OpenAI 兼容接口）──
    CHAT_API_KEY: str = ""
    CHAT_BASE_URL: str = "https://api.openai.com/v1"
    CHAT_MODEL: str = "gpt-4o-mini"

    # ── LLM 调用超时（秒）──
    # 单一来源：客户端（llm.py）与 Worker（base.py）都读它，消除散落的 60/120 双常量。
    # 默认 120s：给代码分析足够时间（实测单次 ~25-30s，并发更慢）；
    # Worker 每次调用会把它作为 asyncio.wait_for 的权威超时传下去。
    LLM_TIMEOUT: float = 120.0

    # ── 置信度阈值（Task 13.2）──
    # Aggregator 层过滤 confidence < 阈值的 finding，移到低置信度区。
    # 0.0 = 不过滤（保留所有 finding）；0.5 = 中等过滤；0.7 = 激进过滤。
    # 默认 0.0：先跑通流程，Task 13.3 阈值扫描后写入最优值。
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.0

    # ── Langfuse 链路追踪（Phase 14）──
    # 未配置时静默降级为 NoOp tracer（不报错，只是不追踪）。
    # 配置方式：在 .env 写 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST
    # 并 pip install langfuse，重启服务即生效。
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "http://localhost:3000"

    # ── 数据库（开发默认本地 SQLite async）──
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/cr_agent.db"

    # ── 日志 ──
    LOG_LEVEL: str = "INFO"

    # ── Webhook 安全 ──
    # 生产环境应设为 True：未配置 GITHUB_WEBHOOK_SECRET 时拒绝请求（防空 secret 免鉴权）。
    # 开发环境默认 False，允许不配 secret 直接跑通（便利）。
    WEBHOOK_SECRET_REQUIRED: bool = False
    GITHUB_WEBHOOK_SECRET: str = ""
    GITHUB_TOKEN: str = ""

    # ── API / MCP 鉴权（JWT）──
    # 默认关闭（开发态免鉴权，便利）。生产部署必须设 API_AUTH_REQUIRED=True
    # 并覆盖 JWT_SECRET（默认值是开发占位符，绝不可用于生产）与 API_KEY。
    # 逻辑与 WEBHOOK_SECRET_REQUIRED 一致：fail-open 开发友好，生产靠配置开关收紧。
    API_AUTH_REQUIRED: bool = False
    JWT_SECRET: str = "dev-insecure-secret-change-me-0000000000"  # 生产必须覆盖（≥32 字节只是降噪）
    API_KEY: str = ""  # 获取 JWT 的凭证；API_AUTH_REQUIRED=True 时必填，否则 /auth/token 拒发
    JWT_EXPIRE_MINUTES: int = 30

    model_config = SettingsConfigDict(
        env_prefix="CR_AGENT_",
        # 先加载 .env（通用），再加载 .env.{APP_ENV}（环境特定，覆盖前者）
        # 绝对路径 → 不受 CWD 影响
        env_file=(
            str(_BACKEND_DIR / ".env"),
            str(_BACKEND_DIR / f".env.{_APP_ENV}"),
        ),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )


settings = Settings()


# 启动期必须齐全的关键配置（缺失则启动失败，避免带着错误配置跑起来）
_REQUIRED_NON_EMPTY = ("CHAT_API_KEY", "CHAT_BASE_URL", "CHAT_MODEL")


def validate_required_settings() -> None:
    """校验关键环境变量/配置是否齐全。

    缺失则在启动期 raise 清晰错误，配合 lifespan 调用实现 fail-fast。
    """
    missing = [name for name in _REQUIRED_NON_EMPTY if not getattr(settings, name, "").strip()]
    if missing:
        raise RuntimeError(
            "启动配置校验失败，缺少以下必要环境变量/配置："
            + ", ".join(missing)
            + "。请在对应 .env 文件中补齐后再启动服务。"
        )
    logger.info("配置校验通过：关键环境变量/配置齐全（ENVIRONMENT=%s）", _APP_ENV)
