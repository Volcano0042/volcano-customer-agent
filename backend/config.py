# -*- coding: utf-8 -*-
"""全局配置（从 .env / 环境变量加载）。"""
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT_DIR / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)


@dataclass(frozen=True)
class Settings:
    """服务运行配置。"""

    # "dashscope" | "openai" | "mock"
    model_provider: str
    model_name: str
    dashscope_api_key: str | None
    openai_api_key: str | None
    openai_base_url: str | None

    host: str = "0.0.0.0"
    port: int = 8000
    session_ttl_minutes: int = 120
    max_sessions: int = 200

    # 智能体对外展示名称
    agent_name: str = "小V"
    # 品牌名（会写进系统提示词）
    brand_name: str = "Volcano"

    # ---- 上下文与推理轮次策略 ----
    # 上下文占模型窗口的比例超过该值时触发压缩（SDK 上限 0.9）
    context_trigger_ratio: float = 0.7
    # 压缩后保留的上下文比例，必须小于 trigger_ratio
    context_reserve_ratio: float = 0.2
    # 压缩阈值之前的预警缓冲区，必须小于 trigger_ratio
    context_buffer_ratio: float = 0.15
    # 单条工具结果的 token 上限，超出即截断，防止单次查询撑爆上下文
    tool_result_limit: int = 4000
    # 一轮回复内最多的"推理-行动"轮次（SDK 默认 50，客服场景过大）
    max_react_iters: int = 12
    # 单轮回复的加权 token 预算（输入+输出）。超出后中间件会注入收尾提示
    # 并把 tool_choice 置为 none，逼模型用现有信息作答，避免失控烧钱。
    reply_token_budget: int = 32000
    # 输出 token 的权重（通常比输入贵），用于加权计算预算消耗
    reply_output_token_weight: float = 2.0

    # ---- 服务降级配置 ----
    degradation_enabled: bool = True
    fallback_provider: str = "mock"
    fallback_model: str = "offline-mock"
    fallback_timeout_seconds: float = 30.0


def _parse_bool(v: str | None) -> bool:
    if not v:
        return False
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_provider() -> str:
    provider = os.getenv("MODEL_PROVIDER", "").strip().lower()
    if provider in {"dashscope", "openai", "mock"}:
        return provider
    if os.getenv("DASHSCOPE_API_KEY"):
        return "dashscope"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "mock"


def _resolve_model(provider: str) -> str:
    if provider == "dashscope":
        return os.getenv("DASHSCOPE_MODEL", "qwen-plus")
    if provider == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return "offline-mock"


@lru_cache
def get_settings() -> Settings:
    provider = _resolve_provider()
    return Settings(
        model_provider=provider,
        model_name=_resolve_model(provider),
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY") or None,
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        session_ttl_minutes=int(os.getenv("SESSION_TTL_MINUTES", "120")),
        max_sessions=int(os.getenv("MAX_SESSIONS", "200")),
        agent_name=os.getenv("AGENT_NAME", "小V"),
        brand_name=os.getenv("BRAND_NAME", "Volcano"),
        context_trigger_ratio=float(os.getenv("CONTEXT_TRIGGER_RATIO", "0.7")),
        context_reserve_ratio=float(os.getenv("CONTEXT_RESERVE_RATIO", "0.2")),
        context_buffer_ratio=float(os.getenv("CONTEXT_BUFFER_RATIO", "0.15")),
        tool_result_limit=int(os.getenv("TOOL_RESULT_LIMIT", "4000")),
        max_react_iters=int(os.getenv("MAX_REACT_ITERS", "12")),
        reply_token_budget=int(os.getenv("REPLY_TOKEN_BUDGET", "32000")),
        reply_output_token_weight=float(
            os.getenv("REPLY_OUTPUT_TOKEN_WEIGHT", "2.0")
        ),
        degradation_enabled=_parse_bool(os.getenv("DEGRADATION_ENABLED", "true")),
        fallback_provider=os.getenv("FALLBACK_PROVIDER", "mock").strip().lower(),
        fallback_model=os.getenv("FALLBACK_MODEL", "offline-mock"),
        fallback_timeout_seconds=float(os.getenv("FALLBACK_TIMEOUT_SECONDS", "30.0")),
    )