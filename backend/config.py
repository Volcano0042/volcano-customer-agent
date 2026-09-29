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
    agent_name: str = "小V"
    brand_name: str = "Volcano"

    # ---- 上下文与推理轮次策略 ----
    context_trigger_ratio: float = 0.7       # 超过窗口该比例触发压缩（上限 0.9）
    context_reserve_ratio: float = 0.2       # 压缩后保留比例，须小于 trigger
    context_buffer_ratio: float = 0.15       # 触发前预警缓冲，须小于 trigger
    tool_result_limit: int = 4000            # 单条工具结果 token 上限
    max_react_iters: int = 12                # 单轮最多推理-行动轮次（默认 50）
    reply_token_budget: int = 32000          # 单轮加权 token 预算，超出即收尾
    reply_output_token_weight: float = 2.0   # 输出 token 权重

    # ---- 服务降级配置 ----
    degradation_enabled: bool = True
    fallback_provider: str = "mock"
    fallback_model: str = "offline-mock"
    fallback_timeout_seconds: float = 30.0

    # ---- RAG 知识库检索配置 ----
    embed_provider: str = "none"
    embed_model: str = ""
    embed_base_url: str = ""
    embed_timeout: float = 15.0
    embed_batch_size: int = 10           # 单次 API 请求最多几条文本
    rag_top_k: int = 3                   # 最终返回的知识条数
    rag_min_score: float = 0.15          # 混合得分下限，低于即视为未命中
    rag_alpha: float = 0.7               # 稠密向量权重，(1-alpha) 为词面权重


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


def _resolve_embed_provider() -> str:
    """向量化后端：显式指定优先，否则有 DashScope Key 就自动启用，无 Key 退回词面。"""
    provider = os.getenv("EMBED_PROVIDER", "").strip().lower()
    if provider in {"dashscope", "none"}:
        return provider
    return "dashscope" if os.getenv("DASHSCOPE_API_KEY") else "none"


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
        embed_provider=_resolve_embed_provider(),
        embed_model=os.getenv("EMBED_MODEL", "").strip(),
        embed_base_url=os.getenv("EMBED_BASE_URL", "").strip().rstrip("/"),
        embed_timeout=float(os.getenv("EMBED_TIMEOUT", "15.0")),
        embed_batch_size=int(os.getenv("EMBED_BATCH_SIZE", "10")),
        rag_top_k=int(os.getenv("RAG_TOP_K", "3")),
        rag_min_score=float(os.getenv("RAG_MIN_SCORE", "0.15")),
        rag_alpha=float(os.getenv("RAG_ALPHA", "0.7")),
    )