# -*- coding: utf-8 -*-
"""全局配置（从 .env / 环境变量加载）。"""
import os
from dataclasses import dataclass, field
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
    # repr=False：密钥不进 repr
    dashscope_api_key: str | None = field(repr=False)
    openai_api_key: str | None = field(repr=False)
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

    # ---- 多智能体（监督者 + 专家子 agent，agent-as-tool）----
    multi_agent_enabled: bool = False    # 关闭时走单 agent 主链路（默认，离线可跑通）
    specialist_max_iters: int = 6        # 单个专家子 agent 的最大推理-行动轮次
    supervisor_model: str = ""           # 监督者用模型，空=与主模型同款
    passthrough_single_delegation: bool = True   # 只委派一个专家时直接采用其答案，跳过终答重写

    # ---- RAG 知识库检索配置 ----
    embed_provider: str = "none"
    embed_model: str = ""
    embed_base_url: str = ""
    embed_timeout: float = 15.0
    embed_batch_size: int = 10           # 单次 API 请求最多几条文本
    rag_top_k: int = 3                   # 最终返回的知识条数
    rag_min_score: float = 0.15          # 混合得分下限，低于即视为未命中
    rag_alpha: float = 0.7               # 稠密向量权重，(1-alpha) 为词面权重

    # ---- Agentic RAG：查询改写 + 精排 + 置信度自检 ----
    query_rewrite_enabled: bool = True   # 是否对电商同义词做 multi-query 扩展再召回
    rag_recall_k: int = 8                # 进入精排的候选短列表大小（召回放宽，精排收敛）
    rerank_provider: str = "none"        # "dashscope" | "none"；配好 key+base_url+model 才启用
    rerank_model: str = ""               # 精排模型名（值走 .env，代码不写死厂商）
    rerank_base_url: str = ""            # 精排接口完整 URL（值走 .env）
    rerank_timeout: float = 8.0          # 精排请求超时（秒）
    rag_high_conf_bar: float = 0.20      # 精排得分下限，低于则标 low 置信并提示澄清/转人工

    # ---- Agentic RAG：反思式检索（Self-RAG / CRAG：检索评估器 + 改写重检）----
    rag_reflect_enabled: bool = False    # 低置信时是否让模型判「够不够答」并改写重检
    reflect_model: str = ""              # 反思评估器用的对话模型名（值走 .env）
    reflect_base_url: str = ""           # OpenAI 兼容 chat 端点 base_url（值走 .env）
    reflect_timeout: float = 8.0         # 单次反思请求超时（秒）
    reflect_max_rounds: int = 1          # 最多反思/重检轮次，控制延迟

    # ---- 端到端 Agent 评测（真实模型 + LLM-as-judge）----
    eval_judge_model: str = ""           # 裁判模型名（值走 .env）
    eval_judge_base_url: str = ""        # OpenAI 兼容 chat 端点 base_url（值走 .env）
    eval_judge_timeout: float = 120.0    # 裁判请求超时（秒），裁判是推理型模型、耗时波动大，给足余量
    eval_case_timeout: float = 90.0      # 单条 case 驱动超时（秒），卡住的真实调用只算失败不拖垮整轮
    eval_temperature: float = 0.0        # 采样温度，0 求可复现
    eval_max_cases: int = 0              # 本次评测最多跑几条，0=全量
    eval_cache_enabled: bool = True      # 按 case 指纹缓存原始输出，重跑免重复计费


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


def _resolve_rerank_provider() -> str:
    """精排后端：显式指定优先；否则配了精排模型且有 DashScope Key 就自动启用，任一缺失则 none（跳过精排）。"""
    provider = os.getenv("RERANK_PROVIDER", "").strip().lower()
    if provider in {"dashscope", "none"}:
        return provider
    if os.getenv("RERANK_MODEL", "").strip() and os.getenv("DASHSCOPE_API_KEY"):
        return "dashscope"
    return "none"


def _resolve_reflect_enabled() -> bool:
    """反思式检索开关：显式设置优先；否则配了反思模型 + 端点且有 DashScope Key 就自动启用。"""
    v = os.getenv("RAG_REFLECT_ENABLED", "").strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    return bool(
        os.getenv("RAG_REFLECT_MODEL", "").strip()
        and os.getenv("RAG_REFLECT_BASE_URL", "").strip()
        and os.getenv("DASHSCOPE_API_KEY")
    )


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
        multi_agent_enabled=_parse_bool(os.getenv("MULTI_AGENT_ENABLED", "false")),
        specialist_max_iters=int(os.getenv("SPECIALIST_MAX_ITERS", "6")),
        supervisor_model=os.getenv("SUPERVISOR_MODEL", "").strip(),
        passthrough_single_delegation=_parse_bool(
            os.getenv("PASSTHROUGH_SINGLE_DELEGATION", "true")
        ),
        embed_provider=_resolve_embed_provider(),
        embed_model=os.getenv("EMBED_MODEL", "").strip(),
        embed_base_url=os.getenv("EMBED_BASE_URL", "").strip().rstrip("/"),
        embed_timeout=float(os.getenv("EMBED_TIMEOUT", "15.0")),
        embed_batch_size=int(os.getenv("EMBED_BATCH_SIZE", "10")),
        rag_top_k=int(os.getenv("RAG_TOP_K", "3")),
        rag_min_score=float(os.getenv("RAG_MIN_SCORE", "0.15")),
        rag_alpha=float(os.getenv("RAG_ALPHA", "0.7")),
        query_rewrite_enabled=_parse_bool(os.getenv("QUERY_REWRITE_ENABLED", "true")),
        rag_recall_k=int(os.getenv("RAG_RECALL_K", "8")),
        rerank_provider=_resolve_rerank_provider(),
        rerank_model=os.getenv("RERANK_MODEL", "").strip(),
        rerank_base_url=os.getenv("RERANK_BASE_URL", "").strip(),
        rerank_timeout=float(os.getenv("RERANK_TIMEOUT", "8.0")),
        rag_high_conf_bar=float(os.getenv("RAG_HIGH_CONF_BAR", "0.20")),
        rag_reflect_enabled=_resolve_reflect_enabled(),
        reflect_model=os.getenv("RAG_REFLECT_MODEL", "").strip(),
        reflect_base_url=os.getenv("RAG_REFLECT_BASE_URL", "").strip().rstrip("/"),
        reflect_timeout=float(os.getenv("RAG_REFLECT_TIMEOUT", "8.0")),
        reflect_max_rounds=int(os.getenv("RAG_REFLECT_MAX_ROUNDS", "1")),
        eval_judge_model=os.getenv("EVAL_JUDGE_MODEL", "").strip(),
        eval_judge_base_url=os.getenv("EVAL_JUDGE_BASE_URL", "").strip().rstrip("/"),
        eval_judge_timeout=float(os.getenv("EVAL_JUDGE_TIMEOUT", "120.0")),
        eval_case_timeout=float(os.getenv("EVAL_CASE_TIMEOUT", "90.0")),
        eval_temperature=float(os.getenv("EVAL_TEMPERATURE", "0.0")),
        eval_max_cases=int(os.getenv("EVAL_MAX_CASES", "0")),
        eval_cache_enabled=_parse_bool(os.getenv("EVAL_CACHE_ENABLED", "true")),
    )