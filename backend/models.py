# -*- coding: utf-8 -*-
"""聊天模型工厂：按配置返回 AgentScope 2.0 的 ChatModelBase 实例。

核心设计：
- create_chat_model: 创建主模型（dashscope / openai / mock），按配置复用实例
- create_fallback_model: 创建降级模型（默认 offline mock）
- create_degraded_model: 用 DegradedChatModel 包装主+备，实现自动降级
- create_supervisor_model: 创建多智能体监督者用的模型
"""
import threading
from dataclasses import replace

from agentscope.model import ChatModelBase

from .config import Settings
from .mock_model import OfflineChatModel

# (provider, model_name, 凭证) -> 模型实例
_MODEL_CACHE: dict[tuple, ChatModelBase] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _build_dashscope(settings: Settings) -> ChatModelBase:
    from agentscope.credential import DashScopeCredential
    from agentscope.model import DashScopeChatModel

    assert settings.dashscope_api_key, "缺少 DASHSCOPE_API_KEY"
    return DashScopeChatModel(
        credential=DashScopeCredential(api_key=settings.dashscope_api_key),
        model=settings.model_name,
        stream=True,
    )


def _build_openai(settings: Settings) -> ChatModelBase:
    from agentscope.credential import OpenAICredential
    from agentscope.model import OpenAIChatModel

    assert settings.openai_api_key, "缺少 OPENAI_API_KEY"
    return OpenAIChatModel(
        credential=OpenAICredential(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        ),
        model=settings.model_name,
        stream=True,
    )


def create_chat_model(settings: Settings) -> ChatModelBase:
    """根据配置创建（并按配置复用）主模型。"""
    return _shared_model(settings, settings.model_name)


def _shared_model(settings: Settings, model_name: str) -> ChatModelBase:
    """按 (provider, model, 凭证) 取模型实例，缺则新建并入缓存。"""
    key = (
        settings.model_provider,
        model_name,
        settings.dashscope_api_key,
        settings.openai_api_key,
        settings.openai_base_url,
    )
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is None:
            model = _build_uncached(settings, model_name)
            _MODEL_CACHE[key] = model
        return model


def _build_uncached(settings: Settings, model_name: str) -> ChatModelBase:
    """实例化模型；model_name 不是主模型时改写一份配置副本。"""
    if model_name != settings.model_name:
        settings = replace(settings, model_name=model_name)
    if settings.model_provider == "dashscope":
        return _build_dashscope(settings)
    if settings.model_provider == "openai":
        return _build_openai(settings)
    return OfflineChatModel()


def create_supervisor_model(settings: Settings) -> ChatModelBase:
    """创建监督者用的模型：配了 SUPERVISOR_MODEL 就用它，否则与主链路同款。

    配了快速模型时降级链为「快速模型 → 主模型」。
    """
    fast = (settings.supervisor_model or "").strip()
    if not fast or fast == settings.model_name:
        return create_degraded_model(settings)

    primary = _shared_model(settings, fast)
    if not settings.degradation_enabled:
        return primary

    from .degradation import DegradedChatModel

    return DegradedChatModel(
        primary=primary,
        fallback=create_chat_model(settings),
        timeout_seconds=settings.fallback_timeout_seconds,
    )


def create_fallback_model(settings: Settings) -> ChatModelBase:
    """创建降级模型（默认 offline mock，零依赖）。"""
    if settings.fallback_provider == "dashscope" and settings.dashscope_api_key:
        return _build_dashscope(
            Settings(
                model_provider="dashscope",
                model_name=settings.fallback_model,
                dashscope_api_key=settings.dashscope_api_key,
                openai_api_key=settings.openai_api_key,
                openai_base_url=settings.openai_base_url,
            )
        )
    if settings.fallback_provider == "openai" and settings.openai_api_key:
        return _build_openai(
            Settings(
                model_provider="openai",
                model_name=settings.fallback_model,
                dashscope_api_key=settings.dashscope_api_key,
                openai_api_key=settings.openai_api_key,
                openai_base_url=settings.openai_base_url,
            )
        )
    return OfflineChatModel()


def create_degraded_model(settings: Settings) -> ChatModelBase:
    """创建带服务降级的模型实例。

    - 若 degradation_enabled=True：返回 DegradedChatModel，主模型失败自动降级
    - 若 degradation_enabled=False：直接返回主模型（与旧行为一致）
    """
    primary = create_chat_model(settings)

    if not settings.degradation_enabled:
        return primary

    from .degradation import DegradedChatModel

    fallback = create_fallback_model(settings)
    return DegradedChatModel(
        primary=primary,
        fallback=fallback,
        timeout_seconds=settings.fallback_timeout_seconds,
    )
