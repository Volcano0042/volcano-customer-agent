# -*- coding: utf-8 -*-
"""RAG 知识库检索子系统。

对外只暴露 get_knowledge_base()：把 FAQ 条目向量化后建稠密索引，
检索时用「稠密向量 + 词面打分」混合排序返回最相关的若干条。
无 Embedding API Key（或调用失败）时自动退回纯词面检索，行为与旧实现一致。
"""
from .retriever import KnowledgeBase, get_knowledge_base

__all__ = ["KnowledgeBase", "get_knowledge_base"]
