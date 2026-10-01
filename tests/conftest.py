# -*- coding: utf-8 -*-
"""pytest 全局前置配置。

在 backend.config 首次读取环境前强制 EMBED_PROVIDER / RERANK_PROVIDER=none，让
search_faq 退回纯词面检索、reranker 跳过：测试全程离线且结果可复现。查询改写为
纯规则、无网络，保留。真实 RAG 链路（Embedding + Rerank）由启动服务后在聊天页提问验证。
"""
import os

os.environ["EMBED_PROVIDER"] = "none"
os.environ["RERANK_PROVIDER"] = "none"
# 评测裁判走真实模型 API：测试里清空裁判配置，Judge.available=False，全程离线确定性判定。
os.environ["EVAL_JUDGE_MODEL"] = ""
os.environ["EVAL_JUDGE_BASE_URL"] = ""
