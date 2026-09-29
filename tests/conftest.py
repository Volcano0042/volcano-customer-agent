# -*- coding: utf-8 -*-
"""pytest 全局前置配置。

测试用例走离线 mock 聊天模型，同样不应依赖外部 Embedding API：
这里在 backend.config 首次读取环境前强制 EMBED_PROVIDER=none，
让 search_faq 退回纯词面检索——既无网络、结果也确定可复现。
精排同理关闭：强制 RERANK_PROVIDER=none，reranker 直接跳过、保持召回原序。
查询改写为纯规则、无网络，可安全保留。
真实 Agentic RAG 链路（Embedding + Rerank）通过正常启动服务、在聊天页问政策类问题来验证。
"""
import os

os.environ["EMBED_PROVIDER"] = "none"
os.environ["RERANK_PROVIDER"] = "none"
# 评测裁判走真实模型 API：测试里清空裁判配置，Judge.available=False，全程离线确定性判定。
os.environ["EVAL_JUDGE_MODEL"] = ""
os.environ["EVAL_JUDGE_BASE_URL"] = ""
