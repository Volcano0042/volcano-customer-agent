# -*- coding: utf-8 -*-
"""pytest 全局前置配置。

测试用例走离线 mock 聊天模型，同样不应依赖外部 Embedding API：
这里在 backend.config 首次读取环境前强制 EMBED_PROVIDER=none，
让 search_faq 退回纯词面检索——既无网络、结果也确定可复现。
真实 RAG 链路（DashScope Embedding）通过正常启动服务、在聊天页问政策类问题来验证。
"""
import os

os.environ["EMBED_PROVIDER"] = "none"
