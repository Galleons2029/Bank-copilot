# -*- coding: utf-8 -*-
# @Time   : 2025/8/4 13:55
# @Author : Galleons
# @File   : dependency.py

"""
依赖配置脚本
"""
from langfuse import Langfuse
from app.configs import agent_config as settings

# Centralized Langfuse client configured from environment via AgentConfig
langfuse = Langfuse(
    public_key=settings.LANGFUSE_PUBLIC_KEY,
    secret_key=settings.LANGFUSE_SECRET_KEY,
    host=settings.LANGFUSE_HOST,
)