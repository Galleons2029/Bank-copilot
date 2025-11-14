# -*- coding: utf-8 -*-
# @Time   : 2025/8/4 13:55
# @Author : Galleons
# @File   : dependency.py

"""
依赖配置脚本
"""
from langfuse import Langfuse
from app.configs import agent_config as settings
from app.configs import db_config
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Centralized Langfuse client configured from environment via AgentConfig
langfuse = Langfuse(
    public_key=settings.LANGFUSE_PUBLIC_KEY,
    secret_key=settings.LANGFUSE_SECRET_KEY,
    host=settings.LANGFUSE_HOST,
)



DB_URL = db_config.POSTGRES_URL

engine = create_engine(
    DB_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()