# -*- coding: utf-8 -*-
"""
Instructor agent workflow.

This module rebuilds the instructor agent described in docs/agent.ipynb into a
multi-stage LangGraph that mirrors the Bank-Copilot tasks:
1. Classify the user's intent across onboarding,检索和总分不平推理。
2. 根据任务类型与上下文动态决定是否检索知识。
3. 生成符合“Learn + Build + Assess”结构的课程/分析结果。
"""

from __future__ import annotations

import os
from typing import Literal, Sequence

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

from app.configs import agent_config, llm_config

try:  # Optional dependency; allow LangGraph server to boot without Qdrant
    from app.core.rag.retriever import VectorRetriever  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for light deployments
    VectorRetriever = None

# --- Domain guidance -----------------------------------------------------------------
PROJECT_CONTEXT = (
    "Bank-Copilot 面向银行财会，解决三个典型任务："
    "① 新员工/非财会同事培训；② 按机构-科目-币种-日期等多维度进行分录穿透检索；"
    "③ 在总账与分户之间排查2025-06-01~2025-06-10的总分不平。"
    "输出需要兼顾术语解释、流程拆解以及差异定位建议，并保持严谨、可执行。"
)

CURRICULUM_STYLE = (
    "最终回答请包含 Learn（知识点/概念）、Build（实操/流程/SQL/图示）、"
    "Assess（校验、测试、异常诊断）三个分段。必要时补充 Next Actions 与需用户补充的信息。"
    "多维检索问题要给出字段映射或步骤表；总分不平需说明类型、定位方法与根因假设。"
)

DEFAULT_COLLECTION = os.getenv("TRAINING_QDRANT_COLLECTION", "zsk_test1")

# --- LLM setup -----------------------------------------------------------------------
MODEL_NAME = agent_config.LLM_MODEL or llm_config.LLM_MODEL or "deepseek-ai/DeepSeek-V3.2"
API_KEY = agent_config.LLM_API_KEY or llm_config.SILICON_KEY or os.getenv("LLM_API_KEY") or os.getenv("API_KEY")
BASE_URL = llm_config.SILICON_BASE_URL or os.getenv("LLM_BASE_URL")
MAX_TOKENS = agent_config.MAX_TOKENS

common_kwargs = {
    "model": MODEL_NAME,
    "api_key": API_KEY,
    "max_tokens": MAX_TOKENS,
}
if BASE_URL:
    common_kwargs["base_url"] = BASE_URL

analysis_model = ChatOpenAI(temperature=0, **common_kwargs)
planner_model = ChatOpenAI(temperature=0, **common_kwargs)
response_model = ChatOpenAI(temperature=0.2, **common_kwargs)

# --- Schemas -------------------------------------------------------------------------


class IntentAnalysis(BaseModel):
    """Structured representation of the instructor intent analysis."""

    task_type: Literal["onboarding", "multidimensional_lookup", "ledger_reasoning"] = Field(description="任务类型：onboarding/检索/总分不平排查")
    task_summary: str = Field(description="一句话总结用户需求与成效标准")
    needs_context: bool = Field(description="是否需要补充知识库/上下文")
    knowledge_focus: list[str] = Field(
        default_factory=list,
        description="检索或回答需要覆盖的关键实体/字段/制度",
    )
    deliverable_tone: str = Field(description="输出应当呈现的语气或表达方式，例如表格/步骤/故事线")
    follow_up_questions: list[str] = Field(default_factory=list, description="若信息不足，需要向用户确认的问题")


class CurriculumPlan(BaseModel):
    """Three-stage training/diagnosis plan."""

    learning_objectives: list[str] = Field(description="Learning 阶段要解决的目标")
    learn_path: list[str] = Field(description="学习材料、关键解释或上下文补充")
    build_path: list[str] = Field(description="实操流程/SQL/图表/案例步骤")
    assess_path: list[str] = Field(description="测试题、验收指标、异常诊断步骤")
    risk_notes: list[str] = Field(description="对潜在风险、依赖和下一步建议的提醒", default_factory=list)


class InstructorState(TypedDict):
    """LangGraph state for the instructor agent."""

    messages: Annotated[list[AnyMessage], add_messages]
    intent: dict | None
    knowledge_context: list[str]
    plan: dict | None
    llm_calls: int


# --- Prompt definitions --------------------------------------------------------------
intent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"你是Bank-Copilot的教练协调器，需理解上下文并挑选适合的教练路径。{PROJECT_CONTEXT} "
            "输出JSON以标记任务、是否需要知识检索以及要追问的缺失信息。",
        ),
        (
            "human",
            "历史对话:\n{history}\n\n当前用户最新输入:\n{question}\n请依据3个核心场景给出最贴合的任务类型。",
        ),
    ]
)

plan_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"你是Bank-Copilot的课程架构师。结合识别出的任务、学员画像以及检索到的知识，设计一个 Learn/Build/Assess 三段式方案。{PROJECT_CONTEXT}",
        ),
        (
            "human",
            "用户问题: {question}\n"
            "任务概要: {task_summary}\n"
            "知识重点: {focus}\n"
            "可用上下文:\n{context}\n"
            "若信息不足，仍需给出在 Learn/Build/Assess 阶段需要补齐的要素。",
        ),
    ]
)

response_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"你是银行财会领域的教练型智能体，输出需严谨且行动导向。{PROJECT_CONTEXT}\n"
            f"{CURRICULUM_STYLE}\n"
            "若知识检索为空，要透明告知并基于现有信息给出建议。"
            "回答默认使用中文，可穿插表格/列表帮助理解。",
        ),
        (
            "human",
            "用户问题: {question}\n"
            "任务识别: {task_summary}\n"
            "课程计划:\n{plan_text}\n"
            "可引用知识:\n{context}\n"
            "历史对话:\n{history}\n"
            "需澄清的问题: {follow_ups}\n"
            "请生成结论并明确下一步行动。",
        ),
    ]
)

intent_chain = intent_prompt | analysis_model.with_structured_output(IntentAnalysis)
plan_chain = plan_prompt | planner_model.with_structured_output(CurriculumPlan)
response_chain = response_prompt | response_model

# --- Helper utilities ----------------------------------------------------------------


def _message_to_text(message: BaseMessage) -> str:
    """Normalize LangChain message content into plain text."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "\n".join(part for part in text_parts if part)
    return str(content)


def get_latest_user_question(messages: Sequence[BaseMessage]) -> str:
    """Return the latest human utterance."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return _message_to_text(message).strip()
    raise ValueError("No human message found in conversation history.")


def render_chat_history(messages: Sequence[BaseMessage], limit: int = 6) -> str:
    """Render the most recent human/assistant messages for prompt conditioning."""
    filtered: list[tuple[str, str]] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            filtered.append(("用户", _message_to_text(msg)))
        elif isinstance(msg, AIMessage):
            filtered.append(("助手", _message_to_text(msg)))
    trimmed = filtered[-limit:]
    return "\n".join(f"{speaker}: {text}" for speaker, text in trimmed) or "无历史上下文"


def format_context(chunks: Sequence[str] | None) -> str:
    """Format retrieved knowledge chunks."""
    if not chunks:
        return "无可用知识片段"
    return "\n---\n".join(chunks)


def format_plan(plan: dict | None) -> str:
    """Convert CurriculumPlan dict into friendly text."""
    if not plan:
        return "未生成课程计划"
    sections = []
    for key, title in [
        ("learning_objectives", "Learning Objectives"),
        ("learn_path", "Learn"),
        ("build_path", "Build"),
        ("assess_path", "Assess"),
        ("risk_notes", "Risks / Next steps"),
    ]:
        values = plan.get(key) or []
        if values:
            bullet = "\n".join(f"- {item}" for item in values)
        else:
            bullet = "- 暂无信息"
        sections.append(f"{title}:\n{bullet}")
    return "\n\n".join(sections)


def build_retrieval_query(messages: Sequence[BaseMessage], intent: dict | None) -> str:
    """Compose a retrieval query using the last user need and intent focus."""
    user_question = get_latest_user_question(messages)
    focus = ", ".join((intent or {}).get("knowledge_focus", []))
    return f"{user_question}\n检索聚焦: {focus or '财会培训/流程/差异定位'}"


def should_fetch_context(state: InstructorState) -> Literal["retrieve", "skip"]:
    """Routing helper after intent classification."""
    intent = state.get("intent") or {}
    if intent.get("needs_context"):
        return "retrieve"
    return "skip"


# --- Graph nodes ---------------------------------------------------------------------
def analyze_intent(state: InstructorState) -> InstructorState:
    question = get_latest_user_question(state["messages"])
    history_text = render_chat_history(state["messages"][:-1])
    analysis = intent_chain.invoke({"history": history_text, "question": question})
    current_calls = state.get("llm_calls", 0) + 1
    return {"intent": analysis.model_dump(), "llm_calls": current_calls}


def retrieve_context(state: InstructorState) -> InstructorState:
    query = build_retrieval_query(state["messages"], state.get("intent"))
    context_chunks: list[str]
    if VectorRetriever is None:
        context_chunks = ["知识库检索暂不可用（缺少 qdrant_client 依赖）。请先安装依赖或手动提供参考资料。"]
    else:
        try:
            retriever = VectorRetriever(query)
            hits = retriever.retrieve_top_k(
                k=6,
                collections=[DEFAULT_COLLECTION],
                to_expand_to_n_queries=3,
            )
            context_chunks = retriever.rerank(hits=hits, keep_top_k=3)
        except Exception as exc:  # pragma: no cover - defensive
            context_chunks = [f"知识检索失败：{exc}"]
    return {"knowledge_context": context_chunks}


def design_curriculum(state: InstructorState) -> InstructorState:
    question = get_latest_user_question(state["messages"])
    intent = state.get("intent") or {}
    context_text = format_context(state.get("knowledge_context"))
    focus = ", ".join(intent.get("knowledge_focus", []))
    plan = plan_chain.invoke(
        {
            "question": question,
            "task_summary": intent.get("task_summary", ""),
            "focus": focus or "科目编码 / 机构 / 币种 / 核算区间",
            "context": context_text,
        }
    )
    current_calls = state.get("llm_calls", 0) + 1
    return {"plan": plan.model_dump(), "llm_calls": current_calls}


def craft_response(state: InstructorState) -> InstructorState:
    question = get_latest_user_question(state["messages"])
    intent = state.get("intent") or {}
    context_text = format_context(state.get("knowledge_context"))
    plan_text = format_plan(state.get("plan"))
    history_text = render_chat_history(state["messages"][:-1])
    followups = intent.get("follow_up_questions") or []
    ai_message = response_chain.invoke(
        {
            "question": question,
            "task_summary": intent.get("task_summary", "未识别任务"),
            "plan_text": plan_text,
            "context": context_text,
            "history": history_text,
            "follow_ups": "；".join(followups) or "暂无",
        }
    )
    current_calls = state.get("llm_calls", 0) + 1
    return {"messages": [ai_message], "llm_calls": current_calls}


# --- Graph compilation ---------------------------------------------------------------
graph_builder = StateGraph(InstructorState)
graph_builder.add_node("analyze_intent", analyze_intent)
graph_builder.add_node("retrieve_context", retrieve_context)
graph_builder.add_node("design_curriculum", design_curriculum)
graph_builder.add_node("respond", craft_response)

graph_builder.set_entry_point("analyze_intent")
graph_builder.add_conditional_edges(
    "analyze_intent",
    should_fetch_context,
    {
        "retrieve": "retrieve_context",
        "skip": "design_curriculum",
    },
)
graph_builder.add_edge("retrieve_context", "design_curriculum")
graph_builder.add_edge("design_curriculum", "respond")
graph_builder.add_edge("respond", END)

instructor_agent = graph_builder.compile()
