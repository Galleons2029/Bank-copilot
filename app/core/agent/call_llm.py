from langchain_openai import ChatOpenAI
from app.configs import (
    llm_config,
    agent_config as settings,
)

client = ChatOpenAI(
            model=llm_config.LLM_MODEL,
            temperature=settings.DEFAULT_LLM_TEMPERATURE,
            api_key=settings.LLM_API_KEY,
            max_tokens=settings.MAX_TOKENS,
            base_url=llm_config.SILICON_BASE_URL,
        )
