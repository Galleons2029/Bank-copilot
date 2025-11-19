from langchain_openai import ChatOpenAI
from app.configs import (
    llm_config
)

client = ChatOpenAI(
            model=llm_config.LLM_MODEL,
            temperature=llm_config.DEFAULT_LLM_TEMPERATURE,
            api_key=llm_config.API_KEY,
            max_tokens=llm_config.MAX_TOKENS,
            base_url=llm_config.SILICON_BASE_URL,
        )

if __name__ == '__main__':
    response = client.invoke("你好！")
    print(response)