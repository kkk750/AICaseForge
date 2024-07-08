import os

from src.llm.abstract_llm_factory import AbstractLLMFactory
from src.llm.zhipu_sdk import ChatZhipuAI


class ZhiPuLLM(AbstractLLMFactory):
    def __init__(self):
        self.api_key = os.getenv("API_KEY")
        self.api_base = os.getenv("API_COMMON_BASE")
        if not self.api_base or not self.api_key:
            raise ValueError("API base URL or API key is not set.")
        self.client = None
        self.embedding_openai = None

    def get_llm(self):
        ...

    def get_chat_llm(self):
        if self.client is None:
            self.client = ChatZhipuAI(
                temperature=0.0,
                api_key=self.api_key,
                model="glm-4",
            )
        return self.client

    def get_embedding(self):
        ...