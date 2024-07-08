from src.llm.openai_llm import OpenAILLM
from src.llm.zhipu_llm import ZhiPuLLM


class LLMFactory:
    @classmethod
    def get_openai_factory(cls):
        return OpenAILLM()

    @classmethod
    def get_zhipu_factory(cls):
        return ZhiPuLLM()
