import os

from langchain_openai.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings

from common.config_parse import ConfigParse
from common.file_path import FilePath
from src.llm.abstract_llm_factory import AbstractLLMFactory


class OpenAILLM(AbstractLLMFactory):
    def __init__(self):
        _config_parser = ConfigParse(FilePath.config_file_path)
        self.api_key = os.getenv("API_KEY", default=_config_parser.get_api_key())
        self.api_base = os.getenv("API_COMMON_BASE", default='http://gpt-proxy.jd.com/gateway/common')
        if not self.api_base or not self.api_key:
            raise ValueError("API base URL or API key is not set.")
        self.client = None
        self.embedding_openai = None

    def get_chat_llm(self):
        if self.client is None:
            self.client = ChatOpenAI(
                openai_api_base=self.api_base,
                openai_api_key=self.api_key,
                model_name="gpt-4-1106-preview",
                temperature=0.0
            )
        return self.client

    def get_gpt3(self):
        if self.client is None:
            self.client = ChatOpenAI(
                openai_api_base=self.api_base,
                openai_api_key=self.api_key,
                model_name="gpt-3.5-turbo-16k",
                temperature=0.4,
            )
        return self.client

    def get_gpt4v(self):
        if self.client is None:
            self.client = ChatOpenAI(
                openai_api_base=self.api_base,
                openai_api_key=self.api_key,
                model_name="gpt-4-vision-preview",
                temperature=0.0,
                max_tokens=4096
            )
        return self.client

    def get_gpt4o(self):
        if self.client is None:
            self.client = ChatOpenAI(
                openai_api_base=self.api_base,
                openai_api_key=self.api_key,
                model_name="gpt-4o",
            )
        return self.client

    def get_embedding(self):
        if self.embedding_openai is None:
            self.embedding_openai = OpenAIEmbeddings(
                openai_api_base=self.api_base,
                openai_api_key=self.api_key,
                model="text-embedding-ada-002"
            )
        return self.embedding_openai
