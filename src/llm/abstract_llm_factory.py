import abc


class AbstractLLMFactory(abc.ABC):

    @abc.abstractmethod
    def get_chat_llm(self):
        ...

    @abc.abstractmethod
    def get_embedding(self):
        ...
