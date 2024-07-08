# -*- coding: utf-8 -*-
import configparser
import os

from text2vec import SentenceModel, Word2Vec


def compute_emb(model: str, sentences: list[str]):
    """本地/远端加载预训练模型，得到句子的向量表示"""
    model = SentenceModel(model)
    sentence_embeddings = model.encode(sentences, show_progress_bar=True, normalize_embeddings=True)
    for sentence, embedding in zip(sentences, sentence_embeddings):
        # print("Sentence:", sentence)
        # print("Embedding shape:", embedding[:10])
        yield [sentence, embedding.ravel().tolist()]


def get_model(model: str) -> str:
    """
    根据指定的模型名称，从配置文件中读取相对路径并解析出模型的绝对路径

    Args:
        model: (str) 配置文件中的模型名称key，用于获取模型的相对路径
    Returns:
        模型路径
    """
    # 获取配置文件的路径
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    config_path = os.path.join(project_root, 'static', 'config', 'llm_config.ini')
    config = configparser.ConfigParser()
    with open(config_path, 'r', encoding='utf-8') as f:
        config.read_file(f)
    model = config['model'][model]

    if model.startswith('model/'):
        # 构建模型的绝对路径
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        model_absolute_path = os.path.join(current_file_path, '..', '..', model)
        model = os.path.normpath(model_absolute_path)
    return model
