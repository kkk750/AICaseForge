import asyncio
import re
from tqdm import tqdm

from src.doc_preprocess.md_splitter import split_markdown_further, split_markdown_by_header
from src.doc_preprocess.pdf2md import pdf2md
from src.knowledge_injection.doc2faq import _Generate_QA_From_Paras
from static.config.prompt_config import PromptConfig
from src.embedding.vearch_client import VearchClient
from src.embedding.text2vector import compute_emb, get_model


def inject_knowledge(file, space_name, analyze_images):
    """
    从PDF文档中提取文本，生成问答对，向量化后存入数据库。

    该函数首先读取PDF文件并清洗文本，然后根据文本内容生成问答对。根据is_summary参数决定，
    是将整个文档的每个段落生成一个总结后向量化，还是将文档的每个句子直接向量化。最后，
    将这些向量化的数据批量存储到指定的向量数据库中。

    Args:
        file: (FileStorage) PDF文件对象
        space_name: (str) 向量空间的名称，指定数据存储的目的地。
        analyze_images: (Bool) 指示是否调大模型对文档进行图片理解后后再向量化

    Returns:
        向量数据库插入操作的响应结果。
    """
    # PDF转md
    processed_text = pdf2md(doc=file, analyze_images=analyze_images)
    # 文本切割聚合
    paras_list = split_markdown_further(documents=split_markdown_by_header(processed_text), min_partition=128)
    # 生成FAQ
    # 由于vearch批量插入建议小于100条/次，因此将FAQ拆分成50/组
    data_list = []
    paras_list_grouped = [paras_list[i:i + 50] for i in range(0, len(paras_list), 50)]
    for group_idx, paras in tqdm(enumerate(paras_list_grouped), total=len(paras_list_grouped)):
        paras_qa_list = asyncio.run(_Generate_QA_From_Paras(paras, PromptConfig.FAQ))
        for idx, metadata in tqdm(enumerate(paras_qa_list), total=len(paras_qa_list)):
            for qa in metadata['qa_pair']:
                qa[0] = re.sub(r'^[?？.。\n]+|[?？.。\n]+$', '', qa[0])
                qa[1] = re.sub(r'^[?？.。\n]+|[?？.。\n]+$', '', qa[1])
            data_list.append({'content': metadata['para'].page_content,
                              'metadata': {
                                  'content_type': 'paragraph',
                                  'heading_hierarchy': {},
                                  'figure_list': [],
                                  'chunk_id': idx,
                                  'file_path': '',
                                  'keywords': [],
                                  'summary': '',
                                  'qa_list': metadata['qa_pair']
                              }})
    # 文本向量化，注入向量库
    vearch = VearchClient()
    model_name = get_model(model='t2v_model')
    for item in data_list:
        sentences = []
        qa_list = item['metadata']['qa_list']
        for qa in qa_list:
            question, answer = qa
            sentences.append(question.strip())  # 移除前后空白字符
            sentences.append(answer.strip())
        documents = []
        for sentence, embedding in compute_emb(model=model_name, sentences=sentences):
            documents.append({
                'feature': {
                    'feature': embedding
                },
                'sentence': sentence,
                'content': item['content'],
                'metadata': item['metadata']
            })
        response = vearch.insert_batch(db_name='llm_test_db_1', space_name=space_name, data_list=documents)
    return response
