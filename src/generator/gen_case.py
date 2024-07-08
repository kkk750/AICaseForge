from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import math
import tiktoken
from io import BytesIO
from pathlib import Path
from typing import Union, Optional

from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from werkzeug.datastructures import FileStorage

from common.file_path import FilePath
from src.generator.json2xmind import merge_json, fix_json_str, gen_xmind
from src.doc_preprocess.pdf2md import pdf2md
from static.config.prompt_config import PromptConfig
from src.generator.json2excel import gen_excel, parse_json
from src.doc_preprocess.cleaner import clean_output_tags, clean_empty_lines, clean
from src.doc_preprocess.md_splitter import MdDocument, split_markdown_further, split_markdown_by_header
from src.embedding.vearch_client import VearchClient
from src.embedding.text2vector import compute_emb, get_model
from src.llm.llm_factory import LLMFactory

logger = logging.getLogger('app')


def gen_case_by_req(
        fp: str | FileStorage,
        case_type: str,
        human_input: Optional[str] = None,
        multimodal: Optional[str] = 'none'
) -> BytesIO:
    """
    仅根据需求文档，生成测试用例

    Args:
        fp: (str | FileStorage) PDF文件路径或FileStorage对象
        case_type: (str) 用例类型，可选excel、XMind、Markdown
        human_input: (str) 用户键入的建议测试点
        multimodal: (str) 多模态类型

    Returns:
        BytesIO: 生成的测试用例
    """
    # 文档预处理: PDF转md
    analyze_images = multimodal == 'gpt4o'
    original_text = pdf2md(doc=fp, analyze_images=analyze_images)
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo-16k-0613")
    token_num = math.ceil(len(tokenizer.encode(original_text)))
    # 获取文件名
    if isinstance(fp, FileStorage):
        file_name = getattr(fp, 'filename', 'unknown')
    else:
        file_name = Path(fp).stem

    # 文档分块
    if token_num < 4096:
        chunks = split_markdown_further(documents=split_markdown_by_header(original_text), min_partition=1024, max_partition=4096)
    else:
        chunks = split_markdown_further(documents=split_markdown_by_header(original_text), min_partition=512, max_partition=4096)
    debug_info = f"chunks sum: {len(chunks)}\n"
    for i, chunk in enumerate(chunks):
        debug_info += f"Chunk {i + 1} - Content: {chunk.page_content[:100]}... - Metadata: {chunk.metadata}\n"
    logger.debug(debug_info)

    # 异步调用生成测试用例
    logger.info("start generating cases")
    output_list = asyncio.run(async_gen_cases(chunks, file_name, case_type, human_input))
    logger.info("finish await, start merging cases")

    # 对用例进行后处理，返回文件流
    if case_type == 'markdown':
        file_stream = BytesIO()
        output = '\n'.join(item for item in output_list if item is not None)
        file_stream.write(output.encode('utf-8'))
        file_stream.seek(0)
    else:
        output_json_list = [json.loads(item) for item in output_list if item is not None]
        if case_type == 'excel':
            all_rows = [row for json_obj in output_json_list for row in parse_json(json_obj)]
            file_stream = gen_excel(all_rows)
        elif case_type == 'xmind':
            file_stream = gen_xmind(merge_json(*output_json_list))
    return file_stream


def gen_case_by_recall(
        file: FileStorage,
        space_name: str,
        handwritten_text: str,
        case_type: str
) -> Union[str, BytesIO]:
    """
    将用户输入问题转为向量，检索向量库得到相关设计文档，并结合本次的需求文档生成测试用例

    Args:
        file: (FileStorage) PDF文件对象
        space_name: (str) 向量表空间名称
        handwritten_text: (str) 用户Query
        case_type: (str) 用例种类，可选excel、XMind、Markdown

    Returns:
        生成的测试用例
    """
    query = [handwritten_text]
    vearch = VearchClient()
    model = get_model(model='t2v_model')
    for sentence, embedding in compute_emb(model=model, sentences=query):
        # 检索到的所有文本
        response = vearch.search_by_vec(db_name='llm_test_db_1', space_name=space_name, vec=embedding, size=1)  # size=1表示只取1个匹配数据
    metadata = json.loads(response['hits']['hits'][0]['_source']['metadata'])
    sentence = response['hits']['hits'][0]['_source']['sentence']
    content = response['hits']['hits'][0]['_source']['content']
    qa_pair = [qa for qa in metadata['qa_list'] if sentence in qa[0] or sentence in qa[1]]
    qa_pair_str = '.这是基于技术文档片段生成的问答对,(问题为:' + qa_pair[0][0] + '回答为:' + qa_pair[0][1] + ')'
    return gen_case(prd_file=file, recall_str=content + qa_pair_str, case_type=case_type)


def gen_case(
        prd_file: FileStorage,
        recall_str: str,
        case_type: str,
        case_name: str = '',
        is_stream: Optional[bool] = True
) -> Union[str, BytesIO]:
    """
    根据需求文档、设计文档生成测试用例。当需求文档与设计文档重合度不高时,以需求文档为准

    Args:
        prd_file: (FileStorage) 需求文档PDF
        recall_str: (str) 检索到的设计文档
        case_type: (str) 用例类型，可选excel、XMind、Markdown
        is_stream: (Bool) 是否以文件流形式返回，默认为True
        case_name: （str） 测试用例名称

    Returns:
        生成的测试用例
    """
    if case_type == 'excel' or case_type == 'xmind':
        empty_case = FilePath.read_file(FilePath.empty_case)
    elif case_type == 'markdown':
        empty_case = FilePath.read_file(FilePath.empty_case_md)
    case_type_new = trans_case_type(case_type)

    # 将需求、设计相关文档设置给memory作为llm的记忆信息
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="你是一位经验丰富的系统测试专家，擅长根据产品需求文档和技术设计文档编写详细的测试用例。"
            ),  # The persistent system prompt
            MessagesPlaceholder(
                variable_name="chat_history"
            ),  # Where the memory will be stored.
            HumanMessagePromptTemplate.from_template(
                "{human_input}"
            ),  # Where the human input will injected
        ]
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prd_text = pdf2md(doc=prd_file, analyze_images=False)
    memory.save_context({"input": prd_text}, {"output": "这是一段产品需求文档，后续输出测试用例需要"})
    memory.save_context({"input": recall_str}, {"output": "这是一段技术设计文档，后续输出测试用例需要"})

    # 调大模型生成测试用例
    llm = LLMFactory.get_openai_factory().get_chat_llm()
    human_input = PromptConfig.GEN_CASE_BY_RECALL.format(case_type=case_type_new) + empty_case
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )
    output_raw = chain.invoke({'human_input': human_input})
    output = output_raw.get('text')
    output = re.search(r'{.*}', output, re.DOTALL).group()

    # 将测试用例内容转换为文件流
    if is_stream:
        if case_type == 'excel':
            data = json.loads(output)
            file_stream = gen_excel(parse_json(data))
        elif case_type == 'xmind':
            data = json.loads(output)
            file_stream = gen_xmind(data)
        elif case_type == 'markdown':
            file_stream = BytesIO()
            file_stream.write(output.encode('utf-8'))
            file_stream.seek(0)
        return file_stream
    else:
        # 保存输出的用例内容
        if not os.path.exists(FilePath.out_file):
            os.makedirs(FilePath.out_file)
        file_path = os.path.join(FilePath.out_file, case_name + ".json")
        counter = 1
        new_file_path = file_path
        while os.path.exists(new_file_path):
            new_file_path = os.path.join(FilePath.out_file, "{}({}).json".format(case_name, counter))
            counter += 1
        with open(new_file_path, 'w', encoding='utf-8') as file:
            file.write(output_raw.get('text'))
        return new_file_path


async def async_gen_cases(
        chunks: list[MdDocument],
        file_name: str,
        case_type: str,
        human_input: Optional[str] = None
):
    tasks = []
    for chunk in chunks:
        task = _gen_case(chunk, file_name, case_type, human_input)
        tasks.append(task)
    return await asyncio.gather(*tasks)


async def _gen_case(
        chunk: MdDocument,
        file_name: str,
        case_type: str,
        human_input: Optional[str] = None
) -> str | None:
    """
    根据切分后的文本块生成测试用例

    Args:
        chunk: (MdDocument) Markdown切分后的文本块
        file_name: (str) 用户上传的文件名
        case_type: (str) 用例类型，可选excel、xmind、markdown
        human_input: (str) 用户键入的建议测试点，可选None

    Returns:
        生成的测试用例：
        若case_type为excel/xmind，则返回Json类型的字符串；
        若case_type为markdown，则返回MD类型的字符串
    """
    # 生成用例类型
    if case_type == 'excel' or case_type == 'xmind':
        empty_case = FilePath.read_file(FilePath.empty_case)
    elif case_type == 'markdown':
        empty_case = FilePath.read_file(FilePath.empty_case_md)
    case_type_new = trans_case_type(case_type)

    # 调大模型生成测试用例
    llm = LLMFactory.get_openai_factory().get_gpt4o()
    llm.temperature = 0.0  # 将随机性调至最低
    llm.max_tokens = 4096
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # 第一轮对话prompt
    prompt1 = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=PromptConfig.REWRITE_DOC_SYS
            ),  # The persistent system prompt
            MessagesPlaceholder(
                variable_name="chat_history"
            ),  # Where the memory will be stored.
            HumanMessagePromptTemplate.from_template(
                "{input}"
            ),  # Where the human input will injected
        ]
    )
    # 第二轮对话prompt
    prompt2 = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=PromptConfig.GEN_CASE_BY_REQ_SYS
            ),  # The persistent system prompt
            MessagesPlaceholder(
                variable_name="chat_history"
            ),  # Where the memory will be stored.
            HumanMessagePromptTemplate.from_template(
                "{input}"
            ),  # Where the human input will injected
        ]
    )
    # 第一轮对话，先改写需求文档
    conversation = ConversationChain(
        llm=llm,
        prompt=prompt1,
        memory=memory,
        verbose=True
    )
    # 进行第一轮对话
    await conversation.apredict(input=PromptConfig.REWRITE_DOC.format(product=file_name, content=chunk.page_content))
    # 进行第二轮对话
    conversation.prompt = prompt2
    if human_input:
        human_input = '9. 当生成用例时，请确保参考建议测试点。若需求文档中不存在与建议测试点相关的片段，请忽略提供的测试点。\n' + '建议测试点：\n' + human_input + '\n'
        human_input = clean(human_input)
    input = PromptConfig.GEN_CASE_BY_REQ.format(case_type=case_type_new, empty_case=empty_case, human_input=human_input)
    output = await conversation.apredict(input=input)
    logger.info(f"output case:\n{output}")

    if case_type == 'markdown':
        output = clean_output_tags(output)
        output = clean_empty_lines(output)
    else:
        # 若输出结果的<output>不成对，则说明数据可能被截断
        # 尝试对JSON进行修复，若修复不成功直接丢弃
        if len(re.findall(r"<output>", output)) != len(re.findall(r"</output>", output)):
            output = re.search(r'{.*}', output, re.DOTALL).group()  # 使用正则表达式提取JSON字符串
            logger.error("Output content is truncated, try to repair...")
            output = fix_json_str(output)
        else:  # 数据正常，不需要修复
            output = re.search(r'{.*}', output, re.DOTALL).group()
    return output


def trans_case_type(case_type: str) -> str:
    if case_type == 'excel' or case_type == 'xmind':
        case_type = 'json'
    elif case_type == 'markdown':
        pass
    return case_type
