import asyncio
import re
import argparse
import json
import math
import tiktoken

from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm

from src.doc_preprocess.md_splitter import extract_header
from src.knowledge_injection.extract_keywords import Extract_Keywords
from src.llm.llm_factory import LLMFactory
from src.doc_preprocess.doc_splitter import read_and_clean_pdf_text, split_text3
from static.config.prompt_config import PromptConfig


def Generate_QA(prompt):
    prompt_template = PromptTemplate(
        input_variables=[],
        template=prompt,
    )
    llm = LLMFactory.get_openai_factory().get_gpt3()
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        verbose=True
    )
    content = chain.invoke(input={})['text']
    content = re.search(r"<output>(.*?)</output>", content, re.DOTALL).group(1)
    arr = content.split('Question:')[1:]
    qa_pair = [p.split('Answer:') for p in arr]
    return qa_pair


def Generate_QA_From_Docs(pages, prompt_template, product_name, out_format="json"):
    for page in tqdm(pages[:]):
        # print(page)
        # yield { "doc" : page.page_content }
        prompt = prompt_template.format(product=product_name, page=page.page_content)
        qa_list = Generate_QA(prompt)
        for q_c, a_c in qa_list:
            if out_format == "json":
                ret = page.metadata
                ret["Q"] = q_c.strip()
                ret["A"] = a_c.strip()
                yield ret
            elif out_format == "QA":
                yield "Question: " + q_c.strip() + "\nAnswer: " + a_c.strip() + "\n\n"


def Generate_QA_From_Paras(paras, prompt_template, out_format="QA"):
    for para in tqdm(paras):
        # 计算token
        tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo-16k-0613")
        token_num = math.ceil(len(tokenizer.encode(para)) / 40)
        # 提取关键词
        keywords = Extract_Keywords(para, math.ceil(token_num / 1.5))
        # 按照第一个空格分割字符串
        parts = para.split(maxsplit=1)
        prompt = prompt_template.format(product=parts[0], page=parts[1], keywords=keywords, nums=token_num)
        qa_list = Generate_QA(prompt)
        for q_c, a_c in qa_list:
            if out_format == "json":
                ret = {"Q": q_c.strip(), "A": a_c.strip()}
                yield ret
            elif out_format == "QA":
                yield "Question: " + q_c.strip() + "\nAnswer: " + a_c.strip() + "\n\n"


def Generate_QA_From_Para(para, prompt_template, out_format="QA"):
    # 计算token
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo-16k-0613")
    token_num = math.ceil(len(tokenizer.encode(para)) / 40)
    # 提取关键词
    keywords = Extract_Keywords(para, math.ceil(token_num / 2))
    # 按照第一个空格分割字符串
    parts = para.split(maxsplit=1)
    prompt = prompt_template.format(product=parts[0], page=parts[1], keywords=keywords, nums=token_num)
    qa_list = Generate_QA(prompt)
    return qa_list


def Extract_QA_list_From_QA(qa_text, out_format="QA"):
    if out_format == "json":
        # 解析每一行的JSON内容
        data = json.loads(line.strip())
        # 提取Q和A字段，并将它们作为一个字典添加到列表中
        qa_dict = {'Q': data['Q'], 'A': data['A']}
        qa_list.append(qa_dict)
    elif out_format == "QA":
        # 将文本内容按行分割
        lines = qa_text.strip().split('\n')
        # 遍历每一行，提取问题和答案
        for i in range(0, len(lines), 3):
            question = lines[i].split('Question: ')[1]
            answer = lines[i + 1].split('Answer: ')[1]
            qa_list.append({'Q': question, 'A': answer})
    return qa_list


async def _Generate_QA(chain):
    content = (await chain.ainvoke(input={})).get('text')
    # 检查<output>与</output>的数量是否相等
    output_tags = re.findall(r"<output>", content)
    end_output_tags = re.findall(r"</output>", content)
    if len(output_tags) != len(end_output_tags):
        # 去除所有的<output>与</output>
        content = re.sub(r"</?output>", "", content)
    else:
        # 如果数量相等，正常解析content
        if output_tags == 0 and end_output_tags == 0:
            pass
        content = re.search(r"<output>(.*?)</output>", content, re.DOTALL).group(1)
    # content = re.search(r"<output>(.*?)</output>", content, re.DOTALL).group(1)
    arr = content.split('Question:')[1:]
    qa_pair = [p.split('Answer:') for p in arr]
    return qa_pair


async def _Generate_QA_From_Paras(paras, prompt_template):
    """
    基于段落生成FAQ
    """
    tasks = []
    paras_list = []
    for para in paras:
        # 计算token
        tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo-16k-0613")
        token_num = math.ceil(len(tokenizer.encode(para.page_content)) / 40)
        # 提取关键词
        keywords = Extract_Keywords(para.page_content, math.ceil(token_num / 1.7))
        # product表示段落的标题
        prompt = prompt_template.format(product=extract_header(para.metadata), page=para.page_content, keywords=keywords, nums=token_num)
        prompt = PromptTemplate(
            input_variables=[],
            template=prompt,
        )
        llm = LLMFactory.get_openai_factory().get_gpt4o()
        llm.temperature = 0.4
        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=True,
        )
        task = _Generate_QA(chain)
        tasks.append(task)
        paras_list.append(para)
    qa_pairs = await asyncio.gather(*tasks)
    output = [{'qa_pair': qa_pair, 'para': para} for qa_pair, para in zip(qa_pairs, paras_list)]
    return output


if __name__ == '__main__':
    """
    解析pdf，转为faq.txt，格式为json/QA
    json格式如下
    {"source": "./人机合一.pdf", "page": 0, "Q": "人机合一产品文档中需要包含哪些内容？", "A": "人机合一产品文档需要包含转接说明增加主动询问的需求，以及转接说明备注内容进入应答流程的要求。"}

    QA格式如下
    Question: 人机合一产品文档中是否需要增加转接说明？
    Answer: 是的，人机合一产品文档中需要增加转接说明。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='D:\\download\\case\\2024.4.26-金采渠道支持资方收支分离结算.pdf', help='input file')
    parser.add_argument('--output_file', type=str, default='./无标题.faq', help='output file')
    parser.add_argument('--product', type=str, default="人机合一", help='specify the product name of doc')
    parser.add_argument('--input_format', type=str, default="pdf", help='specify the format')
    parser.add_argument('--lang', type=str, default="zh", help='specify the language')
    parser.add_argument('--output_format', type=str, default="QA", help='specify the language')
    args = parser.parse_args()
    doc_path = args.input_file
    product_name = args.product
    qa_path = args.output_file
    in_format = args.input_format
    lang = args.lang
    out_format = args.output_format

    prompt_template = PromptConfig.FAQ

    # docs = None
    # if in_format == "pdf":
    #     loader = PyPDFLoader(doc_path)
    #     docs = loader.load_and_split()
    # elif in_format == "md":
    #     in_file = open(doc_path, 'r')
    #     markdown_text = in_file.read()
    #     # markdown_splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=0)
    #     text_splitter = RecursiveCharacterTextSplitter(
    #         separators=["#", "\n\n", "\n"],
    #         chunk_size=1000,
    #         chunk_overlap=0
    #     )
    #     docs = text_splitter.create_documents([markdown_text])
    # else:
    #     raise RuntimeError

    text = asyncio.run(read_and_clean_pdf_text(file=doc_path, bold_detection=True, analyze_images=True))
    paras_list = split_text3(text)

    out_f = open(qa_path, 'w')
    with open(qa_path, 'w', encoding='utf-8') as out_f:
        # for result in Generate_QA_From_Docs(docs, prompt_template, product_name, out_format):
        for result in Generate_QA_From_Paras(paras_list, prompt_template, out_format):
            if out_format == "json":
                out_f.write(json.dumps(result, ensure_ascii=False))
                out_f.write("\n")
            elif out_format == "QA":
                out_f.write(result)

    """
    提取faq.txt的qa，转为列表，元素为一个字典
    格式如下
    [{'Q': '第一个问题？', 'A': '第一个回答。'}, {'Q': '第二个问题？', 'A': '第二个回答。'}]
    """
    qa_list = []
    with open(qa_path, 'r', encoding='utf-8') as file:
        if out_format == "json":
            for line in file:
                # 解析每一行的JSON内容
                data = json.loads(line.strip())
                # 提取Q和A字段，并将它们作为一个字典添加到列表中
                qa_dict = {'Q': data['Q'], 'A': data['A']}
                qa_list.append(qa_dict)
        elif out_format == "QA":
            data = file.read()
            # 将文本内容按行分割
            lines = data.strip().split('\n')
            # 遍历每一行，提取问题和答案
            for i in range(0, len(lines), 3):
                question = lines[i].split('Question: ')[1]
                answer = lines[i + 1].split('Answer: ')[1]
                qa_list.append({'Q': question, 'A': answer})

    print(qa_list)
