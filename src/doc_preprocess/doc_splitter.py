import asyncio
import re
from pathlib import Path
import logging

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from werkzeug.datastructures import FileStorage

from common import file_path
from src.doc_preprocess.image_analysis import process_page_images
from src.llm.llm_factory import LLMFactory
from src.llm.token_limit import TokenLimit


def load_fitz_split():
    logger = logging.getLogger()
    # 初始化文本分割器
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        model_name=TokenLimit.openai_token_limit.get("name"),
        chunk_size=TokenLimit.openai_token_limit.get("limit"),
        chunk_overlap=128
    )
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=20,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )
    # 加载PDF文件
    try:
        loader = PyMuPDFLoader(file_path)
        data = loader.load_and_split(text_splitter)
        page_content_list = [document.page_content for document in data]
        for i in page_content_list:
            text = r_splitter.split_text(i)
        # for i in data:
        #     print(i,type(i))
        #     print('\n')
        # return list(set(data))
        # print(data)
        return data
    except Exception as e:
        # 处理异常，如文件不存在或路径错误
        logger.error(f"加载PDF时出错: {e}")
        return None


def split_text4(text, chunk_size, chunk_overlap=250):
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", " ", ""]
    )
    r_splitter_text = r_splitter.split_text(text)
    return r_splitter_text


def split_text3(paras_str: str, min_chunk_size=40, max_chunk_size=1000) -> list[str]:
    """
    对以段落切分的字符串进行后处理，返回按段落分割的列表，清理合并文本量过小的段落，切分文本量大的段落

    Args:
        paras_str: (str) 已经使用read_and_clean_pdf_text清洗后的字符串，段落之间由`\\n\\n`分割

    Returns:
        切分后的段落列表
    """
    paras_list = paras_str.strip().split('\n\n')
    paras_list_new = []
    temp = ''
    for para in paras_list:
        if para == paras_list[-1]:
            temp += para
            paras_list_new.append(temp)

        elif len(para) > min_chunk_size:
            if temp == '':
                paras_list_new.append(para)
            else:
                temp += para
                paras_list_new.append(temp)
                temp = ''
        else:
            temp += (para + ' ')
            continue
    # 匹配开头的【，紧跟着的数字（如果有的话），直到】，然后替换为去掉数字和【】的部分
    # 匹配开头的"#"和随后的所有空格，然后替换为空字符串
    # 将{}替换为【】
    paras_list_new = [
        re.sub(r'^【\d*(.*?)】', r'\1', re.sub(r'^#+\s*', '', para.replace("{", "【").replace("}", "】")))
        for para in paras_list_new
    ]
    paras_list = paras_list_new
    paras_list_new = []
    for para in paras_list:
        if len(para) > max_chunk_size:
            paras_list_new.extend(split_text4(para, max_chunk_size))
        else:
            paras_list_new.append(para)
    return paras_list_new


def split_text2(text, sentence_size=40, relevant_doc=20):
    # 替换URLs
    url_pattern = re.compile(r'https?://[^\s]+')
    urls = url_pattern.findall(text)
    for i, url in enumerate(urls):
        text = text.replace(url, f'URL_PLACEHOLDER_{i}')

    # 文本清洗和分句
    text = re.sub(r"\n{3,}", r"\n", text)
    text = re.sub('\s', " ", text)
    text = re.sub("\n\n", "", text)
    text = re.sub(r'([;；.!?。！？\?])([^”’])', r"\1\n\2", text)
    text = re.sub(r'(\.{6}|\…{2})([^"’”」』])', r"\1\n\2", text)
    text = re.sub(r'([;；!?。！？\?]["’”」』]{0,2})([^;；!?，。！？\?])', r'\1\n\2', text)
    text = re.sub(r'[ ·【】]', '', text)
    text = re.sub(r"-{3,}", '。', text)
    text = text.rstrip()

    # 分割句子并进一步分割过长的句子
    ls = []
    for sentence in text.split("\n"):
        if sentence:
            while len(sentence) > sentence_size:
                split_pos = max(sentence.rfind(';', 0, sentence_size),
                                sentence.rfind('，', 0, sentence_size),
                                sentence.rfind(',', 0, sentence_size),
                                sentence.rfind('。', 0, sentence_size),
                                sentence.rfind('.', 0, sentence_size))
                if split_pos == -1:  # 如果没有找到空格，强制在sentence_size处分割
                    split_pos = sentence_size - 1
                ls.append(sentence[:split_pos + 1])
                sentence = sentence[split_pos + 1:].lstrip()
            ls.append(sentence)

    # 恢复URLs
    for i, url in enumerate(urls):
        ls = [sentence.replace(f'URL_PLACEHOLDER_{i}', url) for sentence in ls]

    ls_new = [''.join(ls[max(i - relevant_doc, 0):min(i + relevant_doc, len(ls))]) for i in range(len(ls))]

    return ls, ls_new


def split_text1(text):
    text = re.sub(r"\n{3,}", "\n", text)
    text = re.sub('\s', ' ', text)
    text = text.replace("\n\n", "")
    sent_sep_pattern = re.compile('([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')  # del ：；
    sent_list = []
    for ele in sent_sep_pattern.split(text):
        if sent_sep_pattern.match(ele) and sent_list:
            sent_list[-1] += ele
        elif ele:
            sent_list.append(ele)
    return sent_list


def split_text(text, sentence_size=100):  ##此处需要进一步优化逻辑
    url_pattern = re.compile(r'http?://[^\s]+')
    urls = url_pattern.findall(text)
    for i, url in enumerate(urls):
        text = text.replace(url, f'URL_PLACEHOLDER_{i}')

    # 文本清洗和分句
    text = re.sub(r"\n{3,}", r"\n", text)
    text = re.sub('\s', " ", text)
    text = re.sub("\n\n", "", text)

    text = re.sub(r'([;；.!?。！？\?])([^”’])', r"\1\n\2", text)  # 单字符断句符
    text = re.sub(r'(\.{6})([^"’”」』])', r"\1\n\2", text)  # 英文省略号
    text = re.sub(r'(\…{2})([^"’”」』])', r"\1\n\2", text)  # 中文省略号
    text = re.sub(r'([;；!?。！？\?]["’”」』]{0,2})([^;；!?，。！？\?])', r'\1\n\2', text)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    text = text.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    ls = [i for i in text.split("\n") if i]
    for ele in ls:
        if len(ele) > sentence_size:
            ele1 = re.sub(r'([,，.]["’”」』]{0,2})([^,，.])', r'\1\n\2', ele)
            ele1_ls = ele1.split("\n")
            for ele_ele1 in ele1_ls:
                if len(ele_ele1) > sentence_size:
                    ele_ele2 = re.sub(r'([\n]{1,}| {2,}["’”」』]{0,2})([^\s])', r'\1\n\2', ele_ele1)
                    ele2_ls = ele_ele2.split("\n")
                    for ele_ele2 in ele2_ls:
                        if len(ele_ele2) > sentence_size:
                            ele_ele3 = re.sub('( ["’”」』]{0,2})([^ ])', r'\1\n\2', ele_ele2)
                            ele2_id = ele2_ls.index(ele_ele2)
                            ele2_ls = ele2_ls[:ele2_id] + [i for i in ele_ele3.split("\n") if i] + ele2_ls[
                                                                                                   ele2_id + 1:]
                    ele_id = ele1_ls.index(ele_ele1)
                    ele1_ls = ele1_ls[:ele_id] + [i for i in ele2_ls if i] + ele1_ls[ele_id + 1:]

            id = ls.index(ele)
            ls = ls[:id] + [i for i in ele1_ls if i] + ls[id + 1:]
    for i, url in enumerate(urls):
        text = text.replace(f'URL_PLACEHOLDER_{i}', url)
    return ls


async def read_and_clean_pdf_text(file, bold_detection=False, analyze_images=False):
    """
    这个函数用于分割pdf，用了很多trick，逻辑较乱，效果奇好
    **函数功能**
    读取pdf文件并清理其中的文本内容，清理规则包括：
    - 提取所有块元的文本信息，并合并为一个字符串
    - 去除短块（字符数小于100）并替换为回车符
    - 清理多余的空行
    - 合并小写字母开头的段落块并替换为空格
    - 清除重复的换行
    - 将每个换行符替换为两个换行符，使每个段落之间有两个换行符分隔

    Args:
        file: PDF文件路径，可以是本地路径或在线资源
        bold_detection: 是否进行粗体检测，如果为 True，将识别文档中的粗体文本并据此识别段落。
        analyze_images: 是否分析文档中的图片。如果为 True，将使用大模型分析图片，并插入到文本中。

    Returns:
        meta_txt：清理后的文本内容字符串
    """
    import copy
    import re
    import numpy as np
    import importlib.metadata
    if importlib.metadata.version('pymupdf') >= '1.24.3':
        import pymupdf as fitz
    else:
        import fitz
    fc = 0  # Index 0 文本
    fs = 1  # Index 1 字体
    fb = 2  # Index 2 框框
    REMOVE_FOOT_NOTE = True  # 是否丢弃掉 不是正文的内容 （比正文字体小，如参考文献、脚注、图注等）
    REMOVE_FOOT_FFSIZE_PERCENT = 0.95  # 小于正文的？时，判定为不是正文（有些文章的正文部分字体大小不是100%统一的，有肉眼不可见的小变化）

    if isinstance(file, FileStorage):
        doc = fitz.open(stream=file.read(), filetype="pdf")
    else:
        doc = fitz.open(file)

    def primary_ffsize(l):
        """
        提取文本块主字体
        """
        fsize_statiscs = {}
        for wtf in l['spans']:
            if wtf['size'] not in fsize_statiscs: fsize_statiscs[wtf['size']] = 0
            fsize_statiscs[wtf['size']] += len(wtf['text'])
        return max(fsize_statiscs, key=fsize_statiscs.get)

    def ffsize_same(a, b):
        """
        提取字体大小是否近似相等
        """
        return abs((a - b) / max(a, b)) < 0.02

    with doc:
        meta_txt = []
        meta_font = []

        meta_line = []
        meta_span = []
        tasks = []
        ############################## <第 1 步，搜集初始信息> ##################################
        for index, page in enumerate(doc):
            # file_content += page.get_text()
            text_areas = page.get_text("dict")  # 获取页面上的文本信息
            for t in text_areas['blocks']:
                if 'lines' in t:
                    pf = 998
                    for l in t['lines']:
                        txt_line = "".join([wtf['text'] for wtf in l['spans']])
                        if len(txt_line) == 0: continue
                        pf = primary_ffsize(l)
                        meta_line.append([txt_line, pf, l['bbox'], l])
                        for wtf in l['spans']:  # for l in t['lines']:
                            meta_span.append([wtf['text'], wtf['size'], len(wtf['text'])])
                    # meta_line.append(["NEW_BLOCK", pf])
            # 块元提取                           for each word segment with in line                       for each line         cross-line words                          for each block
            meta_txt.extend([" ".join(["".join([wtf['text'] for wtf in l['spans']]) for l in t['lines']]).replace(
                '- ', '') for t in text_areas['blocks'] if 'lines' in t])
            meta_font.extend([np.mean([np.mean([wtf['size'] for wtf in l['spans']])
                                       for l in t['lines']]) for t in text_areas['blocks'] if 'lines' in t])
            if index == 0:
                page_one_meta = [" ".join(["".join([wtf['text'] for wtf in l['spans']]) for l in t['lines']]).replace(
                    '- ', '') for t in text_areas['blocks'] if 'lines' in t]

            if analyze_images:
                llm = LLMFactory.get_openai_factory().get_gpt4o()
                image_task = process_page_images(llm, doc, page)
                tasks.append(image_task)

        if analyze_images:
            def extract_paragraphs(text):
                """
                提取 <output></output> 包裹的字符串，并将其识别为段落
                """
                paragraphs = re.findall(r'<output>(.*?)</output>', text, re.DOTALL)
                cleaned_paragraphs = [para.replace('\n', ' ').replace('\n\n', ' ').strip() for para in paragraphs]
                return cleaned_paragraphs

            images = await asyncio.gather(*tasks)
            _images = []
            for image in images:
                image_paras = extract_paragraphs(image)
                for para in image_paras:
                    _images.append(para)

        ############################## <第 2 步，获取正文主字体> ##################################
        try:
            fsize_statiscs = {}
            for span in meta_span:
                if span[1] not in fsize_statiscs: fsize_statiscs[span[1]] = 0
                fsize_statiscs[span[1]] += span[2]
            main_fsize = max(fsize_statiscs, key=fsize_statiscs.get)
            if REMOVE_FOOT_NOTE:
                give_up_fize_threshold = main_fsize * REMOVE_FOOT_FFSIZE_PERCENT
        except:
            raise RuntimeError(f'抱歉, 暂时无法解析此PDF文档: {file}。')
        ############################## <第 3 步，切分和重新整合> ##################################
        mega_sec = []
        sec = []
        for index, line in enumerate(meta_line):
            if index == 0:
                sec.append(line[fc])
                continue
            if REMOVE_FOOT_NOTE:
                if meta_line[index][fs] <= give_up_fize_threshold:
                    continue
            if bold_detection:
                # 新增的粗体文本检测逻辑
                is_bold = any("Bold" in span['font'] for span in line[3]['spans'])
                if is_bold:
                    # 如果当前行是粗体，且不是新块的开始，则开始一个新的段落
                    if sec and sec[-1] != 'NEW_BLOCK':
                        mega_sec.append(copy.deepcopy(sec))
                        sec = []
                    # sec[-1] += line[fc]
                    sec.append(line[fc])
                    # sec[-1] += "\n"
                    continue  # 继续下一次循环
            if ffsize_same(meta_line[index][fs], meta_line[index - 1][fs]):
                # 尝试识别段落
                if meta_line[index][fc].endswith('.') and \
                        (meta_line[index - 1][fc] != 'NEW_BLOCK') and \
                        (meta_line[index][fb][2] - meta_line[index][fb][0]) < (
                        meta_line[index - 1][fb][2] - meta_line[index - 1][fb][0]) * 0.7:
                    sec[-1] += line[fc]
                    sec[-1] += "\n\n"
                else:
                    sec[-1] += " "
                    sec[-1] += line[fc]
            else:
                if (index + 1 < len(meta_line)) and \
                        meta_line[index][fs] > main_fsize:
                    # 单行 + 字体大
                    mega_sec.append(copy.deepcopy(sec))
                    sec = []
                    sec.append("# " + line[fc])
                else:
                    # 尝试识别section
                    if meta_line[index - 1][fs] > meta_line[index][fs]:
                        sec.append("\n" + line[fc])
                    else:
                        sec.append(line[fc])
        mega_sec.append(copy.deepcopy(sec))

        finals = []
        for ms in mega_sec:
            final = " ".join(ms)
            final = final.replace('- ', ' ')
            finals.append(final)
        meta_txt = finals

        ############################## <第 4 步，乱七八糟的后处理> ##################################
        def 把字符太少的块清除为回车(meta_txt):
            for index, block_txt in enumerate(meta_txt):
                if len(block_txt) < 20:
                    meta_txt[index] = '\n'
            return meta_txt

        meta_txt = 把字符太少的块清除为回车(meta_txt)

        def 清理多余的空行(meta_txt):
            for index in reversed(range(1, len(meta_txt))):
                if meta_txt[index] == '\n' and meta_txt[index - 1] == '\n':
                    meta_txt.pop(index)
            return meta_txt

        meta_txt = 清理多余的空行(meta_txt)

        def 合并小写开头的段落块(meta_txt):
            def starts_with_lowercase_word(s):
                pattern = r"^[a-z]+"
                match = re.match(pattern, s)
                if match:
                    return True
                else:
                    return False

            # 对于某些PDF会有第一个段落就以小写字母开头,为了避免索引错误将其更改为大写
            if starts_with_lowercase_word(meta_txt[0]):
                meta_txt[0] = meta_txt[0].capitalize()
            for _ in range(100):
                for index, block_txt in enumerate(meta_txt):
                    if starts_with_lowercase_word(block_txt):
                        if meta_txt[index - 1] != '\n':
                            meta_txt[index - 1] += ' '
                        else:
                            meta_txt[index - 1] = ''
                        meta_txt[index - 1] += meta_txt[index]
                        meta_txt[index] = '\n'
            return meta_txt

        meta_txt = 合并小写开头的段落块(meta_txt)
        meta_txt = 清理多余的空行(meta_txt)

        meta_txt = '\n'.join(meta_txt)
        # 清除重复的换行
        for _ in range(5):
            meta_txt = meta_txt.replace('\n\n', '\n')

        # 换行 -> 双换行
        meta_txt = meta_txt.replace('\n', '\n\n')

        ############################## <第 5 步，展示分割效果> ##################################
        # for f in finals:
        #    print亮黄(f)
        #    print亮绿('***************************')
        meta_txt = re.sub(r"\n[ ]*\n[ ]*\n*", r"\n\n", meta_txt)

        if analyze_images:
            for image_para in _images:
                meta_txt = meta_txt + '\n\n' + image_para
    return meta_txt
