from typing import Optional, List
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_core.documents import Document
from unstructured.partition.html import partition_html


class MdDocument:
    """Markdown文本块"""

    def __init__(self, page_content, metadata):
        self.page_content = page_content  # 文本内容
        self.metadata = metadata  # 标题信息


def split_markdown_by_header(md_text: str) -> List[Document]:
    """
    基于标题，对Markdown进行粗切分
    Args:
        md_text: (str) Markdown格式的字符串
    Returns:
        List[LcDocument]: 切分后的文档集
    """
    # 根据不同的Markdown风格，可指定`#`、`##` 等
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
        ("######", "Header 6"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False, return_each_line=False
    )
    return markdown_splitter.split_text(md_text)


def split_markdown_further(
        documents,
        max_partition: Optional[int] = 1024,
        min_partition: Optional[int] = 0,
        overlap: Optional[int] = 128
) -> List[MdDocument]:
    """
    基于粗切分后的Markdown，做进行进一步切分聚合，确保每个片段长度均符合要求
    Args:
        documents: (List[Document]) 经初步切分后的Markdown文本块
        max_partition: (int) 文本块最大值，默认为1024
        min_partition: (int) 文本块最小值, 默认为0
        overlap   : (int) 文本块间，重叠的字符数, 默认为128
    Returns:
        List[MdDocument]: 切分聚合后的文档块列表
    """
    # 初步解析文档内容
    parsed_docs = []
    for doc in documents:
        elements = partition_html(text=doc.page_content, source_format="md")
        for element in elements:
            parsed_docs.append(MdDocument(element.text, doc.metadata))

    # 进行递归字符切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_partition,
        chunk_overlap=overlap,
        length_function=len,
    )

    merged_documents = []
    for doc in parsed_docs:
        splits = text_splitter.split_text(doc.page_content)

        # 合并小于最小分区的片段
        temp_content = ""
        temp_metadata = doc.metadata
        for split in splits:
            if len(temp_content) + len(split) <= max_partition:
                temp_content += '\n' + split
            else:
                if len(temp_content) >= min_partition:
                    merged_documents.append(MdDocument(temp_content, temp_metadata))
                    temp_content = split
                else:
                    if len(split) >= min_partition:
                        merged_documents.append(MdDocument(split, temp_metadata))
                    else:
                        if temp_content:
                            temp_content += '\n' + split
                        else:
                            temp_content = '\n' + split

        if len(temp_content) >= min_partition:
            merged_documents.append(MdDocument(temp_content, temp_metadata))
        elif (
                merged_documents
                and len(merged_documents[-1].page_content) + len(temp_content)
                <= max_partition
        ):
            merged_documents[-1].page_content += temp_content
        elif temp_content:
            merged_documents.append(MdDocument(temp_content, temp_metadata))

    return merged_documents


def extract_header(metadata: dict) -> str:
    """提取文本块的标题信息，并合并为单一文本字符串

    For example:
    {'Header 1': '一站式解决方案工作台PRD', 'Header 2': '背景：'}

    Gets converted to:
    一站式解决方案工作台PRD-背景
    """
    result = '-'.join(metadata.values())
    return result


if __name__ == '__main__':
    """
    Markdown分割器，提供标题粗切分（可以基于需要自行更改标题分隔符）和深度切分
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--md_text', type=str, default="D:\\download\\case\\test.md", help='传递需要切分的markdown文本或文件路径')
    parser.add_argument('--is_deep_split', type=bool, default=True, help='是否进行深度切分')
    args = parser.parse_args()
    md_text = args.md_text
    is_deep_split = args.is_deep_split

    if md_text.endswith(".md"):
        with open(md_text, "r", encoding='utf-8') as rf:
            md_text = rf.read()
    else:
        md_text = md_text

    md_list = split_markdown_by_header(md_text)
    if is_deep_split:
        md_list = split_markdown_further(split_markdown_by_header(md_text))

    for md_doc in md_list:
        if is_deep_split:
            print(md_doc.page_content)
            print(md_doc.metadata)
        else:
            print(md_doc)
        print('-------end-------')
