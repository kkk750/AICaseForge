import asyncio
import base64
import importlib.metadata
import logging
import os
import pathlib
import re

from PIL import Image

if importlib.metadata.version('pymupdf') >= '1.24.3':
    import pymupdf as fitz
else:
    import fitz

from langchain_core.messages import HumanMessage

from common.file_path import FilePath
from src.llm.llm_factory import LLMFactory
from static.config.prompt_config import PromptConfig

logger = logging.getLogger('app')


async def analyze_image(llm, prompt, base64_image):
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            }
        ]
    )
    output = await llm.ainvoke([message])
    logger.info("Image processing end with:\n" + output.content)
    return output.content


async def process_page_images(llm, doc, page, text_with_images: str = '', size_threshold=10240):
    """
    处理PDF某一页的所有图像，并使用大模型进行分析。
    Args:
        llm: (ChatOpenAI client) 大模型实例
        doc: (fitz.Document) PDF文档对象
        page: (fitz.Page) 当前页面对象
        text_with_images: (str) 当前页面的文本内容
        size_threshold: (int) 需要忽略图像的大小阈值
    Returns:
        图像理解结果
    """
    tasks = []
    image_list = page.get_images(full=True)
    for image_index, img in enumerate(image_list):
        base_image = doc.extract_image(img[0])
        image_bytes = base_image["image"]
        # 若图像大小小于阈值，则跳过
        pix = fitz.Pixmap(doc, img[0])
        if pix.size < size_threshold:
            continue
        img_base64 = base64.b64encode(image_bytes).decode('utf-8')
        content = text_with_images[:]  # TODO 是否需要截断？
        prompt = PromptConfig.IMG_ANALYSIS
        task = analyze_image(llm, prompt, img_base64)
        tasks.append(task)
    image_understandings = await asyncio.gather(*tasks)
    return "\n".join([f"\nImage Understanding: {text}\n" for text in image_understandings])


async def extract_text_and_images(file):
    """
    提取PDF文档中的文本和图像，并使用大模型进行分析。

    Args:
        file: (FileStorage) PDF文件对象

    Return:
        包含文本和图像分析结果的字符串
    """
    doc = fitz.open(stream=file.read(), filetype="pdf")
    output = ""
    llm = LLMFactory.get_openai_factory().get_gpt4o()
    tasks = []
    # 一页对应一个异步任务
    for page_num, page in enumerate(doc):
        text = page.get_text()
        output += text
        image_task = process_page_images(llm, doc, page, output)
        tasks.append(image_task)
    pages_understanding = await asyncio.gather(*tasks)
    for understanding in pages_understanding:
        output += understanding
    doc.close()
    return output


def analyze_pdf(file) -> str:
    """
    提取PDF中的所有图像，并逐个使用大模型进行理解。图片理解结果将添加到文本末尾

    Args:
        file: (FileStorage) PDF文件对象

    Returns:
        包含原始文本、图像理解结果的字符串
    """
    output = asyncio.run(extract_text_and_images(file))
    return output


def image_filter(
        image_path: str, size_threshold: tuple[int, int] = (100, 100), color_threshold: float = 0.9
) -> bool:
    """检查图片是否符合规定"""
    # 若图片不存在返回 False
    if not os.path.isfile(image_path):
        return False
    try:
        with Image.open(image_path) as img:
            # 检查尺寸
            if img.width < size_threshold[0] or img.height < size_threshold[1]:
                return False
            # 如果某种颜色占据了绝大部分像素，则认为是无意义的图像
            histogram = img.histogram()
            total_pixels = sum(histogram)
            if total_pixels == 0:
                return False
            if max(histogram) / total_pixels > color_threshold:
                return False
    except OSError:
        return False
    return True


def clean_temp_images():
    """清理临时存储的图片"""
    image_folder = pathlib.Path(FilePath.images)
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.svg"]
    for ext in extensions:
        files = image_folder.glob(ext)
        for file in files:
            if os.path.isfile(file):
                os.remove(file)


async def convert_image_links(md_str: str) -> str:
    """调大模型对md中的图片进行理解，并替换对应的图片链接"""

    # 匹配所有图像链接
    image_pattern = re.compile(r"!\[.*?\]\((.*?)\)")
    image_links = image_pattern.findall(md_str)
    if not image_links:
        return md_str
    image_analysis_results = []
    image_folder = FilePath.images
    # 遍历所有图像链接
    for image_link in image_links:
        if image_link.lower().endswith((".png", ".jpg", ".jpeg", ".svg")):
            image_path = os.path.join(image_folder, image_link)

            # 检查图像是否满足分析条件
            if image_filter(image_path):
                logger.info("Image processing start: " + image_link)
                # 读取图像文件并转换为base64
                with open(image_path, "rb") as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
                llm = LLMFactory.get_openai_factory().get_gpt4o()
                prompt = PromptConfig.IMG_ANALYSIS

                task = analyze_image(llm, prompt, img_base64)
                image_analysis_results.append((image_link, task))  # 存储任务和对应的图像链接

    # 等待所有异步任务完成
    completed_tasks = await asyncio.gather(
        *[task for _, task in image_analysis_results]
    )

    # 替换Markdown中的图像链接
    output_md = md_str
    for result, (image_link, _) in zip(completed_tasks, image_analysis_results):
        new_content = result
        output_md = re.sub(
            rf"!\[.*?\]\({re.escape(image_link)}\)", new_content, output_md, count=1
        )

    # 清空临时文件
    clean_temp_images()

    return output_md


def analyze_md(md_str: str) -> str:
    """
    对md中的图像进行理解分析，md中的图片链接将被替换为分析后的结果

    Args:
        md_str: (str) md类型的字符串

    Returns:
        包含图像整理结果的md字符串
    """
    logger.info("Start analyzing images in Markdown")
    output = asyncio.run(convert_image_links(md_str))
    logger.info("Finish analyzing images in Markdown")
    return output


if __name__ == '__main__':
    clean_temp_images()
