import os
import string
import pathlib

from common.file_path import FilePath
from src.doc_preprocess.cleaner import clean, optional_clean_output_tags
from src.doc_preprocess.image_analysis import clean_temp_images, analyze_md

try:
    import pymupdf as fitz  # available with v1.24.3
except ImportError:
    import fitz

from pymupdf4llm.helpers.get_text_lines import get_raw_lines, is_white
from pymupdf4llm.helpers.multi_column import column_boxes
from werkzeug.datastructures import FileStorage

if fitz.pymupdf_version_tuple < (1, 24, 2):
    raise NotImplementedError("PyMuPDF version 1.24.2 or later is needed.")

bullet = ("- ", "* ", chr(0xF0A7), chr(0xF0B7), chr(0xB7), chr(8226), chr(9679))
GRAPHICS_TEXT = "\n![%s](%s)\n"


class IdentifyHeaders:
    """识别文档标题"""

    def __init__(
            self,
            doc: str,
            pages: list = None,
            body_limit: float = 12,
    ):
        """读取所有文本，生成包含字体大小信息的字典

        Args:
            pages: 可选的页码列表
            body_limit: 将大于此字体大小的文本视为某种标题
        """
        if isinstance(doc, fitz.Document):
            mydoc = doc
        elif isinstance(doc, FileStorage):
            doc.seek(0)
            mydoc = fitz.open(stream=doc.read(), filetype="pdf")
        else:
            mydoc = fitz.open(doc)

        if pages is None:  # 如果未指定页码，则使用所有页
            pages = range(mydoc.page_count)

        fontsizes = {}
        for pno in pages:
            page = mydoc.load_page(pno)
            blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]
            for span in [  # 查看所有非空的水平跨度
                s
                for b in blocks
                for l in b["lines"]
                for s in l["spans"]
                if not is_white(s["text"])
            ]:
                fontsz = round(span["size"])
                count = fontsizes.get(fontsz, 0) + len(span["text"].strip())
                fontsizes[fontsz] = count

        if mydoc != doc:
            mydoc.close()

        # 将字体大小映射到多个#标题标签字符的字符串
        self.header_id = {}

        # 若未提供，选择出现次数最多的字体大小作为正文文本
        # 若所有页面都没有文本，则使用12
        # 无论如何，所有不超过body_limit的字体都将被视为正文文本
        temp = sorted(
            [(k, v) for k, v in fontsizes.items()],
            key=lambda i: i[1],
            reverse=True,
        )
        if temp:
            b_limit = max(body_limit, temp[0][0])
        else:
            b_limit = body_limit

        # 识别最多6种字体大小作为标题候选
        sizes = sorted(
            [f for f in fontsizes.keys() if f > b_limit],
            reverse=True,
        )[:6]

        # 标题标签字典
        for i, size in enumerate(sizes):
            self.header_id[size] = "#" * (i + 1) + " "

    def get_header_id(self, span: dict, page=None) -> str:
        """返回合适的Markdown标题前缀

        给定一个从"dict"/"rawdict"提取的文本跨度，确定
        Markdown标题前缀字符串，包含0到n个连接的'#'字符
        """
        fontsize = round(span["size"])  # 提取字体大小
        hdr_id = self.header_id.get(fontsize, "")
        return hdr_id


def to_markdown(
        doc: str | FileStorage,
        *,
        pages: list = None,
        hdr_info=None,
        write_images: bool = False,
        page_chunks: bool = False,
        margins=(0, 50, 0, 50),
) -> str:
    """处理文档，返回所选页面转换后的Markdown文本"""

    clean_temp_images()  # 清理本地临时图片
    if isinstance(doc, str):
        doc = fitz.open(doc)
        filename = os.path.basename(doc.name)
    elif isinstance(doc, FileStorage):
        filename = doc.filename
        doc = fitz.open(stream=doc.read(), filetype="pdf")

    if pages is None:  # 若未提供，则处理所有页面
        pages = list(range(doc.page_count))

    if hasattr(margins, "__float__"):
        margins = [margins] * 4
    if len(margins) == 2:
        margins = (0, margins[0], 0, margins[1])
    if len(margins) != 4:
        raise ValueError("margins must have length 2 or 4 or be a number.")
    elif not all([hasattr(m, "__float__") for m in margins]):
        raise ValueError("margin values must be numbers")

    # 如果"hdr_info"不是具有"get_header_id"方法的对象
    # 则扫描文档并使用字体大小作为标题级别指示器
    if callable(hdr_info):
        get_header_id = hdr_info
    elif hasattr(hdr_info, "get_header_id") and callable(hdr_info.get_header_id):
        get_header_id = hdr_info.get_header_id
    else:
        hdr_info = IdentifyHeaders(doc)
        get_header_id = hdr_info.get_header_id

    def resolve_links(links, span):
        """接受一个跨度，返回一个Markdown格式的链接字符串"""
        bbox = fitz.Rect(span["bbox"])  # span bbox
        # 链接应至少覆盖跨度的70%
        bbox_area = 0.7 * abs(bbox)
        for link in links:
            hot = link["from"]  # the hot area of the link
            if not abs(hot & bbox) >= bbox_area:
                continue  # 不触及边界框
            text = f'[{span["text"].strip()}]({link["uri"]})'
            return text

    def save_image(page, rect, i):
        """可以选择渲染页面的矩形区域并保存为图像"""
        # filename = page.parent.name.replace("\\", "/")
        # image_path = f"{filename}-{page.number}-{i}.png"
        # downloads_dir = pathlib.Path(__file__).parent.parent.parent / "downloads"  # Path
        downloads_dir = pathlib.Path(FilePath.images)  # 将图像保存到downloads文件夹
        downloads_dir.mkdir(exist_ok=True)
        image_path = downloads_dir / f"{filename}-{page.number}-{i}.png"
        if write_images is True:
            pix = page.get_pixmap(dpi=300, clip=rect)
            if pix.width < 100 or pix.height < 100:
                del pix
                return ""
            pix.save(image_path)
            del pix
            return image_path.name
            # return os.path.basename(image_path)
        return ""

    def write_text(
            page: fitz.Page,
            textpage: fitz.TextPage,
            clip: fitz.Rect,
            tabs=None,
            tab_rects: dict = None,
            img_rects: dict = None,
            links: list = None,
    ) -> string:
        """输出在给定clip内找到的文本

        能够识别标题、正文文本、代码块、内联代码、粗体、斜体和粗斜体样式。
        对有序/无序列表提供了一些支持，
        典型字符被替换为相应的Markdown字符。

        'tab_rects'/'img_rects'是表格和图像或矢量图形矩形的字典。
        一般Markdown文本生成会跳过这些区域。表格通过其自己的'to_markdown'方法写入。
        图像和矢量图形可选保存为文件，并转换为对应的Markdown格式的图片链接
        """
        if clip is None:
            clip = textpage.rect
        out_string = ""

        # This is a list of tuples (linerect, spanlist)
        nlines = get_raw_lines(textpage, clip=clip, tolerance=3)

        tab_rects0 = list(tab_rects.values())
        img_rects0 = list(img_rects.values())

        prev_lrect = None  # previous line rectangle
        prev_bno = -1  # previous block number of line
        code = False  # mode indicator: outputting code
        prev_hdr_string = None

        for lrect, spans in nlines:
            # there may tables or images inside the text block: skip them
            if intersects_rects(lrect, tab_rects0) or intersects_rects(
                    lrect, img_rects0
            ):
                continue

            # Pick up tables intersecting this text block
            for i, tab_rect in sorted(
                    [
                        j
                        for j in tab_rects.items()
                        if j[1].y1 <= lrect.y0 and not (j[1] & clip).is_empty
                    ],
                    key=lambda j: (j[1].y1, j[1].x0),
            ):
                out_string += "\n" + tabs[i].to_markdown(clean=False) + "\n"
                del tab_rects[i]

            # Pick up images / graphics intersecting this text block
            for i, img_rect in sorted(
                    [
                        j
                        for j in img_rects.items()
                        if j[1].y1 <= lrect.y0 and not (j[1] & clip).is_empty
                    ],
                    key=lambda j: (j[1].y1, j[1].x0),
            ):
                pathname = save_image(page, img_rect, i)
                if pathname:
                    out_string += GRAPHICS_TEXT % (pathname, pathname)
                del img_rects[i]

            text = " ".join([s["text"] for s in spans])

            # if the full line mono-spaced?
            all_mono = all([s["flags"] & 8 for s in spans])

            if all_mono:
                if not code:  # if not already in code output  mode:
                    out_string += "```\n"  # switch on "code" mode
                    code = True
                # compute approx. distance from left - assuming a width
                # of 0.5*fontsize.
                delta = int((lrect.x0 - clip.x0) / (spans[0]["size"] * 0.5))
                indent = " " * delta

                out_string += indent + text + "\n"
                continue  # done with this line

            span0 = spans[0]
            bno = span0["block"]  # block number of line
            if bno != prev_bno:
                out_string += "\n"
                prev_bno = bno

            if (  # check if we need another line break
                    prev_lrect
                    and lrect.y1 - prev_lrect.y1 > lrect.height * 1.5
                    or span0["text"].startswith("[")
                    or span0["text"].startswith(bullet)
                    or span0["flags"] & 1  # superscript?
            ):
                out_string += "\n"
            prev_lrect = lrect

            # if line is a header, this will return multiple "#" characters
            hdr_string = get_header_id(span0, page=page)

            # intercept if header text has been broken in multiple lines
            if hdr_string and hdr_string == prev_hdr_string:
                out_string = out_string[:-1] + " " + text + "\n"
                continue

            prev_hdr_string = hdr_string
            if hdr_string.startswith("#"):  # if a header line skip the rest
                out_string += hdr_string + text + "\n"
                continue

            # this line is not all-mono, so switch off "code" mode
            if code:  # still in code output mode?
                out_string += "```\n"  # switch of code mode
                code = False

            # Check if the entire line is bold
            all_bold = all([s["flags"] & 16 for s in spans])
            if all_bold:
                out_string += "###### " + text + "\n"
                continue

            for i, s in enumerate(spans):  # iterate spans of the line
                # decode font properties
                mono = s["flags"] & 8
                bold = s["flags"] & 16
                italic = s["flags"] & 2

                if mono:
                    # this is text in some monospaced font
                    out_string += f"`{s['text'].strip()}` "
                else:  # not a mono text
                    prefix = ""
                    suffix = ""
                    if hdr_string == "":
                        if bold:
                            prefix = "**"
                            suffix += "**"
                        if italic:
                            prefix += "_"
                            suffix = "_" + suffix

                    # convert intersecting link into markdown syntax
                    ltext = resolve_links(links, s)
                    if ltext:
                        text = f"{hdr_string}{prefix}{ltext}{suffix} "
                    else:
                        text = f"{hdr_string}{prefix}{s['text'].strip()}{suffix} "

                    if text.startswith(bullet):
                        text = "-  " + text[1:]
                    out_string += text
            if not code:
                out_string += "\n"
        out_string += "\n"
        if code:
            out_string += "```\n"  # switch of code mode
            code = False

        return (
            out_string.replace(" \n", "\n").replace("  ", " ").replace("\n\n\n", "\n\n")
        )

    def is_in_rects(rect, rect_list):
        """检查矩形是否包含在列表中的某个矩形内"""
        for i, r in enumerate(rect_list, start=1):
            if rect in r:
                return i
        return 0

    def intersects_rects(rect, rect_list):
        """检查矩形的中点是否包含在列表中的某个矩形内"""
        for i, r in enumerate(rect_list, start=1):
            if (rect.tl + rect.br) / 2 in r:  # middle point is inside r
                return i
        return 0

    def output_tables(tabs, text_rect, tab_rects):
        """输出位于文本矩形上方的表格"""
        this_md = ""  # markdown string for table content
        if text_rect is not None:  # select tables above the text block
            for i, trect in sorted(
                    [j for j in tab_rects.items() if j[1].y1 <= text_rect.y0],
                    key=lambda j: (j[1].y1, j[1].x0),
            ):
                this_md += tabs[i].to_markdown(clean=False)
                del tab_rects[i]  # do not touch this table twice

        else:  # output all remaining table
            for i, trect in sorted(
                    tab_rects.items(),
                    key=lambda j: (j[1].y1, j[1].x0),
            ):
                this_md += tabs[i].to_markdown(clean=False)
                del tab_rects[i]  # do not touch this table twice
        return this_md

    def output_images(page, text_rect, img_rects):
        """输出在文本矩形上的图像/矢量图"""
        if img_rects is None:
            return ""
        this_md = ""  # markdown string
        if text_rect is not None:  # select tables above the text block
            for i, img_rect in sorted(
                    [j for j in img_rects.items() if j[1].y1 <= text_rect.y0],
                    key=lambda j: (j[1].y1, j[1].x0),
            ):
                pathname = save_image(page, img_rect, i)
                if pathname:
                    this_md += GRAPHICS_TEXT % (pathname, pathname)
                del img_rects[i]  # do not touch this image twice

        else:  # output all remaining table
            for i, img_rect in sorted(
                    img_rects.items(),
                    key=lambda j: (j[1].y1, j[1].x0),
            ):
                pathname = save_image(page, img_rect, i)
                if pathname:
                    this_md += GRAPHICS_TEXT % (pathname, pathname)
                del img_rects[i]  # do not touch this image twice
        return this_md

    def get_metadata(doc, pno):
        meta = doc.metadata.copy()
        meta["file_path"] = doc.name
        meta["page_count"] = doc.page_count
        meta["page"] = pno + 1
        return meta

    def get_page_output(doc, pno, margins, textflags):
        """处理单页PDF

        Args:
            doc: fitz.Document
            pno: 0-based page number
            textflags: text extraction flag bits

        Returns:
            Markdown格式的字符串，包括文本内容、图像、标题、表格
        """
        page = doc[pno]
        md_string = ""
        left, top, right, bottom = margins
        clip = page.rect + (left, top, -right, -bottom)
        # extract all links on page
        links = [l for l in page.get_links() if l["kind"] == 2]

        # make a TextPage for all later extractions
        textpage = page.get_textpage(flags=textflags, clip=clip)

        img_info = [img for img in page.get_image_info() if img["bbox"] in clip]
        images = img_info[:]
        tables = []
        graphics = []

        # Locate all tables on page
        tabs = page.find_tables(clip=clip, strategy="lines_strict")

        # Make a list of table boundary boxes.
        # Must include the header bbox (may exist outside tab.bbox)
        tab_rects = {}
        for i, t in enumerate(tabs):
            tab_rects[i] = fitz.Rect(t.bbox) | fitz.Rect(t.header.bbox)
            tab_dict = {
                "bbox": tuple(tab_rects[i]),
                "rows": t.row_count,
                "columns": t.col_count,
            }
            tables.append(tab_dict)
        tab_rects0 = list(tab_rects.values())

        # Select paths that are not contained in any table
        page_clip = page.rect + (36, 36, -36, -36)  # ignore full page graphics
        paths = [
            p
            for p in page.get_drawings()
            if not intersects_rects(p["rect"], tab_rects0)
               and p["rect"] in page_clip
               and p["rect"].width < page_clip.width
               and p["rect"].height < page_clip.height
        ]

        # Determine vector graphics outside any tables, filerting out any
        # which contain no stroked paths
        vg_clusters = []
        for bbox in page.cluster_drawings(drawings=paths):
            include = False
            for p in [p for p in paths if p["rect"] in bbox]:
                if p["type"] != "f":
                    include = True
                    break
                if [item[0] for item in p["items"] if item[0] == "c"]:
                    include = True
                    break
                if include is True:
                    vg_clusters.append(bbox)

        actual_paths = [p for p in paths if is_in_rects(p["rect"], vg_clusters)]

        vg_clusters0 = [
            r
            for r in vg_clusters
            if not intersects_rects(r, tab_rects0) and r.height > 20
        ]

        if write_images is True:
            vg_clusters0 += [fitz.Rect(i["bbox"]) for i in img_info]

        vg_clusters = dict((i, r) for i, r in enumerate(vg_clusters0))

        # Determine text column bboxes on page, avoiding tables and graphics
        text_rects = column_boxes(
            page,
            paths=actual_paths,
            no_image_text=write_images,
            textpage=textpage,
            avoid=tab_rects0 + vg_clusters0,
        )
        """遍历文本矩形提取 Markdown 文本。
        输出任何表格（可能位于文本矩形的上方、下方或内部）
        """
        for text_rect in text_rects:
            # output tables above this block of text
            md_string += output_tables(tabs, text_rect, tab_rects)
            md_string += output_images(page, text_rect, vg_clusters)

            # output text inside this rectangle
            md_string += write_text(
                page,
                textpage,
                text_rect,
                tabs=tabs,
                tab_rects=tab_rects,
                img_rects=vg_clusters,
                links=links,
            )

        # write any remaining tables and images
        md_string += output_tables(tabs, None, tab_rects)
        md_string += output_images(None, tab_rects, None)
        md_string += "\n-----\n\n"
        while md_string.startswith("\n"):
            md_string = md_string[1:]
        return md_string, images, tables, graphics

    if page_chunks is False:
        document_output = ""
    else:
        document_output = []

    # read the Table of Contents
    toc = doc.get_toc()
    textflags = fitz.TEXT_DEHYPHENATE | fitz.TEXT_MEDIABOX_CLIP
    for pno in pages:

        page_output, images, tables, graphics = get_page_output(
            doc, pno, margins, textflags
        )
        if page_chunks is False:
            document_output += page_output
        else:
            # build subet of TOC for this page
            page_tocs = [t for t in toc if t[-1] == pno + 1]

            metadata = get_metadata(doc, pno)
            document_output.append(
                {
                    "metadata": metadata,
                    "toc_items": page_tocs,
                    "tables": tables,
                    "images": images,
                    "graphics": graphics,
                    "text": page_output,
                }
            )

    return document_output


def pdf2md(doc: str | FileStorage, analyze_images: bool) -> str:
    """
    pdf转md，可对图像进行分析
    """
    text = to_markdown(doc=doc, write_images=analyze_images)
    text = clean(text)
    if analyze_images:  # 图像分析
        processed_text = analyze_md(text)
        _clean = optional_clean_output_tags(analyze_images)(clean)
        return _clean(processed_text)
    else:
        return text


if __name__ == "__main__":
    """
    将PDF文档转换为Markdown

    命令行调用方式如下：
    python pdf2md.py input.pdf [-pages PAGES]

    "PAGES"参数是一个不包含空格的逗号分隔的页码字符串。
    每个项可以是单个页码或范围"m-n"。使用"N"表示文档的最后一页。
    例如："-pages 2-15,40,43-N"

    将生成一个名为"input.md"的Markdown文本文件

    Dependencies
    -------------
    PyMuPDF v1.24.2 or later
    """
    import sys
    import time

    t0 = time.perf_counter()

    # 开始转换
    md_string = to_markdown("D:\\download\\case\\体验热力地图方案--实时版.pdf", write_images=False)
    print(md_string)

    # 写入md文件
    pathlib.Path("output.md").write_bytes(md_string.encode())
    t1 = time.perf_counter()
    print(f"Markdown creation time for {round(t1 - t0, 2)} sec.")
