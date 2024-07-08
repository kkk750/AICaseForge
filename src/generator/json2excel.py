import json
import re
from io import BytesIO

from openpyxl.workbook import Workbook


def parse_json(data, path=[], count=[0]):
    rows = []
    for key, value in data.items():
        new_path = path + [key]
        if isinstance(value, dict):
            if key.startswith("用例标题"):
                count[0] += 1
                steps = "\n".join([f"{i + 1}.{step[f'步骤{i + 1}']}" for i, step in enumerate(value['步骤'])])
                expected_results = "\n".join([f"{i + 1}.{step[f'预期结果']}" for i, step in enumerate(value['步骤'])])
                row = {
                    "一级分组": new_path[-4].split(": ")[1] if len(new_path) > 3 and ":" in new_path[-4] else (
                        new_path[-4] if len(new_path) > 3 else ""),
                    "二级分组": new_path[-3].split(": ")[1] if len(new_path) > 2 and ":" in new_path[-3] else (
                        new_path[-3] if len(new_path) > 2 else ""),
                    "三级分组": new_path[-2].split(": ")[1] if len(new_path) > 1 and ":" in new_path[-2] else (
                        new_path[-2] if len(new_path) > 1 else ""),
                    "四级分组": "",
                    "五级分组": "",
                    "六级分组": "",
                    "用例编号": count[0],
                    "用例标题": key.split(": ")[1],  # 对`用例标题: xxx` 取`xxx`
                    "用例类型": "功能测试",
                    "执行方式": "手动",
                    "前置条件": "xxx",
                    "步骤": steps,
                    "预期结果": expected_results,
                    "优先级": "P2",
                    "描述": "xxx"
                }
                rows.append(row)
            else:
                rows.extend(parse_json(value, new_path))
    return rows


def gen_excel(data):
    file_stream = BytesIO()
    wb = Workbook()
    ws = wb.active
    ws.append(
        ['一级分组', '二级分组', '三级分组', '四级分组', '五级分组', '六级分组', '用例编号', '用例标题', '用例类型', '执行方式', '前置条件', '步骤',
         '预期结果', '优先级', '描述'])
    for case in data:
        ws.append([
            case.get('一级分组', ''),
            case.get('二级分组', ''),
            case.get('三级分组', ''),
            case.get('四级分组', ''),
            case.get('五级分组', ''),
            case.get('六级分组', ''),
            case.get('用例编号', ''),
            case.get('用例标题', ''),
            case.get('用例类型', ''),
            case.get('执行方式', ''),
            case.get('前置条件', ''),
            case.get('步骤', ''),
            case.get('预期结果', ''),
            case.get('优先级', ''),
            case.get('描述', ''),
        ])
    wb.save(file_stream)
    file_stream.seek(0)
    # 返回文件流
    return file_stream


if __name__ == '__main__':
    """
    将String形式的json转换为Excel
    """
    json_str = """
{
    "私信讲解": {
        "富文本功能": {
            "格式展示": {
                "用例标题: 富文本格式正确展示": {
                    "步骤": [
                        {
                            "步骤1": "在私信编辑器中输入包含加粗、下划线、倾斜、颜色、背景色、大小、超链接、a标签、按钮、表格、图片的富文本内容。",
                            "预期结果": "私信预览中正确展示所有富文本格式，包括字体加粗、下划线、倾斜、颜色、背景色、大小（12px-48px）、超链接、a标签、插入按钮、表格、图片。"
                        }
                    ]
                }
            },
            "复制功能": {
                "用例标题: 富文本内容复制": {
                    "步骤": [
                        {
                            "步骤1": "在私信预览中选择富文本内容并执行复制操作。",
                            "预期结果": "富文本内容能够被成功复制，包括所有格式和样式。"
                        }
                    ]
                }
            }
        }
    },
    "弹窗讲解": {
        "配置项验证": {
            "必填项": {
                "用例标题: 验证所有必填配置项": {
                    "步骤": [
                        {
                            "步骤1": "在弹窗讲解配置界面，尝试保存一个未填写横幅、图标、主标题、按钮文字、按钮跳转内容的配置。",
                            "预期结果": "系统提示必填项未完成，保存操作失败。"
                        }
                    ]
                }
            },
            "非必填项": {
                "用例标题: 验证非必填配置项": {
                    "步骤": [
                        {
                            "步骤1": "在弹窗讲解配置界面，仅填写必填项，忽略副标题和标签。",
                            "预期结果": "系统允许保存配置，非必填项可以为空。"
                        }
                    ]
                }
            }
        }
    },
    "贴图讲解": {
        "图片上传功能": {
            "大小及格式限制": {
                "用例标题: 验证图片大小及格式限制": {
                    "步骤": [
                        {
                            "步骤1": "尝试上传一张大于200KB的图片。",
                            "预期结果": "系统提示图片大小超出限制，上传失败。"
                        },
                        {
                            "步骤2": "尝试上传一张小于200KB但格式不是jpg、jpeg、png的图片。",
                            "预期结果": "系统提示图片格式不支持，上传失败。"
                        },
                        {
                            "步骤3": "尝试上传一张小于200KB且格式为jpg、jpeg、png的图片。",
                            "预期结果": "图片成功上传。"
                        }
                    ]
                }
            }
        }
    }
}
    """
    data = json.loads(json_str)
    file_stream = gen_excel(parse_json(data))
    # 用文件流生成excel
    with open('output.xlsx', 'wb') as new_file:
        new_file.write(file_stream.getvalue())
