import glob
import logging
import json
import os
import re
from io import BytesIO

import xmind

logger = logging.getLogger('app')


def preprocess(json_obj):
    """
    预处理json，主要做如下处理：
    更改"步骤"对应层级，示例：
    "步骤": [
        {
            "步骤1": "步骤说明",
            "预期结果": "结果说明"
        }
    ]
    更改为：
    "步骤": [
        {
            "步骤说明": "结果说明",
        }
    ]
    """
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            if key == "步骤":
                new_steps = []
                for step in value:
                    if isinstance(step, dict) and len(step) == 2:
                        new_step = {list(step.values())[0]: list(step.values())[1]}
                        new_steps.append(new_step)
                    else:
                        new_steps.append(step)
                json_obj[key] = new_steps
            else:
                preprocess(value)
    elif isinstance(json_obj, list):
        for item in json_obj:
            preprocess(item)


def create_xmind_from_json(json_obj, parent_topic, add_plus=True):
    """
    递归地将JSON对象转换为XMind主题。
    在主题前添加"+"，以标识对应层级
    """
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            # 创建新的子主题
            topic = parent_topic.addSubTopic()
            if add_plus and not key.startswith(("用例标题", "步骤", "预期结果")):
                # 匹配并去除"n级分组: "前缀
                match = re.match(r"([一二三四五六七八九十]+)级分组: (.+)", key)
                if match:
                    key = match.group(2)
                key = "+" + key
            # 将 "用例名称: xxx"改为"xxx"
            if key.startswith("用例标题:"):
                key = key.replace("用例标题: ", "")
            topic.setTitle(key)

            # 如果当前键为"步骤"，则将add_plus设置为False，不再添加"+"
            if key == "步骤":
                add_plus = False

            create_xmind_from_json(value, topic, add_plus)
    elif isinstance(json_obj, list):
        for item in json_obj:
            create_xmind_from_json(item, parent_topic, add_plus)
    else:
        # 对于非字典和列表的叶子节点，创建一个新的子主题
        topic = parent_topic.addSubTopic()
        topic.setTitle(str(json_obj))


def generate_xmind_from_json(json_data, file_name='output.xmind'):
    """
    从JSON数据生成XMind文件
    """
    # 清理临时XMind文件
    temp_file = glob.glob('*.xmind')
    for file in temp_file:
        if os.path.isfile(file):
            os.remove(file)

    preprocess(json_data)  # 预处理json数据
    workbook = xmind.load(file_name)  # 加载或创建 xmind 工作簿
    first_sheet = workbook.getPrimarySheet()  # 获取第一个工作表
    first_sheet.setTitle("测试用例")  # 设置工作表的标题
    root_topic = first_sheet.getRootTopic()  # 获取根主题
    root_topic.setTitle("测试用例集")  # 设置根主题的标题

    # 设置根主题的布局为逻辑图（所有分支在同一侧）
    root_topic.setStructureClass("org.xmind.ui.logic.right")

    create_xmind_from_json(json_data, root_topic)

    xmind.save(workbook, file_name)  # 保存工作簿到指定文件
    return file_name


def open_and_delete_xmind(file_name):
    """
    打开XMind文件并返回文件流，然后删除文件。
    """
    try:
        with open(file_name, 'rb') as file:
            file_stream = file.read()
        os.remove(file_name)
        print(f"temp file {file_name} has been deleted.")
        return BytesIO(file_stream)
    except Exception as e:
        print(f"Error: {e}")


def fix_json_str(json_str: str, max_attempts: int = 5) -> str | None:
    """修复因大模型输出限制被截断的json"""
    attempts = 0
    ori_json_str = json_str

    # 法1：在末尾添加"}"，直到解析成功
    while attempts < max_attempts:
        try:
            # 尝试解析JSON字符串
            json.loads(json_str)
            logger.info("Successfully repairing JSON data")
            return json_str
        except json.JSONDecodeError:
            # 如果解析失败，在末尾添加一个}
            json_str += '}'
            attempts += 1

    # 法2：若截断发生在"步骤"，则先添加"]"再添加"}" 例：
    # "三级分组: xxx": {
    #     "用例标题: xxx": {
    #         "步骤": [
    #             {
    #                 "步骤1": "xxx",
    #                 "预期结果": "xxx"
    #             }
    json_str = ori_json_str  # 重置json_str
    json_str += ']}}}}}'
    try:
        json.loads(json_str)
        logger.info("Successfully repairing JSON data")
        return json_str
    except json.JSONDecodeError:
        pass
    # 如果尝试5次后仍然失败，返回None
    logger.error("Failed to repair JSON data")
    return None


def merge_json(*json_objects):
    """
    合并多个JSON对象为一个JSON对象
    """
    merged_json = {}
    for json_obj in json_objects:
        if isinstance(json_obj, dict):
            merged_json.update(json_obj)
    return merged_json


def gen_xmind(json_data: dict) -> BytesIO:
    """
    根据JSON生成xmind
    """
    file_name = generate_xmind_from_json(json_data=json_data)
    return open_and_delete_xmind(file_name=file_name)


if __name__ == '__main__':
    """
    将json转为xmind
    """
    json_data = {
        "一级分组: 功能测试": {
            "二级分组: 资方收支分离诉求": {
                "三级分组: 交易清分计费": {
                    "用例标题: 正向交易D+1日资方汇总结算": {
                        "步骤": [
                            {
                                "步骤1": "模拟D日发生的正向交易",
                                "预期结果": "交易成功，等待D+1日资方汇总结算"
                            },
                            {
                                "步骤2": "到达D+1日，检查资方汇总结算是否发生",
                                "预期结果": "资方汇总结算成功，备付金结算"
                            }
                        ]
                    },
                    "用例标题: 退款交易D日结算到华润银行账户": {
                        "步骤": [
                            {
                                "步骤1": "模拟D日用户发起退款请求",
                                "预期结果": "退款请求成功，等待当日结算到华润银行账户"
                            },
                            {
                                "步骤2": "检查退款是否在D日结算到华润银行账户",
                                "预期结果": "退款成功结算到华润银行账户，融资人额度恢复"
                            }
                        ]
                    }
                }
            },
            "二级分组: 系统流程": {
                "三级分组: 交易统一事件处理": {
                    "用例标题: 退款交易统一事件识别特殊标识": {
                        "步骤": [
                            {
                                "步骤1": "模拟退款成功的交易统一事件，包含特殊标识",
                                "预期结果": "系统正确识别特殊标识，并按退款处理逻辑落库"
                            }
                        ]
                    }
                },
                "三级分组: 指令结算支持": {
                    "用例标题: 指令结算周期D0立即处理": {
                        "步骤": [
                            {
                                "步骤1": "选择D0作为结算周期，发起结算指令",
                                "预期结果": "系统立即处理结算和付款"
                            }
                        ]
                    },
                    "用例标题: 指令结算周期D1处理结算": {
                        "步骤": [
                            {
                                "步骤1": "选择D1作为结算周期，发起结算指令",
                                "预期结果": "系统在D+1日处理结算，不立即付款"
                            }
                        ]
                    }
                }
            },
            "二级分组: 对账系统": {
                "三级分组: 对账文件核对": {
                    "用例标题: 核对对账文件与银行明细": {
                        "步骤": [
                            {
                                "步骤1": "获取对账文件和银行明细",
                                "预期结果": "对账文件与银行明细匹配，无差异"
                            }
                        ]
                    }
                }
            }
        },
        "一级分组: 边界值测试": {
            "二级分组: 交易金额": {
                "三级分组: 最小金额": {
                    "用例标题: 正向交易最小金额": {
                        "步骤": [
                            {
                                "步骤1": "发起一笔正向交易，交易金额设置为允许的最小金额",
                                "预期结果": "交易成功，无错误提示"
                            }
                        ]
                    },
                    "用例标题: 退款交易最小金额": {
                        "步骤": [
                            {
                                "步骤1": "发起一笔退款交易，交易金额设置为允许的最小金额",
                                "预期结果": "退款成功，无错误提示"
                            }
                        ]
                    }
                },
                "三级分组: 最大金额": {
                    "用例标题: 正向交易最大金额": {
                        "步骤": [
                            {
                                "步骤1": "发起一笔正向交易，交易金额设置为允许的最大金额",
                                "预期结果": "交易成功，无错误提示"
                            }
                        ]
                    },
                    "用例标题: 退款交易最大金额": {
                        "步骤": [
                            {
                                "步骤1": "发起一笔退款交易，交易金额设置为允许的最大金额",
                                "预期结果": "退款成功，无错误提示"
                            }
                        ]
                    }
                }
            }
        },
        "一级分组: 错误猜测": {
            "二级分组: 交易类型错误": {
                "三级分组: 交易类型与子类型不匹配": {
                    "用例标题: 交易类型与子类型不一致": {
                        "步骤": [
                            {
                                "步骤1": "发起一笔交易，故意设置交易类型与子类型不一致",
                                "预期结果": "系统拒绝该交易，返回类型不匹配错误"
                            }
                        ]
                    }
                }
            }
        }
    }

    # 生成XMind文件
    # merged_json = merge_json(*json_data_list)  # 合并JSON
    file_name = generate_xmind_from_json(json_data, "output.xmind")
    file_stream = open_and_delete_xmind(file_name)  # 删除xmind，返回文件流
    # 重新用文件流生成xmind
    with open('output.xmind', 'wb') as new_file:
        new_file.write(file_stream.getvalue())
