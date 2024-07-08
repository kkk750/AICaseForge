import os


class FilePath:
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 输出内容相关文件路径
    empty_case = os.path.join(root_path, 'static', 'config', 'case_config.json')  # json格式的测试用例模板
    empty_case_md = os.path.join(root_path, 'static', 'config', 'case_config.md')  # markdown格式的测试用例模板
    images = os.path.join(root_path, 'downloads')  # 临时文件目录，用于存放pdf提取的图像
    out_file = os.path.join(root_path, 'static', 'output')

    # 项目需要的配置文件路径
    config_file_path = os.path.join(root_path, 'static', 'config', 'llm_config.ini')

    # 停用词路径
    stopwords_file_path = os.path.join(root_path, 'static', 'config', 'stopwords.txt')

    @staticmethod
    def read_file(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except FileNotFoundError:
            print(f"文件未找到: {file_path}")
            return None
        except Exception as e:
            print(f"读取文件时发生错误: {file_path}, 错误信息: {e}")
            return None
