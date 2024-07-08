import re
from functools import wraps

# 手机号，可匹配如下
# （+86） 199 0000 1715
# （+0086） 199 0000 1715
# （+86） 199-0000-1715
# (+86) 199-0000-1715
# (+86) 199-0000-1715
# +86 199-0000-1715
# +86199-0000-1715
# +86 199 0000 1715
# 86 199 0000 1715
# +8619900001715
# 8619900001715
# 199-0000-1715
# 199 0000 1715
# 19900001715
PHONE_NUM = re.compile(
    r"\b(\+?0?0?86[-\s]?)?(13[0-9]|14[01456879]|15[0-35-9]|16[2567]|17[0-8]|18[0-9]|19[0-35-9])[-\s]?\d{4}[-\s]?\d{4}\b")

# 身份证号
ID_RE_18 = re.compile(
    r"\b([1-6][1-9]|50)\d{4}(18|19|20)\d{2}((0[1-9])|10|11|12)(([0-2][1-9])|10|20|30|31)\d{3}[0-9Xx]\b")
ID_RE_15 = re.compile(r"\b([1-6][1-9]|50)\d{4}\d{2}((0[1-9])|10|11|12)(([0-2][1-9])|10|20|30|31)\d{3}\b")

# 电子邮箱
EMAIL_ADDRESS = re.compile(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+")

# 密码
PASSWORD = re.compile(r"^(?=.*?[a-z])(?=.*?[0-9])(?=.*?[#+?!@$%^&*-]).{4,}$")

# md链接
MD_LINK = re.compile(r"(http[s]?://[^\s\n]+(?:\s|\n)+[^\s\n]+)")


def optional_clean_output_tags(enable: bool):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            text = func(*args, **kwargs)
            if enable:
                text = clean_output_tags(text)
            return text

        return wrapper

    return decorator


def optional_clean_spaces_and_dashes(enable: bool):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            text = func(*args, **kwargs)
            if enable:
                text = clean_spaces_and_dashes(text)
            return text

        return wrapper

    return decorator


def clean_links(text: str) -> str:
    """清理链接
    For example:

    '''UI： https://www.xxx.com/web/project/page/a7ff5403-

    16c4a96245cc/'''

    Gets converted to
    '''UI： https://www.xxx.com/web/project/page/a7ff5403-16c4a96245cc/'''
    """

    def replace_link(match):
        # 匹配链接，包括中间可能的换行符和空格
        link = match.group(0)
        cleaned_link = link.replace("\n", "").replace(" ", "")
        return cleaned_link

    matches = MD_LINK.findall(text)

    if not matches:
        return text

    cleaned_text = re.sub(MD_LINK, replace_link, text)
    return cleaned_text


def clean_empty_lines(text: str) -> str:
    """将多个连续的空行替换为1个换行符"""
    text = text.replace('\r', '')
    return re.sub(r"\n\s*?\n", "\n", text)


def clean_curly_braces(text: str) -> str:
    """将{}替换为【】"""
    return text.replace("{", "【").replace("}", "】")


def clean_output_tags(text: str) -> str:
    """删除<output></output>标签对"""
    return text.replace("<output>", "").replace("</output>", "")


def clean_spaces_and_dashes(text: str) -> str:
    """清理空格和横线"""
    # 将连续的多个空格替换为1个空格
    text = re.sub(r' +', ' ', text)
    # 将连续的多个-替换为1个-
    text = re.sub(r'-+', '-', text)
    return text


def clean_sensitive_words(text: str) -> str:
    """敏感词脱敏处理"""

    def phone_desensitization(match):
        # 手机号脱敏
        # 将手机号中的第4到第6位替换为 'x'
        phone_number = match.group()
        return phone_number[:3] + 'xxx' + phone_number[6:]

    def email_desensitization(match):
        # 邮箱脱敏
        # 将 @xxx.com 换为 @xxx*com
        mail_add = match.group()
        return mail_add.split('.')[0] + "*" + mail_add.split('.')[1]

    def password_desensitization(text):
        # 密码脱敏，替换为'*'
        if PASSWORD.match(text):
            show_count = min(len(text), 3)  # 默认只正常显示后三位
            text = '*' * (len(text) - show_count) + text[-show_count:]
        # 将 '密码' 改为 '密/码'
        return re.sub(r'密码', '密/码', text)

    desensitized_text = PHONE_NUM.sub(phone_desensitization, text)
    desensitized_text = EMAIL_ADDRESS.sub(email_desensitization, desensitized_text)
    desensitized_text = password_desensitization(desensitized_text)
    return desensitized_text


@optional_clean_output_tags(enable=False)
@optional_clean_spaces_and_dashes(enable=True)
def clean(text: str) -> str:
    """清理文本"""
    cleaned_text = clean_links(text)
    cleaned_text = clean_curly_braces(cleaned_text)
    cleaned_text = clean_sensitive_words(cleaned_text)
    cleaned_text = clean_empty_lines(cleaned_text)
    return cleaned_text


if __name__ == "__main__":
    """文本清洗器"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, default='D:\\code\\test-llm-generete\\test\\原文.txt',
                        help='待清洗的字符串或txt文件路径')
    args = parser.parse_args()
    text = args.text

    if text.endswith('.txt'):
        with open(text, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        pass

    cleaned_text = clean(text)
    print(cleaned_text)
