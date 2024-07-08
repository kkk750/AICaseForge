import argparse
import base64
import os
import time

import openai
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, PromptTemplate
from src.llm.llm_factory import LLMFactory


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_understanding_gpt4o(image_path, prompt):
    base64_image = encode_image(image_path)
    llm = LLMFactory.get_openai_factory().get_gpt4o()
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            }
        ]
    )
    print(llm.invoke([message]))


def image_understanding_gpt4v(image_path, prompt):
    base64_image = encode_image(image_path)
    llm = LLMFactory.get_openai_factory().get_gpt4v()
    output = llm.invoke(
        [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{base64_image}"
                    },
                ]
            )
        ]
    )
    print(output)


if __name__ == "__main__":
    """
    测试各种大模型的调用效果，可以查看模型调用耗时
    
    命令行参数:
    - model: 模型名称，可选项
        gpt3：文字生成
        gpt4：文字生成
        gpt4v：图片理解
        gpt4o：图片理解
        gpt4o-chat：文字生成
        
    - input_file: 输入文件的路径，若调用图片理解模型，则填入图片路径，若调用文字生成模型则为空
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt4o", help="模型名称，可选值：'gpt3'、'gpt4'、'gpt4v'、'gpt4o'、'gpt4o-chat'")
    parser.add_argument("--input_file", type=str, default="D:\\download\\untitled.jpg")
    args = parser.parse_args()
    model = args.model
    input_file = args.input_file

    # 图片理解prompt
    prompt_image_understanding = """
    请详细描述这幅图片，将流程图、时序图转换为Mermaid对应的code，保留所有类别标签。
    """
    # 文字生成prompt
    prompt_text_generation = """
    请讲一段长度为400字的小故事
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=prompt_text_generation
            ),
            HumanMessagePromptTemplate.from_template(
                "{input}"
            ),
        ]
    )

    s = time.perf_counter()

    if model == 'gpt4v':
        image_understanding_gpt4v(input_file, prompt_image_understanding)
    elif model == 'gpt4o':
        image_understanding_gpt4o(input_file, prompt_image_understanding)
    else:
        if model == 'gpt3':
            llm = LLMFactory.get_openai_factory().get_gpt3()
        elif model == 'gpt4':
            llm = LLMFactory.get_openai_factory().get_chat_llm()
        elif model == 'gpt4o-chat':
            llm = LLMFactory.get_openai_factory().get_gpt4o()
        # elif model == 'claude-3-sonnet':
        #     llm = LLMFactory.get_openai_factory().get_claude_3_sonnet()
        conversation = ConversationChain(
            llm=llm,
            memory=ConversationBufferMemory(),
            verbose=True
        )
        output = conversation.run(prompt_text_generation)
        print(output)

    elapsed = time.perf_counter() - s
    print("\033[1m" + f"\n模型名称: {model}; 调用时长：{elapsed:0.2f} s." + "\033[0m")
