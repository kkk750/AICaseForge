import jieba
import jieba.analyse

from common.file_path import FilePath


def Extract_Keywords(text: str, keywords_num: int) -> list:
    # 设置停用词
    stopwords_file_path = FilePath.stopwords_file_path
    jieba.analyse.set_stop_words(stopwords_file_path)

    # 提取关键词
    keywords = jieba.analyse.extract_tags(text, topK=keywords_num)

    return keywords


if __name__ == '__main__':
    """使用TF-IDF提取关键词"""

    # 待提取关键词的文本
    text = """
【RPF构建】
在机器人创建完成后可以点击配置进入如下图所示的RPF配置页面，配置自己所需的流
程，RPF中会初始化当前基线版本的流程。具体内容让如下：
其中初始化的内容分为5个基础流程（流程的定义以一次调用为准，要完成整体大流程可能出现的会有以下这5个流程；有些是基于实现来适配，如TTS语音获取流程）
• 破冰流程
• 输入补全流程
• 应答流程
• 结束流程（用于结束会话）
• TTS语音流获取流程（语音上的特殊情况兼容）
每个流程独立支持暂存和启停，如果启用则生效，停用则无效，流程的启停状态和机器人的启停状态独立，但从合理角度讲，需要机器人和对应期望的流程都是启用状态，机器人才能达到预期的状态；无论是机器人本身是停用，或者是流程是停用，那么该机器人下的该流程都无法正常生效；此外我们是需要支持在流程启用的状态下，无缝更新流程的；即更新流程和启用停用也是独立的；不互相干扰。如果是启用状态，更新生效前的用户走新流程，更新生效后的用户走老流程。其中通用节点中，包含俩类RPU单元
第一类为逻辑类RPU，也就是所谓的父节点，主要分为串行单元和并行单元，使用土黄色表示，这俩个节点默认初始化好，无需单独配置；
第二类为功能类RPU，也就是子类，即应答单元，使用粉红色表示，该类单元可以在应答单元配置中生成客制化节点，则均为功能类RPU，使用蓝色表示，单有机器人功能超出BASE而又无法通用抽象时，则可以新实现客制化RPU实例，通过应答单元配置后，进入RPF配置供选择使用。用户在调整自己的流程后，可以通过【使用基线版本】功能，一键回到基线版本RPF；
"""

    keywords = Extract_Keywords(text, keywords_num=10)

    print("关键词：")
    for keyword in keywords:
        print(keyword)
