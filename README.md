# AICaseForge: Automated test case generation
> 测试用例智能生成平台，仅需上传PRD文档，便可自动化生成多种格式（XMind、Excel、Markdown）的测试用例，帮助持续优化测试技术、提高测试效率

# 环境准备

1. 要求：`python >= 3.10`
2. 执行`pip install -r requirements.txt`命令安装依赖包
3. 更改`static/config/llm_config.ini`配置，配置openai api-key，例

```
[api_key]
API_KEY = aa341ea-b423bb-c423cc-d45d-d73b48759ae9
```

# 项目结构

```commandline
test-llm-generate/
├─ bin/  
├─ common/  
├─ model/  # 文本向量化本地模型  
├─ src/  
│  ├─ doc_preprocess/  # 文档预处理模块
│  ├─ embedding/  # 嵌入模块
│  ├─ generator/  # 测试用例生成模块
│  ├─ knowledge_injection/  # 知识注入模块
│  └─ llm/  # 大模型SDK接口封装
├─templates/  # 前端HTML模板文件
├─ static/  
│  ├─ config/  # 配置文件，包括用例模板、大模型对话模板等
│  ├─ css/  # CSS样式文件
│  ├─ images/  # 图片文件
│  └─ js/  # JavaScript文件
└─ test/  # 测试代码和测试数据
```





