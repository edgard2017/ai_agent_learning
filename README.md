# 海洋设备智能售前与技术服务 Agent（学习项目）

当前已完成第一个本地 RAG 版本：
**可核验产品数据 + 技术资料片段 + 4个 Agents SDK Tool + 本地 Qwen 单 Agent**。

## 数据边界

- 当前 4 条记录来自 Sea-Bird Scientific 和 RBR 的厂家官网公开资料。
- 它们统一标记为 `third_party_public_product`，不表示杭州海询科技有限公司自研、代理、在售或作出服务承诺。
- 官网没有确认的参数保持为空；配置相关的深度、压力范围和采样率不合并成一个无条件数字。
- 盐度、深度等属于根据 CTD 原始测量值计算的派生参数，不写成传感器直接测量值。

## 目录

```text
ocean_agent/
├── models.py         # Pydantic 数据模型
├── product_data.py   # 4 条厂家公开产品记录
├── document_data.py  # 7 个带来源的技术资料片段
├── tools.py          # 产品查询和本地关键词检索
├── agent_tools.py    # Agents SDK Tool 包装和 JSON 输出
├── agent.py          # Agent、Runner 和命令行入口
├── sessions.py       # SQLite 多轮会话与本地持久化
└── config.py         # 从 .env 安全加载配置
tests/
├── test_tools.py       # 产品查询测试
├── test_agent_tools.py # Tool schema 和 Agent 注册测试
└── test_config.py      # 配置与密钥保护测试
```

## 运行测试

```bash
cd /mnt/scatch/xiaolonz/ai_agent_learning
conda activate ocean-agent
python -m pip install -r requirements.txt
python3 -m unittest discover -v
```

## 模型配置：默认使用本地 Qwen

项目默认连接本机 Ollama 的 OpenAI 兼容接口，不消耗 OpenAI API 额度：

```dotenv
MODEL_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11435/v1
OLLAMA_MODEL=hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
```

如需切回 OpenAI，只需修改 `.env`，Agent 和 Tool 代码不用改：

```dotenv
MODEL_PROVIDER=openai
OPENAI_API_KEY=你的API-Key
OPENAI_MODEL=gpt-5.6-luna
```

`.env` 已被 Git 忽略。代码通过 `ocean_agent.config.get_settings()` 读取配置，
并使用 `SecretStr` 避免在普通日志和对象输出中显示完整 Key。

## 不消耗 API 额度：直接测试产品函数

```python
from ocean_agent.tools import search_products

products = search_products(
    minimum_depth_m=5000,
    required_parameters=["温度", "盐度", "压力"],
)
for product in products:
    print(product.manufacturer, product.model)
```

## 运行第一个 Agent

下面的命令默认请求本地 Qwen，不产生 OpenAI API 费用：

```bash
conda activate ocean-agent
cd /mnt/scatch/xiaolonz/ai_agent_learning
python -m ocean_agent.agent "我需要在5000米水深测量温度、盐度和压力，请给出候选产品并说明配置风险。"
```

学习 Runner Loop 时可以开启调试模式：

```bash
python -m ocean_agent.agent --debug "请介绍 SBE 19plus V2 SeaCAT 的工作深度和通信接口。"
```

调试信息按时间线显示每轮模型行为、Runner 执行的 Tool、参数和返回长度，
不打印完整产品 JSON，也不展示模型的隐藏推理内容。

执行过程可以用大白话理解为：

1. 用户把自然语言需求交给 Agent。
2. 模型决定调用 `search_ocean_products`，并把水深和参数整理成结构化入参。
3. 本地 Python 函数只从已核验目录筛选产品，返回 JSON，不让模型自行编型号。
4. 模型根据 Tool 结果整理中文答复，并提醒选配、派生参数和配置风险。

当前 Agent 提供4个 Tool：

- `search_ocean_products`：按条件筛选候选产品。
- `get_ocean_product_details`：查询一个明确型号的完整详情。
- `compare_ocean_products`：按统一字段比较两个或多个候选产品。
- `search_ocean_documents`：检索带厂家来源的本地技术资料片段。

多 Tool Loop 示例：

```bash
python -m ocean_agent.agent --debug \
  "请先筛选能在5000米测量温度、盐度和压力的CTD，再比较全部候选产品。"
```

## 多轮会话

普通命令每次都是一段独立对话。传入相同的 `--session-id` 后，Runner 会从本地
SQLite 数据库读取这个会话的历史记录，并在本轮结束后自动保存新增内容：

```bash
python -m ocean_agent.agent --chat --session-id learning-demo
```

可以连续输入：

```text
我需要在5000米测量温度、盐度和压力，请筛选候选产品。
比较它们的通信接口。
/exit
```

记录默认保存在 `.agent_data/conversations.db`，该目录已加入 `.gitignore`，不会提交
用户聊天数据。不同的 `session-id` 相当于不同聊天窗口；再次使用相同 ID，可以在程序
重启后继续之前的对话。

这里学习的仍然是 OpenAI Agents SDK：`Agent`、`Runner`、Function Tool 和运行循环没有改变；
只是把负责理解问题和决定是否调用 Tool 的模型，从 OpenAI 模型替换成了本地 Qwen。

## 本地技术资料检索（第一版 RAG）

本项目暂时不使用向量数据库。技术资料先拆成可以独立引用的短片段，再用普通 Python
关键词检索。Agent 遇到操作、连接、接线、供电、采样设置、维护、校准、故障排查或
说明书问题时，可以调用 `search_ocean_documents`：

```bash
python -m ocean_agent.agent --debug \
  "请查技术资料：SBE 19plus V2 使用什么通信接口，公开资料能否说明具体接线步骤？"
```

Tool 返回内容包括：

- 相关资料片段；
- 产品 ID、章节和资料标题；
- 厂家来源 URL；
- 只用于排序的相关度分数。

当前7个片段是对已核验厂家产品页和数据表的中文整理，并非完整产品手册。没有收录的
校准步骤、故障码或接线细节必须回答“当前资料不足”，不能让模型凭记忆补写。

这里的 RAG 可以用一句话理解：

```text
用户问题 → 搜索相关资料片段 → 把片段交给模型 → 模型根据片段回答并标明来源
```
