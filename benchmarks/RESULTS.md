# Embedding 模型领域测试结果

测试日期：2026-08-24

## 测试目的

选择适合“中文提问、中文或英文海洋设备资料”的本地 Embedding 模型。测试集包含
16 个中英文资料片段、10 个有正确答案的问题和 2 个当前资料无答案的问题。候选答案中
刻意加入了同一产品的不同参数，避免模型只凭产品名命中。

这是一组项目选型小测试，不是通用排行榜，也不能替代大规模公开基准。

## 本机实测结果

| 模型 | Top-1 | Recall@3 | MRR | 平均领先分差 | 中→英分差 | 英→中分差 | 维度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BGE-M3 | 100% | 100% | 1.000 | 0.1406 | 0.1371 | 0.1182 | 1024 |
| Qwen3-Embedding-0.6B | 100% | 100% | 1.000 | 0.1293 | **0.1602** | 0.1049 | 1024 |
| Nomic Embed Text v2 MoE | 100% | 100% | 1.000 | 0.1195 | 0.0912 | 0.1071 | 768 |
| multilingual-e5-large-instruct | 100% | 100% | 1.000 | 0.0585 | 0.0626 | 0.0529 | 1024 |

“平均领先分差”是正确资料的相似度减去最强错误资料的相似度。四个模型虽然都排对了，
但 BGE-M3 和 Qwen3 的正确答案通常与错误答案拉得更开。E5 的绝对相似度经常集中在较高
区间，因此不能把不同模型的相似度数字直接横向比较。

## 当前选择

第一阶段建议优先使用 **BGE-M3**：本项目小测试的总体领先分差最大，中英双向表现均衡，
模型支持多语言和较长文本，而且许可证为 MIT。若项目主要是“中文问题检索英文资料”，
Qwen3-Embedding-0.6B 也很值得保留为对照，它在这个方向的领先分差最高。

暂时不把任何模型接入正式 RAG。当前测试只有 10 个有答案案例，下一步要扩展到至少
30～50 个真实问题，并加入更长文档、缩写、错别字、多个正确片段和更多无答案问题。

## 无答案问题

| 模型 | 两个无答案查询的最高相似度均值 |
| --- | ---: |
| BGE-M3 | 0.6143 |
| Qwen3-Embedding-0.6B | 0.6375 |
| Nomic Embed Text v2 MoE | 0.5090 |
| multilingual-e5-large-instruct | 0.8213 |

这些数字不能直接作为统一拒答阈值：每个模型的分数分布不同，而且只有两个无答案案例。
生产系统通常还会结合关键词命中、重排模型、来源质量和答案证据判断是否拒答。

## 公开通用测试

- [MTEB](https://docs.mteb.org/overview/)：综合评测检索、语义相似度、分类、聚类、重排等能力。
- [MMTEB](https://docs.mteb.org/overview/available_benchmarks/)：MTEB 的大规模多语言版本，覆盖 250 多种语言。
- [C-MTEB 论文](https://openreview.net/pdf?id=yN5t4WDyEL)：专门评测中文文本向量能力。
- [MIRACL](https://github.com/project-miracl/miracl)：包含中文在内的 18 种语言人工标注检索数据集。

候选模型的官方模型卡也公开了基准结果和使用方法：
[Qwen3 Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)、
[BGE-M3](https://huggingface.co/BAAI/bge-m3)、
[multilingual-e5](https://huggingface.co/intfloat/multilingual-e5-large-instruct)、
[Nomic v2](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe)。

公开分数适合初筛模型，但海洋设备型号、参数表达和中英混合属于本项目自己的数据分布，
最终选择仍应以领域测试为准。

## 复现说明

测试在 RTX 4090 上以 FP16、batch size 8 逐模型运行。模型首次运行包含下载和加载时间，
因此本次记录的加载耗时不适合作为严格性能比较；编码耗时也因为测试集很小，只用于确认
模型能够正常在 GPU 上工作。完整逐案例排名保存在 `benchmarks/results/*.json`。
