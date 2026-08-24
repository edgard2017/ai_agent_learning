# Embedding 模型评测

这里保存独立于 Agent 的中英跨语言海洋设备检索测试。

## 评测方向

- 中文问题检索英文资料；
- 英文问题检索中文资料；
- 中文问题检索中文资料；
- 英文问题检索英文资料；
- 同一产品不同参数的硬负例；
- 当前资料没有答案的查询。

## 指标

- `Top-1 Accuracy`：排在第一的资料是否正确；
- `Hit@3`：前三名是否至少包含一个正确资料；
- `Recall@3`：全部正确资料中有多少进入前三；
- `MRR`：正确资料排名越靠前，分数越高；
- `nDCG@3`：同时考虑前三名中多个正确资料的位置；
- `Mean Margin`：正确资料分数领先最佳错误资料多少；
- 无答案查询的最高相似度：用于以后研究拒答阈值。

## 运行

使用独立环境，模型一次只运行一个：

```bash
conda activate embedding-benchmark
python benchmarks/run_embedding_benchmark.py --model qwen3-0.6b
python benchmarks/run_embedding_benchmark.py --model bge-m3
python benchmarks/run_embedding_benchmark.py --model multilingual-e5
python benchmarks/run_embedding_benchmark.py --model nomic-v2
```

运行包含40题的困难压力测试：

```bash
python benchmarks/run_embedding_benchmark.py \
  --model qwen3-0.6b \
  --dataset benchmarks/ocean_embedding_stress_cases.json \
  --output-dir benchmarks/results/stress
```

评测数据结构测试：

```bash
python -m unittest benchmarks.test_benchmark_dataset -v
```

完整结果保存在 `benchmarks/results/`。这个小型测试集用于项目选型，不能替代
MTEB、C-MTEB、MIRACL 和 MLDR 等公开通用基准。

本机四模型横向结果和选择理由见 [`RESULTS.md`](RESULTS.md)。评测依赖单独保存在
`benchmarks/requirements.txt`，没有加入正式 Agent 的运行环境。
