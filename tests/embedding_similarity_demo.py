"""运行真实本地模型，打印不同句子之间的余弦相似度。

运行方式：
python -m tests.embedding_similarity_demo
"""

from ocean_agent.config import get_settings
from ocean_agent.embedding_search import (
    cosine_similarity,
    create_embedding,
    create_embeddings,
)


TEST_CASES = (
    (
        "中文测试",
        "如何把海洋仪器连接到计算机？",
        (
            "设备通过 RS-232 串口与电脑通信。",
            "仪器可以使用外部电源供电。",
            "钛合金壳体最大工作深度为6000米。",
            "今天杭州天气很好，适合去公园散步。",
        ),
    ),
    (
        "English control test",
        "How can I connect the ocean instrument to a computer?",
        (
            "The device communicates with a computer through an RS-232 serial port.",
            "The instrument can use an external power supply.",
            "The titanium housing has a maximum working depth of 6000 meters.",
            "The weather in Hangzhou is pleasant for a walk in the park.",
        ),
    ),
)


def print_case(title: str, query: str, sentences: tuple[str, ...]) -> int:
    settings = get_settings()
    query_vector = create_embedding(query, input_type="query", settings=settings)
    sentence_vectors = create_embeddings(
        sentences,
        input_type="document",
        settings=settings,
    )

    rows = [
        (cosine_similarity(query_vector, vector), sentence)
        for sentence, vector in zip(sentences, sentence_vectors, strict=True)
    ]
    rows.sort(key=lambda item: item[0], reverse=True)

    print(f"\n=== {title} ===")
    print(f"问题: {query}")
    print("\n余弦相似度（越接近1，语义方向越接近）：")
    for similarity, sentence in rows:
        print(f"{similarity:.6f}  {sentence}")
    return len(query_vector)


def main() -> None:
    settings = get_settings()
    print(f"Embedding 模型: {settings.ollama_embedding_model}")

    dimensions = [
        print_case(title, query, sentences)
        for title, query, sentences in TEST_CASES
    ]
    print(f"\n向量维度: {dimensions[0]}")


if __name__ == "__main__":
    main()
