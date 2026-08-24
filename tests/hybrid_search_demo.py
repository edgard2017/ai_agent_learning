"""直接观察关键词检索与混合检索差异。"""

from ocean_agent.hybrid_search import hybrid_search_documents
from ocean_agent.tools import search_documents


CASES = (
    ("19plus怎么把数据传到电脑里？", None),
    ("MicroCAT的压力是不是选配？", None),
    ("怎么把数据传到电脑里？", "SBE 19plus V2"),
)


def main() -> None:
    for query, model_or_id in CASES:
        print(f"\n问题: {query}")
        print(f"型号过滤: {model_or_id or '未指定'}")

        keyword_matches = search_documents(
            query,
            model_or_id=model_or_id,
            limit=3,
        )
        print("关键词结果:")
        if not keyword_matches:
            print("  无")
        for match in keyword_matches:
            print(f"  {match.chunk.chunk_id}: keyword_score={match.score}")

        print("混合结果:")
        for match in hybrid_search_documents(
            query,
            model_or_id=model_or_id,
            limit=3,
        ):
            similarity = (
                f"{match.embedding_similarity:.4f}"
                if match.embedding_similarity is not None
                else "None"
            )
            print(
                f"  {match.chunk.chunk_id}: fused={match.fused_score:.6f}, "
                f"keyword_rank={match.keyword_rank}, "
                f"embedding_rank={match.embedding_rank}, "
                f"similarity={similarity}"
            )


if __name__ == "__main__":
    main()
