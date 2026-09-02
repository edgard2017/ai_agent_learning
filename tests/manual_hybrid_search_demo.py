"""直接运行：python -m tests.manual_hybrid_search_demo"""

from ocean_agent.manual_hybrid_search import hybrid_search_manual_chunks


QUERIES = (
    ("SBE 19plus如何连接电脑并上传数据？", "seabird-sbe-19plus-v2"),
    ("RBR外部MCBH接口的针脚定义是什么？", "rbr-concerto3-ctd"),
    ("RBR仪器应该怎样更换O型圈？", "rbr-concerto3-ctd"),
    ("Can USB-C power the RBR instrument while sampling?", "rbr-concerto3-ctd"),
)


def _summary(chunk) -> str:
    return chunk.content[:150].replace("\n", " ")


def main() -> None:
    for query, product_id in QUERIES:
        print("\n" + "=" * 88)
        print(f"问题: {query}")
        groups = hybrid_search_manual_chunks(
            query,
            product_id=product_id,
            limit=5,
            candidate_limit=20,
        )
        for rank, group in enumerate(groups, start=1):
            match = group.anchor
            chunk = match.chunk
            print(
                f"\n#{rank} methods={match.retrieval_methods} "
                f"keyword_rank={match.keyword_rank} "
                f"embedding_rank={match.embedding_rank} "
                f"page={chunk.page_number} review={chunk.review_status}"
            )
            print(f"ANCHOR {chunk.chunk_id} | {chunk.section}")
            print(_summary(chunk))
            if group.previous_chunk:
                print(
                    f"  PREV {group.previous_chunk.chunk_id} "
                    f"| page {group.previous_chunk.page_number} "
                    f"| {_summary(group.previous_chunk)}"
                )
            if group.next_chunk:
                print(
                    f"  NEXT {group.next_chunk.chunk_id} "
                    f"| page {group.next_chunk.page_number} "
                    f"| {_summary(group.next_chunk)}"
                )


if __name__ == "__main__":
    main()
