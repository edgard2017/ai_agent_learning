"""直接运行：python -m tests.manual_embedding_search_demo"""

from ocean_agent.manual_embedding_index import semantic_search_manual_chunks


QUERIES = (
    ("SBE 19plus如何连接电脑并上传数据？", "seabird-sbe-19plus-v2"),
    ("RBR外部MCBH接口的针脚定义是什么？", "rbr-concerto3-ctd"),
    ("RBR仪器应该怎样更换O型圈？", "rbr-concerto3-ctd"),
    ("Can USB-C power the RBR instrument while sampling?", "rbr-concerto3-ctd"),
)


def main() -> None:
    for query, product_id in QUERIES:
        print("\n" + "=" * 80)
        print(f"问题: {query}")
        matches = semantic_search_manual_chunks(
            query,
            product_id=product_id,
            limit=5,
        )
        for match in matches:
            chunk = match.chunk
            print(
                f"#{match.rank} similarity={match.similarity:.4f} "
                f"page={chunk.page_number} review={chunk.review_status}"
            )
            print(f"{chunk.chunk_id} | {chunk.section}")
            print(chunk.content[:180].replace("\n", " "))


if __name__ == "__main__":
    main()
