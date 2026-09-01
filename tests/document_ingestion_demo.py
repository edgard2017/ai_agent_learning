"""直接运行：python -m tests.document_ingestion_demo"""

from pathlib import Path

from ocean_agent.document_chunker import chunk_documents
from ocean_agent.document_loader import load_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    documents = load_documents(PROJECT_ROOT / "documents")
    chunks = chunk_documents(documents, max_chars=180)

    print(f"读取文档数: {len(documents)}")
    print(f"生成 Chunk 数: {len(chunks)}")
    for chunk in chunks:
        print("\n" + "=" * 72)
        print(f"chunk_id: {chunk.chunk_id}")
        print(f"product_id: {chunk.product_id}")
        print(f"section: {chunk.section}")
        print(f"字符数: {len(chunk.content)}")
        print(f"source: {chunk.source.url}")
        print(f"content: {chunk.content}")


if __name__ == "__main__":
    main()
