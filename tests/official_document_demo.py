"""直接运行：python -m tests.official_document_demo"""

from pathlib import Path

from ocean_agent.document_chunker import chunk_documents
from ocean_agent.document_loader import load_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    documents = load_documents(
        PROJECT_ROOT / "documents", manifest_name="official_manifest.json"
    )
    chunks = chunk_documents(documents, max_chars=800)

    print(f"官方PDF数: {len(documents)}")
    print(f"总页数: {sum(len(document.pages) for document in documents)}")
    print(f"生成Chunk数: {len(chunks)}")
    for document in documents:
        own_chunks = [chunk for chunk in chunks if chunk.title == document.title]
        print("\n" + "=" * 72)
        print(f"文档: {document.title}")
        print(f"页数: {len(document.pages)}")
        print(f"Chunk数: {len(own_chunks)}")
        print(f"来源: {document.source.url}")
        if own_chunks:
            sample = own_chunks[0]
            print(f"首个Chunk: {sample.chunk_id}，PDF第{sample.page_number}页")
            print(sample.content[:240].replace("\n", " ") + "...")


if __name__ == "__main__":
    main()
