from src.obsidian.obsidian_loader import process_obsidian_vault
from src.vectorstore.vector_db import VectorDB


def main():
    vault_path = '/Users/mrbluesky/Documents/memo'

    # 함수로 파싱
    documents = process_obsidian_vault(vault_path)

    # 클래스로 벡터DB 관리
    db = VectorDB("./obsidian_vectordb")
    db.add_documents(documents)

    # 검색
    while True:
        query = input("\n검색어 (q=종료): ")
        if query.lower() == "q":
            break

        results = db.search(query, k=3)

        print("\n🔍 검색 결과:")
        for i, doc in enumerate(results):
            meta = doc.metadata
            print(
                f"\n{i + 1}. [{meta.get('title')}] ({meta.get('chunk_index') + 1}/{meta.get('total_chunks')})"
            )
            print(f"   {doc.page_content[:200]}...")


if __name__ == "__main__":
    main()
