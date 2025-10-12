#!/usr/bin/env python3
"""LangGraph 플로우 테스트"""
from src.graphs.indexing_graph import index_obsidian_vault
from src.graphs.query_graph import query_obsidian


def test_indexing():
    """인덱싱 플로우 테스트"""
    print("=" * 60)
    print("📚 인덱싱 플로우 테스트 시작")
    print("=" * 60)

    result = index_obsidian_vault(
        vault_path="/Users/mrbluesky/Documents/memo",
        config={
            "db_path": "./obsidian_vectordb",
            "chunk_size": 1000,
            "chunk_overlap": 200
        }
    )

    if result.get("error"):
        print(f"❌ 에러 발생: {result['error']}")
        return False
    else:
        print(f"✅ 문서 읽기 완료: {len(result['documents'])}개")
        print(f"✅ 청크 생성 완료: {len(result['chunks'])}개")
        print(f"✅ 벡터DB 저장 완료")
        return True


def test_query():
    """쿼리 플로우 테스트"""
    print("\n" + "=" * 60)
    print("🔍 쿼리 플로우 테스트 시작")
    print("=" * 60)

    query_text = input("\n검색어를 입력하세요 (엔터=기본값 'langgraph'): ").strip()
    if not query_text:
        query_text = "langgraph"

    print(f"\n검색어: '{query_text}'")

    result = query_obsidian(
        query_text=query_text,
        top_k=5,
        config={
            "db_path": "./obsidian_vectordb",
            "max_context_results": 3
        }
    )

    if result.get("error"):
        print(f"❌ 에러 발생: {result['error']}")
        return False
    else:
        print(f"✅ 검색 완료: {len(result['retrieved_results'])}개 결과")
        print("\n" + "=" * 60)
        print("📄 컨텍스트:")
        print("=" * 60)
        print(result['context'])
        return True


def main():
    """메인 함수"""
    print("\n🚀 LangGraph RAG 플로우 테스트\n")

    # 1. 인덱싱 테스트
    indexing_success = test_indexing()

    if not indexing_success:
        print("\n❌ 인덱싱 실패. 쿼리 테스트를 건너뜁니다.")
        return

    # 2. 쿼리 테스트
    test_query()

    # 3. 추가 쿼리 (선택)
    while True:
        again = input("\n다시 검색하시겠습니까? (y/n): ").strip().lower()
        if again == 'y':
            test_query()
        else:
            break

    print("\n✅ 테스트 완료!")


if __name__ == "__main__":
    main()
