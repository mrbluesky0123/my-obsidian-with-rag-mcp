"""Node for retrieving similar documents from vector store."""
import os
from langgraph.types import RunnableConfig

from src.schemas.query import QueryState, SearchResult
from src.vectorstore.vector_db import VectorDB


def retrieval_node(state: QueryState, config: RunnableConfig) -> QueryState:
    """벡터 DB에서 유사 문서를 검색하는 노드"""
    try:
        if state.error or not state.query:
            return state

        # 설정
        db_path = config.get("configurable", {}).get("db_path", "./obsidian_vectordb")
        embedding_type = os.getenv("EMBEDDING_TYPE", "ollama")
        top_k = state.query.top_k

        # 기존 VectorDB 사용
        vector_db = VectorDB(
            persist_directory=db_path,
            embedding_type=embedding_type,
            use_reranking=False  # retrieval 단계에선 리랭킹 안함
        )

        # 검색 (점수 포함)
        results = vector_db.search_with_score(state.query.text, k=top_k)

        # SearchResult 스키마로 변환
        search_results = []
        for doc, score in results:
            search_result = SearchResult(
                content=doc.page_content,
                score=float(1 - score),  # distance → similarity 변환
                document_id=doc.metadata.get("id", ""),
                chunk_id=doc.metadata.get("chunk_id", ""),
                metadata=doc.metadata
            )
            search_results.append(search_result)

        state.retrieved_results = search_results
        return state

    except Exception as e:
        state.error = f"Failed to retrieve documents: {str(e)}"
        return state