"""Node for storing embeddings in vector database."""
import os
from langgraph.types import RunnableConfig

from src.schemas.document import IndexingState
from src.vectorstore.vector_db import VectorDB


def vector_store_node(state: IndexingState, config: RunnableConfig) -> IndexingState:
    try:
        if state.error:
            return state

        # 설정
        db_path = config.get("configurable", {}).get("db_path", "./obsidian_vectordb")
        embedding_type = os.getenv("EMBEDDING_TYPE", "ollama")
        use_reranking = config.get("configurable", {}).get("use_reranking", False)

        vector_db = VectorDB(
            persist_directory=db_path,
            embedding_type=embedding_type,
            use_reranking=use_reranking,
        )

        # chunk 스키마 → 딕셔너리 변환
        documents = [
            {
                "content": chunk.content,
                "metadata": chunk.metadata,
            }
            for chunk in state.chunks
        ]

        vector_db.add_documents(documents)

        return state

    except Exception as e:
        state.error = f"Failed to store in vector db: {str(e)}"
        return state