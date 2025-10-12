"""Node for reading Obsidian documents."""
from langgraph.types import RunnableConfig

from src.obsidian.obsidian_loader import get_raw_documents
from src.schemas.document import IndexingState, Document


def obsidian_read_node(state: IndexingState, config: RunnableConfig) -> IndexingState:
    """옵시디언 문서를 읽어오는 노드"""
    try:
        # RunnableConfig에서 configurable 가져오기
        vault_path = config.get("configurable", {}).get("vault_path")
        if not vault_path:
            state.error = "vault_path not provided in config"
            return state

        raw_documents = get_raw_documents(vault_path)
        documents = []
        for doc in raw_documents:
            documents.append(
                Document(
                    id=doc["metadata"]["id"],
                    content=doc["content"],
                    metadata=doc["metadata"],
                    created_at=doc["metadata"].get("create_date"),
                )
            )
        state.documents = documents
        return state
    except Exception as e:
        state.error = f"Failed to obsidian read documents: {str(e)}"
        return state
