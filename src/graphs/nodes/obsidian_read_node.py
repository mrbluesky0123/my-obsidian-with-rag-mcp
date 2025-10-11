"""Node for reading Obsidian documents."""
from typing import Any, Dict

from src.obsidian.obsidian_loader import process_obsidian_vault
from src.schemas.document import IndexingState, Document


def obsidian_read_node(
        state: IndexingState,
        config: Dict[str, Any],
) -> IndexingState:
    """옵시디언 문서를 읽어오는 노드"""
    try:
        vault_path = config.get("vault_path")
        if not vault_path:
            state.error = "valut_path not provided in config"
            return state

        raw_documents = process_obsidian_vault(vault_path)
        documents = []
        for doc in raw_documents:
            documents.append(
                Document(
                    id=doc.get()
                )
            )