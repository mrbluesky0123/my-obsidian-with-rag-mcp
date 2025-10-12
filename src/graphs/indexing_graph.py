"""Indexing workflow for processing Obsidian documents into vector store."""
from typing import Dict, Any

from langgraph.constants import END, START
from langgraph.graph import StateGraph

from src.graphs.nodes import obsidian_read_node, chunking_node, vector_store_node
from src.schemas.document import IndexingState


def create_indexing_graph():
    # 그래프 초기화
    graph = StateGraph(IndexingState)

    # 노드 추가
    graph.add_node("read", obsidian_read_node)
    graph.add_node("chunk", chunking_node)
    graph.add_node("store", vector_store_node)

    # 엣지 연결
    graph.add_edge(START, "read")
    graph.add_edge("read", "chunk")
    graph.add_edge("chunk", "store")
    graph.add_edge("store", END)

    return graph.compile()

def index_obsidian_vault(vault_path: str, config: Dict[str, Any] = None):
    """옵시디언 볼트를 인덱싱"""
    if config is None:
        config = {}

    config["vault_path"] = vault_path

    graph = create_indexing_graph()
    initial_state = IndexingState()

    # configurable로 전달
    result = graph.invoke(initial_state, config={"configurable": config})
    return result