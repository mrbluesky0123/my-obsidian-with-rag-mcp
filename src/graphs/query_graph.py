"""Query workflow for retrieving and reranking documents."""
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from src.schemas.query import QueryState, Query
from src.graphs.nodes.retrieval_node import retrieval_node
from src.graphs.nodes.context_builder_node import context_builder_node


def create_query_graph():
    """쿼리 그래프 생성"""

    # 그래프 초기화
    graph = StateGraph(QueryState)

    # 노드 추가
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("build_context", context_builder_node)

    # 엣지 연결
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "build_context")
    graph.add_edge("build_context", END)

    return graph.compile()


# 편의 함수
def query_obsidian(query_text: str, top_k: int = 5, config: Dict[str, Any] = None):
    """옵시디언 노트 검색"""
    if config is None:
        config = {}

    graph = create_query_graph()
    initial_state = QueryState(
        query=Query(text=query_text, top_k=top_k)
    )

    result = graph.invoke(initial_state, config={"configurable": config})
    return result
