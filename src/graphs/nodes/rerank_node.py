"""Node for reranking retrieved documents using cross-encoder."""
# from typing import Dict, Any
#
# from src.schemas.query import QueryState
#
#
# def rerank_node(state: QueryState, config: Dict[str, Any]) -> QueryState:
#     try:
#         if state.error or not state.query or not state.retrieved_results:
#             return state
#
#     model_name = config.get("rerank_model", "")
