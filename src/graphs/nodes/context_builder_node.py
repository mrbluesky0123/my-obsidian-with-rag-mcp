"""Node for building context from retrieved documents for MCP response."""
from langgraph.types import RunnableConfig

from src.schemas.query import QueryState
from src.logging.logger_factory import LoggerFactory
logger = LoggerFactory.get_logger("obsidian_rag.ollama_embeddings")

def context_builder_node(state: QueryState, config: RunnableConfig) -> QueryState:
    """MCP 응답용 컨텍스트를 생성하는 노드"""
    logger.info(f"{'*' * 50}")
    logger.info("CONTEXT_BUILDER_NODE")
    logger.info(f"{'*' * 50}")

    try:
        if state.error or not state.retrieved_results:
            return state

        max_results = config.get("configurable", {}).get("max_context_results", 5)

        # 상위 N개 결과만 사용
        top_results = state.retrieved_results[:max_results]

        # 컨텍스트 포맷팅
        context_parts = []
        for idx, result in enumerate(top_results, 1):
            context_parts.append(
                f"[문서 {idx}] (점수: {result.score:.3f})\n"
                f"출처: {result.metadata.get('source', 'unknown')}\n"
                f"제목: {result.metadata.get('title', 'unknown')}\n"
                f"내용:\n{result.content}\n"
            )

        state.context = "\n" + "="*60 + "\n\n".join(context_parts)
        return state

    except Exception as e:
        state.error = f"Failed to build context: {str(e)}"
        return state
