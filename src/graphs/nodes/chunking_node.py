"""Node for chunking documents."""
from langgraph.types import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.schemas.document import IndexingState, Chunk

from src.logging.logger_factory import LoggerFactory
logger = LoggerFactory.get_logger("obsidian_rag.ollama_embeddings")

def chunking_node(state: IndexingState, config: RunnableConfig) -> IndexingState:
    logger.info(f"{'*' * 50}")
    logger.info("CHUNKING_NODE")
    logger.info(f"{'*' * 50}")

    try:
        if state.error:
            return state

        # 청크 분할
        chunk_size = config.get("configurable", {}).get("chunk_size", 500)
        chunk_overlap = config.get("configurable", {}).get("chunk_overlap", 50)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        chunks = []
        for doc in state.documents:
            text_chunks = splitter.split_text(doc.content)
            total_chunks = len(text_chunks)

            for idx, chunk_text in enumerate(text_chunks):
                chunk_id = f"{doc.id}#chunk_{idx}"
                chunk_metadata = {**doc.metadata,
                                  "chunk_id": chunk_id,
                                  "chunk_index": idx,
                                  "total_chunks": total_chunks}
                chunk = Chunk(
                    id= chunk_id,
                    content=chunk_text,
                    document_id=doc.id,
                    chunk_index=idx,
                    metadata=chunk_metadata
                )
                chunks.append(chunk)

        state.chunks = chunks
        return state

    except Exception as e:
        state.error = f"Failed to obsidian chunk documents: {str(e)}"
        return state