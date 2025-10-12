"""Document and chunk schemas."""
from datetime import datetime
from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str = Field(description="문서 고유의 ID(파일 경로 기반)")
    content: str = Field(description="문서 전체의 내용")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="문서의 메타데이터(파일명, 경로, 태그 등)"
    )
    created_at: Optional[str] = Field(
        default=None,
        description="문서 생성 시간"
    )

class Chunk(BaseModel):
    id: str = Field(description="청크 고유 ID")
    content: str = Field(description="청크 내용")
    embedding: Optional[List[float]] = Field(
        default=None,
        description="임베딩 벡터"
    )
    document_id: str = Field(description="원본 문서 ID")
    chunk_index: int = Field(description="문서 내 청크 순서")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="청크 메타데이터"
    )

class IndexingState(BaseModel):
    documents: List[Document] = Field(
        default_factory=list,
        description="읽어온 문서들"
    )
    chunks: List[Chunk] = Field(
        default_factory=list,
        description="분할된 청크들"
    )
    error: Optional[str] = Field(
        default=None,
        description="에러 메시지"
    )