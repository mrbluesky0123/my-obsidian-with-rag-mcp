"""Query and search result schemas."""
from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field


class Query(BaseModel):
    """쿼리 모델"""
    text: str = Field(description="검색 쿼리 텍스트")
    top_k: int = Field(default=5, description="검색할 문서의 개수")
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        descrpition="필터 조건 (메타데이터 기반)"
    )

class SearchResult(BaseModel):
    """검색 결과 모델"""
    content: str = Field(description="검색된 청크 내용")
    score: float = Field(description="유사도 점수")
    document_id: str = Field(description="원본 문서 ID")
    chunk_id: str = Field(description="청크ID")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="메타데이터"
    )

class QueryState(BaseModel):
    """쿼리 그래프의 state"""
    query: Optional[Query] = Field(
        default=None,
        description="사용자 쿼리"
    )
    query_embeddings: Optional[List[float]] = Field(
        default=None,
        description="쿼리 임베딩"
    )
    retrieved_results: List[SearchResult] = Field(
        default_factory=list,
        description="검색된 결과들"
    )
    context: Optional[str] = Field(
        default=None,
        description="MCP로 반환할 최종 컨텍스트"
    )
    error: Optional[str] = Field(
        default=None,
        description="에러 메시지"
    )