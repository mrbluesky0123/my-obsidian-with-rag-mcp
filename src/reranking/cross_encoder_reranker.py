from typing import List, Tuple
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document


class CrossEncoderReranker:
    """Cross-encoder 기반 리랭킹 클래스"""

    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask"):
        """
        Cross-encoder 리랭커 초기화

        Args:
            model_name: 사용할 cross-encoder 모델명
        """
        self.model_name = model_name
        print(f"🎯 Cross-encoder 리랭커 로딩: {model_name}")
        self.cross_encoder = CrossEncoder(model_name)
        print("✅ Cross-encoder 로딩 완료!")

    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Cross-encoder로 문서들을 리랭킹

        Args:
            query: 검색 쿼리
            documents: bi-encoder로 검색된 후보 문서들
            top_k: 최종 반환할 문서 개수

        Returns:
            (문서, 점수) 튜플의 리스트 (점수 내림차순)
        """
        if not documents:
            return []

        print(f"🔄 {len(documents)}개 문서를 Cross-encoder로 리랭킹 중...")

        # 쿼리-문서 쌍 생성
        query_doc_pairs = []
        for doc in documents:
            # 문서 내용이 너무 길면 잘라내기 (cross-encoder 입력 길이 제한)
            content = doc.page_content[:512] if len(doc.page_content) > 512 else doc.page_content
            query_doc_pairs.append([query, content])

        # Cross-encoder로 점수 계산
        scores = self.cross_encoder.predict(query_doc_pairs)

        # 문서와 점수를 함께 묶어서 점수 내림차순 정렬
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # 상위 k개만 반환
        top_results = doc_scores[:top_k]

        print(f"✅ 리랭킹 완료! 상위 {len(top_results)}개 문서 반환")
        return top_results

    def rerank_with_details(self, query: str, documents: List[Document], top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        상세 정보와 함께 리랭킹 (디버깅용)
        """
        results = self.rerank(query, documents, top_k)

        print(f"\n📊 리랭킹 결과 상세:")
        for i, (doc, score) in enumerate(results):
            title = doc.metadata.get('title', '제목없음')
            print(f"{i+1}. {title} (점수: {score:.4f})")
            print(f"   내용: {doc.page_content[:100]}...")

        return results