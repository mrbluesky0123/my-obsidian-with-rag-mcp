import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from typing import List, Dict, Any, Literal
from src.embeddings.kosimcse_embeddings import KoSimCSEEmbeddings
from src.embeddings.ollama_embeddings import OllamaEmbeddings
from src.reranking.cross_encoder_reranker import CrossEncoderReranker

load_dotenv()


class VectorDB:
    """벡터 데이터베이스 관리 클래스"""

    def __init__(self,
                 persist_directory: str = "./chroma_db",
                 embedding_type: str = "ollama",
                 use_reranking: bool = False):
        """
        벡터DB 초기화

        Args:
            persist_directory: 벡터DB 저장 경로
            embedding_type: 사용할 임베딩 타입 ("google", "kosimcse", "ollama")
            use_reranking: Cross-encoder 리랭킹 사용 여부
        """
        self.persist_directory = persist_directory
        self.embedding_type = embedding_type
        self.use_reranking = use_reranking
        self.embeddings = self._create_embeddings()
        self.vectorstore = self._create_vectorstore()

        # 리랭커 초기화 (지연 로딩)
        self.reranker = None
        if use_reranking:
            self._init_reranker()

    def _create_embeddings(self):
        """임베딩 인스턴스 생성"""
        if self.embedding_type == "kosimcse":
            print("🇰🇷 한국어 특화 KoSimCSE 임베딩을 사용합니다")
            return KoSimCSEEmbeddings()
        elif self.embedding_type == "ollama":
            print("🤖 Ollama Qwen3-Embedding-8B 임베딩을 사용합니다")
            return OllamaEmbeddings()
        else:  # default: google
            print("🌍 Google Generative AI 임베딩을 사용합니다")
            return GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )

    def _create_vectorstore(self):
        return Chroma(
            persist_directory=self.persist_directory, embedding_function=self.embeddings
        )

    def _init_reranker(self):
        """리랭커 초기화"""
        try:
            self.reranker = CrossEncoderReranker()
        except Exception as e:
            print(f"⚠️ 리랭커 초기화 실패: {e}")
            self.use_reranking = False

    def add_documents(self, documents: List[Dict[str, Any]]):
        """문서 추가"""
        if not documents:
            return

        texts = [doc["content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]

        self.vectorstore.add_texts(texts, metadatas=metadatas)
        print(f"✅ {len(documents)}개 문서 벡터DB에 저장 완료!")

    def search(self, query: str, k: int = 5):
        """검색"""
        return self.vectorstore.similarity_search(query, k=k)

    def search_with_score(self, query: str, k: int = 5):
        """점수 포함 검색"""
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def search_with_reranking(self, query: str, k: int = 5, candidate_k: int = 20):
        """리랭킹 포함 검색"""
        if not self.use_reranking or self.reranker is None:
            print("⚠️ 리랭킹이 비활성화되어 있습니다. 일반 검색을 수행합니다.")
            return self.search(query, k)

        # 1단계: bi-encoder로 후보 추림
        print(f"🔍 1단계: bi-encoder로 상위 {candidate_k}개 후보 검색")
        candidates = self.search(query, k=candidate_k)

        if not candidates:
            return []

        # 2단계: cross-encoder로 리랭킹
        print(f"🎯 2단계: cross-encoder로 상위 {k}개 리랭킹")
        reranked = self.reranker.rerank(query, candidates, top_k=k)

        # 결과는 (Document, score) 형태로 반환
        return [doc for doc, score in reranked]

    def search_with_reranking_and_scores(self, query: str, k: int = 5, candidate_k: int = 20):
        """리랭킹 포함 검색 (점수도 함께 반환)"""
        if not self.use_reranking or self.reranker is None:
            print("⚠️ 리랭킹이 비활성화되어 있습니다. 일반 검색을 수행합니다.")
            return [(doc, score) for doc, score in self.search_with_score(query, k)]

        # 1단계: bi-encoder로 후보 추림
        print(f"🔍 1단계: bi-encoder로 상위 {candidate_k}개 후보 검색")
        candidates = self.search(query, k=candidate_k)

        if not candidates:
            return []

        # 2단계: cross-encoder로 리랭킹 (점수 포함)
        print(f"🎯 2단계: cross-encoder로 상위 {k}개 리랭킹")
        return self.reranker.rerank(query, candidates, top_k=k)
