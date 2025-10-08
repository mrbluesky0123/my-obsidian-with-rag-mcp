import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from typing import List, Dict, Any, Literal
from src.embeddings.kosimcse_embeddings import KoSimCSEEmbeddings
from src.embeddings.ollama_embeddings import OllamaEmbeddings
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.logging.logger_factory import LoggerFactory

load_dotenv()

logger = LoggerFactory.get_logger("obsidian_rag.vector_db")


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
            logger.info("🇰🇷 한국어 특화 KoSimCSE 임베딩을 사용합니다")
            return KoSimCSEEmbeddings()
        elif self.embedding_type == "ollama":
            logger.info("🤖 Ollama Qwen3-Embedding-8B 임베딩을 사용합니다")
            return OllamaEmbeddings()
        else:  # default: google
            logger.info("🌍 Google Generative AI 임베딩을 사용합니다")
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
            logger.info("Cross-encoder 리랭커 초기화 중...")
            self.reranker = CrossEncoderReranker()
            logger.info("✅ 리랭커 초기화 완료")
        except Exception as e:
            logger.warning(f"⚠️ 리랭커 초기화 실패: {e}")
            self.use_reranking = False

    def add_documents(self, documents: List[Dict[str, Any]]):
        """문서 추가"""
        if not documents:
            logger.warning("추가할 문서가 없습니다")
            return

        logger.info(f"📝 {len(documents)}개 문서를 벡터DB에 추가 중...")
        texts = [doc["content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]

        try:
            self.vectorstore.add_texts(texts, metadatas=metadatas)
            logger.info(f"✅ {len(documents)}개 문서 벡터DB에 저장 완료!")
        except Exception as e:
            logger.error(f"문서 저장 실패: {e}", exc_info=True)
            raise

    def search(self, query: str, k: int = 5):
        """검색"""
        logger.debug(f"🔍 검색 실행: '{query}' (결과 수: {k})")
        results = self.vectorstore.similarity_search(query, k=k)
        logger.info(f"✅ 검색 완료: {len(results)}개 결과 반환")

        # 결과 상세 로깅
        for i, doc in enumerate(results):
            title = doc.metadata.get('title', '제목없음')
            logger.debug(f"  {i+1}. {title} (내용 길이: {len(doc.page_content)})")

        return results

    def search_with_score(self, query: str, k: int = 5):
        """점수 포함 검색"""
        logger.debug(f"🔍 점수 포함 검색 실행: '{query}' (결과 수: {k})")
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        logger.info(f"✅ 점수 포함 검색 완료: {len(results)}개 결과 반환")

        # 결과 상세 로깅 (점수 포함)
        for i, (doc, score) in enumerate(results):
            title = doc.metadata.get('title', '제목없음')
            has_keyword = query in doc.page_content
            logger.debug(f"  {i+1}. {title} (점수: {score:.4f}, 키워드 포함: {has_keyword})")

        return results

    def search_with_reranking(self, query: str, k: int = 5, candidate_k: int = 20):
        """리랭킹 포함 검색"""
        if not self.use_reranking or self.reranker is None:
            logger.warning("⚠️ 리랭킹이 비활성화되어 있습니다. 일반 검색을 수행합니다.")
            return self.search(query, k)

        logger.info(f"🔍 하이브리드 검색 시작: '{query}' (후보: {candidate_k}, 최종: {k})")

        # 1단계: bi-encoder로 후보 추림
        logger.debug(f"🔍 1단계: bi-encoder로 상위 {candidate_k}개 후보 검색")
        candidates = self.search(query, k=candidate_k)

        if not candidates:
            logger.info("검색 결과가 없습니다")
            return []

        # 2단계: cross-encoder로 리랭킹
        logger.debug(f"🎯 2단계: cross-encoder로 상위 {k}개 리랭킹")
        reranked = self.reranker.rerank(query, candidates, top_k=k)

        logger.info(f"✅ 하이브리드 검색 완료: {len(reranked)}개 결과")

        # 리랭킹 결과 상세 로깅
        for i, (doc, score) in enumerate(reranked):
            title = doc.metadata.get('title', '제목없음')
            has_keyword = query in doc.page_content
            logger.debug(f"  리랭킹 {i+1}. {title} (크로스 점수: {score:.4f}, 키워드 포함: {has_keyword})")

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
