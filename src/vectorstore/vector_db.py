import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from typing import List, Dict, Any

load_dotenv()


class VectorDB:
    """벡터 데이터베이스 관리 클래스"""

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = self._create_embeddings()
        self.vectorstore = self._create_vectorstore()

    def _create_embeddings(self):
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY")
        )

    def _create_vectorstore(self):
        return Chroma(
            persist_directory=self.persist_directory, embedding_function=self.embeddings
        )

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
