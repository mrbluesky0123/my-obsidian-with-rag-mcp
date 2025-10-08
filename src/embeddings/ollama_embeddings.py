import requests
from typing import List
from langchain.embeddings.base import Embeddings
from src.logging.logger_factory import LoggerFactory

logger = LoggerFactory.get_logger("obsidian_rag.ollama_embeddings")


class OllamaEmbeddings(Embeddings):
    """Ollama 임베딩 클래스"""

    def __init__(self, model_name: str = "hf.co/Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        logger.info(f"🤖 Ollama 임베딩 초기화: {model_name}, URL: {base_url}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서들을 임베딩"""
        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.model_name,
                    "input": text
                }
            )
            if response.status_code == 200:
                embedding = response.json()["embeddings"][0]
                embeddings.append(embedding)
            else:
                raise Exception(f"Ollama 임베딩 실패: {response.status_code}, {response.text}")

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """쿼리를 임베딩"""
        response = requests.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.model_name,
                "input": text
            }
        )
        if response.status_code == 200:
            return response.json()["embeddings"][0]
        else:
            raise Exception(f"Ollama 쿼리 임베딩 실패: {response.status_code}, {response.text}")