"""
KoSimCSE 기반 한국어 문장 임베딩 클래스
BM-K/KoSimCSE-roberta 모델을 LangChain Embeddings 인터페이스에 맞게 래핑
"""
import torch
from typing import List
from transformers import AutoModel, AutoTokenizer
from langchain_core.embeddings import Embeddings


class KoSimCSEEmbeddings(Embeddings):
    """KoSimCSE 모델을 사용하는 한국어 임베딩 클래스"""

    def __init__(self, model_name: str = "BM-K/KoSimCSE-roberta", device: str = None):
        """
        KoSimCSE 임베딩 초기화

        Args:
            model_name: 사용할 모델명 (기본: BM-K/KoSimCSE-roberta)
            device: 사용할 디바이스 (기본: auto-detect)
        """
        self.model_name = model_name

        # 디바이스 설정
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"🤖 KoSimCSE 모델 로딩 중... (device: {self.device})")

        # 모델과 토크나이저 로드
        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # 모델을 디바이스로 이동
        self.model.to(self.device)
        self.model.eval()  # 평가 모드로 설정

        print("✅ KoSimCSE 모델 로딩 완료!")

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트 리스트를 임베딩 벡터로 변환

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            임베딩 벡터 리스트 (각 벡터는 768차원)
        """
        if not texts:
            return []

        # 토큰화
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512  # KoSimCSE 권장 최대 길이
        )

        # 입력을 디바이스로 이동
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 임베딩 추출 (gradient 계산 비활성화)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # [CLS] 토큰의 임베딩을 사용 (첫 번째 토큰)
            embeddings = outputs.last_hidden_state[:, 0, :]

        # CPU로 이동하고 리스트로 변환
        embeddings = embeddings.cpu().numpy().tolist()

        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        문서 리스트를 임베딩으로 변환 (LangChain 인터페이스)

        Args:
            texts: 임베딩할 문서 텍스트 리스트

        Returns:
            임베딩 벡터 리스트
        """
        return self._get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        쿼리 텍스트를 임베딩으로 변환 (LangChain 인터페이스)

        Args:
            text: 임베딩할 쿼리 텍스트

        Returns:
            임베딩 벡터
        """
        embeddings = self._get_embeddings([text])
        return embeddings[0] if embeddings else []

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        두 텍스트 간의 코사인 유사도 계산 (참조 코드의 cal_score 기능)

        Args:
            text1: 첫 번째 텍스트
            text2: 두 번째 텍스트

        Returns:
            유사도 점수 (0-100 범위)
        """
        embeddings = self._get_embeddings([text1, text2])
        if len(embeddings) != 2:
            return 0.0

        # 텐서로 변환
        a = torch.tensor(embeddings[0])
        b = torch.tensor(embeddings[1])

        # 정규화
        a_norm = a / a.norm()
        b_norm = b / b.norm()

        # 코사인 유사도 계산 (0-100 스케일)
        similarity = torch.dot(a_norm, b_norm) * 100

        return similarity.item()


def test_kosimcse():
    """KoSimCSE 임베딩 테스트 함수"""
    print("🧪 KoSimCSE 임베딩 테스트 시작...")

    # 임베딩 인스턴스 생성
    embeddings = KoSimCSEEmbeddings()

    # 테스트 문장들 (참조 코드와 동일)
    sentences = [
        '치타가 들판을 가로 질러 먹이를 쫓는다.',
        '치타 한 마리가 먹이 뒤에서 달리고 있다.',
        '원숭이 한 마리가 드럼을 연주한다.'
    ]

    print(f"📝 테스트 문장: {sentences}")

    # 임베딩 벡터 생성
    doc_embeddings = embeddings.embed_documents(sentences)
    print(f"📊 임베딩 차원: {len(doc_embeddings[0])}")

    # 유사도 계산
    score01 = embeddings.calculate_similarity(sentences[0], sentences[1])
    score02 = embeddings.calculate_similarity(sentences[0], sentences[2])

    print(f"🔍 유사도 결과:")
    print(f"  '{sentences[0]}' vs '{sentences[1]}': {score01:.2f}")
    print(f"  '{sentences[0]}' vs '{sentences[2]}': {score02:.2f}")

    print("✅ 테스트 완료!")


if __name__ == "__main__":
    test_kosimcse()