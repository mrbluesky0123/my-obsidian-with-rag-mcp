import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv(verbose=True)


def test_gimini_connection():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # 간단한 텍스트
    test_text = "안녕하세요, 테스트입니다."
    result = embeddings.embed_query(test_text)

    print(f"임베딩 차원: {len(result)}")
    print("GEMINI 임베딩 연결")


if __name__ == "__main__":
    test_gimini_connection()
