import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()


def test_chroma():
    # 임베딩 모델
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # 벡터 스토어 생성
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    test_docs = [
        "오늘은 랭체인과 크로마를 연동하는 작업을 했다",
        "GEMINI 임베딩 모델 성능이 좋은 것 같다",
        "uv 패키지 매니저를 처음 써봤는데 편하다",
    ]
    vectorstore.add_texts(test_docs)

    query = "랭체인 작업"
    results = vectorstore.similarity_search(query, k=2)
    print("🔍 검색 결과:")
    for i, doc in enumerate(results):
        print(f"{i + 1}.{doc.page_content}")

    print("✅ Chroma 연결 및 검색 성공!")


if __name__ == "__main__":
    test_chroma()
