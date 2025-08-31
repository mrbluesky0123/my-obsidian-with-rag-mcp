# 🧠 Obsidian RAG with MCP

옵시디언 노트와 RAG(Retrieval Augmented Generation) 기술을 결합하여 Claude Code에서 개인 메모를 활용할 수 있는 MCP 서버입니다.

## ✨ 주요 기능

- **🔍 의미 기반 검색**: GEMINI 임베딩을 사용한 벡터 검색
- **📚 옵시디언 통합**: 마크다운 파일 자동 파싱 및 메타데이터 추출
- **🤖 Claude MCP 연동**: Claude Desktop/Code에서 직접 사용 가능
- **⚡ 실시간 업데이트**: 벡터DB 새로고침 기능
- **🌏 한글 지원**: 한글 문서 완벽 지원

## 🛠 기술 스택

- **임베딩**: Google GEMINI Embedding
- **벡터DB**: ChromaDB (SQLite + HNSW)
- **프레임워크**: LangChain
- **패키지 관리**: uv
- **프로토콜**: MCP (Model Context Protocol)

## 📦 설치

### 1. 프로젝트 클론
```bash
git clone https://github.com/your-username/obsidian-rag-mcp.git
cd obsidian-rag-mcp
```

### 2. 의존성 설치
```bash
# uv 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 패키지 설치
uv sync
```

### 3. 환경 변수 설정
```bash
# .env 파일 생성
echo "GOOGLE_API_KEY=your_gemini_api_key_here" > .env
```

> 🔑 **GEMINI API 키 발급**: [Google AI Studio](https://makersuite.google.com/app/apikey)에서 발급

### 4. 옵시디언 경로 설정
`mcp_server.py`에서 본인의 옵시디언 볼트 경로로 수정:
```python
VAULT_PATH = "/Users/your-username/Documents/your-obsidian-vault"
```

## 🚀 사용법

### MCP 서버 단독 실행 (테스트)
```bash
uv run python mcp_server.py
```

### Claude Desktop 연동

1. **Claude Desktop Settings**에서 MCP Servers 추가
2. 아래 설정 사용:
```json
{
  "mcpServers": {
    "obsidian-rag": {
      "command": "/path/to/project/.venv/bin/python",
      "args": ["/path/to/project/mcp_server.py"],
      "cwd": "/path/to/project"
    }
  }
}
```

### 사용 예시
Claude Desktop에서:
- **"옵시디언에서 '프로젝트' 관련 내용 찾아서 설명해줘"**
- **"최근 작성한 노트 목록 보여줘"**
- **"내 메모를 바탕으로 오늘 할 일을 정리해줘"**

## 🔧 MCP 도구 목록

### 1. `search_obsidian_notes`
- **기능**: 의미 기반 노트 검색
- **파라미터**: `query` (검색어), `limit` (결과 수)
- **예시**: "랭체인 사용법"

### 2. `get_obsidian_note` 
- **기능**: 특정 노트 전체 내용 조회
- **파라미터**: `file_path` (노트 경로)

### 3. `list_recent_obsidian_notes`
- **기능**: 최근 수정 노트 목록
- **파라미터**: `limit` (목록 수)

### 4. `refresh_obsidian_vectordb`
- **기능**: 벡터DB 새로고침
- **용도**: 새 노트 추가 후 업데이트

## 🏗 프로젝트 구조

```
obsidian-rag-mcp/
├── src/
│   ├── obsidian/
│   │   └── obsidian_loader.py    # 옵시디언 파싱
│   └── vectorstore/
│       └── vector_db.py          # 벡터DB 관리
├── mcp_server.py                 # MCP 서버 메인
├── mcp_config.json              # MCP 설정 파일
├── test_mcp.py                  # 테스트 스크립트
├── .env.example                 # 환경변수 예시
└── README.md
```

## ⚙️ 설정 커스터마이징

### 청킹 설정
```python
# obsidian_loader.py
def create_text_splitter(chunk_size=1000, chunk_overlap=200):
    # chunk_size: 청크 크기
    # chunk_overlap: 중복 구간
```

### 검색 결과 수 조정
```python
# mcp_server.py - search_obsidian_notes
limit = min(arguments.get("limit", 5), 10)  # 최대 10개
```

## 🐛 문제 해결

### 1. "ModuleNotFoundError: No module named 'mcp'"
```bash
# 가상환경 Python 직접 사용
/path/to/.venv/bin/python mcp_server.py
```

### 2. "Read-only file system" 오류
```bash
# 벡터DB 경로를 홈 디렉토리로 변경
VECTORDB_PATH = os.path.expanduser("~/obsidian_vectordb")
```

### 3. UTF-8 인코딩 오류
- 자동으로 서로게이트 문자 제거됨
- 문제 지속 시 `clean_text()` 함수 확인

## 📈 성능 최적화

- **첫 실행**: 벡터DB 구축으로 시간 소요 (노트 수에 비례)
- **이후 검색**: 밀리초 단위 고속 검색
- **메모리 사용량**: 약 768차원 × 노트 수 × 4바이트

## 🤝 기여

1. Fork the Project
2. Create Feature Branch (`git checkout -b feature/amazing-feature`)
3. Commit Changes (`git commit -m 'Add amazing feature'`)
4. Push to Branch (`git push origin feature/amazing-feature`)
5. Open Pull Request
---
⭐ 이 프로젝트가 유용하다면 Star를 눌러주세요!