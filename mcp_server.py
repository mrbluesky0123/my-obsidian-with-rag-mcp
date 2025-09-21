#!/usr/bin/env python3
"""
옵시디언 RAG MCP 서버
Claude Code에서 옵시디언 노트를 검색하고 조회할 수 있게 해주는 MCP 서버
"""
import asyncio
import os
from pathlib import Path
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import ServerCapabilities
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import frontmatter

from src.vectorstore.vector_db import VectorDB
from src.obsidian.obsidian_loader import process_obsidian_vault, clean_text

# MCP 서버 생성
server = Server("obsidian-rag")

# 설정
VAULT_PATH = "/Users/mrbluesky/Documents/memo"  # 옵시디언 볼트 경로
# Claude Desktop 샌드박스를 위해 홈 디렉토리 사용
VECTORDB_PATH = os.path.expanduser("~/obsidian_vectordb")
# 임베딩 타입 설정 ("google" 또는 "kosimcse" 또는 "ollama")
EMBEDDING_TYPE = os.getenv("EMBEDDING_TYPE", "ollama")

# 벡터DB 인스턴스 (지연 로딩)
db = None

def ensure_vectordb():
    """벡터DB 초기화 (필요시)"""
    global db
    if db is None:
        db = VectorDB(VECTORDB_PATH, embedding_type=EMBEDDING_TYPE)
        
        # 벡터DB가 비어있으면 초기화
        try:
            test_results = db.search("test", k=1)
            if not test_results:
                print("벡터DB가 비어있습니다. 옵시디언 노트를 로딩중...")
                documents = process_obsidian_vault(VAULT_PATH)
                db.add_documents(documents)
                print("✅ 옵시디언 노트 로딩 완료!")
        except Exception as e:
            print(f"벡터DB 초기화 중: {e}")
    
    return db


@server.list_tools()
async def list_tools() -> list[Tool]:
    """사용 가능한 도구 목록"""
    return [
        Tool(
            name="search_obsidian_notes",
            description="옵시디언 노트에서 내용을 검색합니다. 업무 일정, 메모, 아이디어 등을 찾을 때 사용하세요.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "검색할 내용 (한글/영어 모두 지원)"
                    },
                    "limit": {
                        "type": "number", 
                        "description": "검색 결과 개수 (기본값: 5, 최대 10)",
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_obsidian_note",
            description="특정 옵시디언 노트의 전체 내용을 조회합니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string", 
                        "description": "노트 파일의 전체 경로"
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="list_recent_obsidian_notes",
            description="최근 수정된 옵시디언 노트 목록을 보여줍니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "number",
                        "description": "표시할 노트 개수 (기본값: 10)",
                        "minimum": 1,
                        "maximum": 20
                    }
                }
            }
        ),
        Tool(
            name="refresh_obsidian_vectordb",
            description="옵시디언 노트가 업데이트되었을 때 벡터DB를 새로고침합니다.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """도구 실행"""
    
    if name == "search_obsidian_notes":
        try:
            db_instance = ensure_vectordb()
            query = arguments["query"]
            limit = min(arguments.get("limit", 5), 10)
            
            results = db_instance.search(query, k=limit)
            
            if not results:
                response = f"'{query}'에 대한 검색 결과가 없습니다."
            else:
                response = f"🔍 '{query}' 검색 결과 ({len(results)}개):\n\n"
                
                for i, doc in enumerate(results):
                    meta = doc.metadata
                    title = meta.get('title', '제목 없음')
                    source = meta.get('source', '경로 없음')
                    chunk_info = f"({meta.get('chunk_index', 0)+1}/{meta.get('total_chunks', 1)} 청크)"
                    
                    response += f"**{i+1}. {title}** {chunk_info}\n"
                    response += f"📁 `{source}`\n"
                    response += f"📄 {doc.page_content[:300]}{'...' if len(doc.page_content) > 300 else ''}\n\n"
                    response += "---\n\n"
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"❌ 검색 중 오류 발생: {str(e)}")]
    
    elif name == "get_obsidian_note":
        try:
            file_path = arguments["file_path"]
            
            if not os.path.exists(file_path):
                return [TextContent(type="text", text=f"❌ 파일을 찾을 수 없습니다: {file_path}")]
            
            with open(file_path, 'r', encoding='utf-8') as f:
                post = frontmatter.load(f)
            
            response = f"# {post.metadata.get('title', Path(file_path).stem)}\n\n"
            
            # 메타데이터 표시
            if post.metadata:
                response += "## 메타데이터\n"
                for key, value in post.metadata.items():
                    response += f"- **{key}**: {value}\n"
                response += "\n"
            
            response += "## 내용\n\n"
            response += clean_text(post.content)
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"❌ 파일 읽기 실패: {str(e)}")]
    
    elif name == "list_recent_obsidian_notes":
        try:
            limit = min(arguments.get("limit", 10), 20)
            vault_path = Path(VAULT_PATH)
            
            # .md 파일들을 수정 시간 순으로 정렬
            md_files = list(vault_path.rglob("*.md"))
            md_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            response = f"📚 최근 수정된 옵시디언 노트 (최대 {limit}개):\n\n"
            
            for i, file_path in enumerate(md_files[:limit]):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        post = frontmatter.load(f)
                    
                    title = post.metadata.get('title', file_path.stem)
                    modified_time = file_path.stat().st_mtime
                    from datetime import datetime
                    modified_str = datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d %H:%M')
                    
                    response += f"**{i+1}. {title}**\n"
                    response += f"📁 `{file_path}`\n" 
                    response += f"🕒 {modified_str}\n"
                    
                    # 첫 몇 줄 미리보기
                    preview = clean_text(post.content)[:150]
                    response += f"📄 {preview}{'...' if len(post.content) > 150 else ''}\n\n"
                    
                except Exception:
                    continue
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"❌ 노트 목록 조회 실패: {str(e)}")]
    
    elif name == "refresh_obsidian_vectordb":
        try:
            global db
            
            print("🔄 벡터DB 새로고침 시작...")
            
            # 기존 벡터DB 삭제
            import shutil
            if os.path.exists(VECTORDB_PATH):
                shutil.rmtree(VECTORDB_PATH)
            
            # 새로운 벡터DB 생성
            db = VectorDB(VECTORDB_PATH, embedding_type=EMBEDDING_TYPE)
            documents = process_obsidian_vault(VAULT_PATH)
            db.add_documents(documents)
            
            response = f"✅ 벡터DB 새로고침 완료!\n"
            response += f"📊 총 {len(documents)}개 문서 청크가 업데이트되었습니다."
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"❌ 벡터DB 새로고침 실패: {str(e)}")]
    
    return [TextContent(type="text", text=f"❌ 알 수 없는 도구: {name}")]


async def main():
    """MCP 서버 실행"""
    try:
        # 초기화
        print("🚀 옵시디언 RAG MCP 서버 시작 중...")
        ensure_vectordb()
        print("✅ 초기화 완료!")
        
        # 서버 실행
        async with stdio_server() as streams:
            await server.run(
                streams[0], streams[1], 
                InitializationOptions(
                    server_name="obsidian-rag",
                    server_version="1.0.0",
                    capabilities=ServerCapabilities(
                        tools={}
                    )
                )
            )
    except Exception as e:
        print(f"❌ MCP 서버 오류: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())