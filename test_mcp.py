#!/usr/bin/env python3
"""
MCP 서버 테스트 스크립트
실제 MCP 클라이언트 없이도 도구들이 잘 동작하는지 테스트
"""
import asyncio
from mcp_server import ensure_vectordb, call_tool

async def test_tools():
    """MCP 도구들 테스트"""
    
    print("🧪 MCP 도구 테스트 시작...\n")
    
    # 벡터DB 초기화
    print("📊 벡터DB 초기화...")
    try:
        ensure_vectordb()
        print("✅ 벡터DB 준비 완료!\n")
    except Exception as e:
        print(f"❌ 벡터DB 초기화 실패: {e}")
        return
    
    # 테스트 케이스들
    test_cases = [
        {
            "name": "search_obsidian_notes",
            "args": {"query": "랭체인", "limit": 3},
            "description": "옵시디언에서 '랭체인' 검색"
        },
        {
            "name": "list_recent_obsidian_notes", 
            "args": {"limit": 5},
            "description": "최근 노트 5개 목록"
        },
        {
            "name": "search_obsidian_notes",
            "args": {"query": "구현", "limit": 2}, 
            "description": "옵시디언에서 '구현' 검색"
        }
    ]
    
    # 각 테스트 실행
    for i, test in enumerate(test_cases, 1):
        print(f"🔍 테스트 {i}: {test['description']}")
        print(f"도구: {test['name']}")
        print(f"인자: {test['args']}")
        print("-" * 50)
        
        try:
            results = await call_tool(test['name'], test['args'])
            for result in results:
                print(result.text[:500] + ("..." if len(result.text) > 500 else ""))
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
        
        print("\n" + "="*60 + "\n")
    
    print("🎉 MCP 도구 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(test_tools())