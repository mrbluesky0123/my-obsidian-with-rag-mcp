import frontmatter
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import re


def clean_text(text: str) -> str:
    """서로게이트 에러 완전 해결하는 텍스트 정리"""
    if not text:
        return ""
    
    try:
        # 1단계: 문자 단위로 서로게이트 제거
        cleaned_chars = []
        for char in text:
            try:
                # 서로게이트 범위 확인 (0xD800-0xDFFF)
                code_point = ord(char)
                if 0xD800 <= code_point <= 0xDFFF:
                    # 서로게이트 문자는 스킵
                    continue
                    
                # UTF-8로 인코딩/디코딩 가능한지 테스트
                char.encode('utf-8').decode('utf-8')
                cleaned_chars.append(char)
                
            except (UnicodeError, ValueError):
                # 문제 있는 문자는 스킵
                continue
        
        cleaned_text = ''.join(cleaned_chars)
        
        # 2단계: 전체 텍스트 다시 정리
        cleaned_text = cleaned_text.encode('utf-8', 'ignore').decode('utf-8')
        
        # 3단계: 제어 문자 및 공백 정리
        cleaned_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        return cleaned_text.strip()
        
    except Exception as e:
        print(f"⚠️ 치명적 텍스트 정리 에러: {e}")
        # 최후의 수단: 아스키만 남기기
        return ''.join(char for char in text if ord(char) < 128).strip()


def create_text_splitter(chunk_size: int = 1000, chunk_overlap: int = 200):
    """텍스트 스플리터 생성"""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )


def parse_markdown_file(file_path: Path) -> Dict[str, Any]:
    """마크다운 파일 하나 파싱"""
    with open(file_path, "r", encoding="utf-8") as f:
        post = frontmatter.load(f)

    return {
        "content": clean_text(post.content),  # 텍스트 정리
        "metadata": {
            "source": str(file_path),
            "title": clean_text(post.metadata.get("title", file_path.stem)),  # 제목도 정리
            "tags": ", ".join(post.metadata.get("tags", [])),  # 리스트를 문자열로 변환
            "date": str(post.metadata.get("date", "")),  # None일 수 있으니 문자열로 변환
            "file_name": file_path.name,
        },
    }


def chunk_text(content: str, text_splitter) -> List[str]:
    """텍스트를 청크로 분할"""
    return text_splitter.split_text(content) if content else []


def create_document_chunks(
    parsed_doc: Dict[str, Any], text_splitter
) -> List[Dict[str, Any]]:
    """파싱된 문서를 청크로 변환"""
    chunks = chunk_text(parsed_doc["content"], text_splitter)

    document_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_metadata = parsed_doc["metadata"].copy()
        chunk_metadata.update({"chunk_index": i, "total_chunks": len(chunks)})

        document_chunks.append({"content": chunk, "metadata": chunk_metadata})

    return document_chunks


def process_obsidian_vault(
    vault_path: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Dict[str, Any]]:
    """옵시디언 볼트 전체 처리"""
    vault_path = Path(vault_path)
    text_splitter = create_text_splitter(chunk_size, chunk_overlap)
    all_chunks = []

    for md_file in vault_path.rglob("*.md"):
        print(f"📖 처리 중: {md_file.name}")

        try:
            parsed_doc = parse_markdown_file(md_file)
            chunks = create_document_chunks(parsed_doc, text_splitter)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"❌ 에러 {md_file}: {e}")

    print(f"✅ 총 {len(all_chunks)}개 청크 생성")
    return all_chunks
