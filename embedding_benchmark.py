#!/usr/bin/env python3
"""
임베딩 모델 성능 비교 프로그램
Google Generative AI vs KoSimCSE 성능을 비교합니다.
"""
import os
import time
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
from dotenv import load_dotenv

from src.vectorstore.vector_db import VectorDB
from src.obsidian.obsidian_loader import process_obsidian_vault

load_dotenv()


class EmbeddingBenchmark:
    """임베딩 모델 성능 비교 클래스"""

    def __init__(self, vault_path: str):
        self.vault_path = vault_path
        self.test_queries = self._setup_test_queries()

    def _setup_test_queries(self) -> List[Dict[str, Any]]:
        """테스트 쿼리 설정"""
        return [
            {
                "query": "오늘 할 일",
                "category": "업무 일정",
                "expected_keywords": ["todo", "할일", "업무", "작업", "스케줄"]
            },
            {
                "query": "회의 일정",
                "category": "미팅",
                "expected_keywords": ["회의", "미팅", "meeting", "일정", "약속"]
            },
            {
                "query": "프로젝트 아이디어",
                "category": "아이디어",
                "expected_keywords": ["아이디어", "프로젝트", "기획", "계획", "구상"]
            },
            {
                "query": "개발 학습",
                "category": "학습",
                "expected_keywords": ["개발", "학습", "공부", "코딩", "프로그래밍"]
            },
            {
                "query": "파이썬 문법",
                "category": "기술",
                "expected_keywords": ["파이썬", "python", "문법", "코드", "함수"]
            },
            {
                "query": "버그 수정",
                "category": "디버깅",
                "expected_keywords": ["버그", "오류", "에러", "수정", "디버그"]
            },
            {
                "query": "데이터베이스 설계",
                "category": "설계",
                "expected_keywords": ["데이터베이스", "DB", "설계", "스키마", "테이블"]
            },
            {
                "query": "API 개발",
                "category": "개발",
                "expected_keywords": ["API", "개발", "엔드포인트", "서버", "클라이언트"]
            }
        ]

    def setup_databases(self) -> Tuple[VectorDB, VectorDB]:
        """두 임베딩 모델로 벡터DB 설정"""
        print("📚 옵시디언 노트 로딩 중...")
        documents = process_obsidian_vault(self.vault_path)
        print(f"✅ {len(documents)}개 문서 청크 로딩 완료!")

        # Google 임베딩 DB
        print("\n🌍 Google 임베딩 벡터DB 생성 중...")
        google_db_path = "./benchmark_google_db"
        if os.path.exists(google_db_path):
            import shutil
            shutil.rmtree(google_db_path)

        google_db = VectorDB(google_db_path, embedding_type="google")
        start_time = time.time()
        google_db.add_documents(documents)
        google_setup_time = time.time() - start_time

        # KoSimCSE 임베딩 DB
        print("\n🇰🇷 KoSimCSE 임베딩 벡터DB 생성 중...")
        kosimcse_db_path = "./benchmark_kosimcse_db"
        if os.path.exists(kosimcse_db_path):
            import shutil
            shutil.rmtree(kosimcse_db_path)

        kosimcse_db = VectorDB(kosimcse_db_path, embedding_type="kosimcse")
        start_time = time.time()
        kosimcse_db.add_documents(documents)
        kosimcse_setup_time = time.time() - start_time

        print(f"\n⏱️ 초기화 시간:")
        print(f"  Google: {google_setup_time:.2f}초")
        print(f"  KoSimCSE: {kosimcse_setup_time:.2f}초")

        return google_db, kosimcse_db

    def evaluate_search_quality(self, results: List[Any], expected_keywords: List[str]) -> float:
        """검색 결과 품질 평가"""
        if not results:
            return 0.0

        total_score = 0.0
        for doc in results:
            content = doc.page_content.lower()
            title = doc.metadata.get('title', '').lower()
            combined_text = content + ' ' + title

            # 키워드 매치 점수 계산
            matched_keywords = sum(1 for keyword in expected_keywords
                                 if keyword.lower() in combined_text)
            keyword_score = matched_keywords / len(expected_keywords)

            # 위치 가중치 (상위 결과일수록 높은 점수)
            position_weight = (len(results) - results.index(doc)) / len(results)

            total_score += keyword_score * position_weight

        return total_score / len(results)

    def run_benchmark(self) -> Dict[str, Any]:
        """벤치마크 실행"""
        print("🚀 임베딩 모델 성능 비교 시작!\n")

        # 벡터DB 설정
        google_db, kosimcse_db = self.setup_databases()

        results = {
            "google": {
                "search_times": [],
                "quality_scores": [],
                "total_results": []
            },
            "kosimcse": {
                "search_times": [],
                "quality_scores": [],
                "total_results": []
            },
            "test_details": []
        }

        print("\n🔍 검색 성능 테스트 시작...\n")

        for i, test_case in enumerate(self.test_queries):
            query = test_case["query"]
            category = test_case["category"]
            expected_keywords = test_case["expected_keywords"]

            print(f"📌 테스트 {i+1}: '{query}' ({category})")

            # Google 임베딩 테스트
            start_time = time.time()
            google_results = google_db.search(query, k=5)
            google_search_time = time.time() - start_time
            google_quality = self.evaluate_search_quality(google_results, expected_keywords)

            # KoSimCSE 임베딩 테스트
            start_time = time.time()
            kosimcse_results = kosimcse_db.search(query, k=5)
            kosimcse_search_time = time.time() - start_time
            kosimcse_quality = self.evaluate_search_quality(kosimcse_results, expected_keywords)

            # 결과 저장
            results["google"]["search_times"].append(google_search_time)
            results["google"]["quality_scores"].append(google_quality)
            results["google"]["total_results"].append(len(google_results))

            results["kosimcse"]["search_times"].append(kosimcse_search_time)
            results["kosimcse"]["quality_scores"].append(kosimcse_quality)
            results["kosimcse"]["total_results"].append(len(kosimcse_results))

            # 상세 결과 저장
            test_detail = {
                "query": query,
                "category": category,
                "google": {
                    "search_time": google_search_time,
                    "quality_score": google_quality,
                    "results_count": len(google_results),
                    "top_result_title": google_results[0].metadata.get('title', 'N/A') if google_results else 'N/A'
                },
                "kosimcse": {
                    "search_time": kosimcse_search_time,
                    "quality_score": kosimcse_quality,
                    "results_count": len(kosimcse_results),
                    "top_result_title": kosimcse_results[0].metadata.get('title', 'N/A') if kosimcse_results else 'N/A'
                }
            }
            results["test_details"].append(test_detail)

            # 실시간 결과 출력
            print(f"  Google:   {google_search_time:.3f}초, 품질: {google_quality:.3f}, 결과: {len(google_results)}개")
            print(f"  KoSimCSE: {kosimcse_search_time:.3f}초, 품질: {kosimcse_quality:.3f}, 결과: {len(kosimcse_results)}개")

            # 승자 표시
            if google_quality > kosimcse_quality:
                print("  🏆 Google 승리!")
            elif kosimcse_quality > google_quality:
                print("  🏆 KoSimCSE 승리!")
            else:
                print("  🤝 무승부!")
            print()

        return results

    def print_summary(self, results: Dict[str, Any]):
        """결과 요약 출력"""
        google_data = results["google"]
        kosimcse_data = results["kosimcse"]

        print("=" * 60)
        print("📊 벤치마크 결과 요약")
        print("=" * 60)

        # 평균 검색 시간
        google_avg_time = sum(google_data["search_times"]) / len(google_data["search_times"])
        kosimcse_avg_time = sum(kosimcse_data["search_times"]) / len(kosimcse_data["search_times"])

        print(f"\n⏱️ 평균 검색 시간:")
        print(f"  Google:   {google_avg_time:.3f}초")
        print(f"  KoSimCSE: {kosimcse_avg_time:.3f}초")

        if google_avg_time < kosimcse_avg_time:
            print(f"  🚀 Google이 {((kosimcse_avg_time/google_avg_time - 1) * 100):.1f}% 더 빠름")
        else:
            print(f"  🚀 KoSimCSE가 {((google_avg_time/kosimcse_avg_time - 1) * 100):.1f}% 더 빠름")

        # 평균 검색 품질
        google_avg_quality = sum(google_data["quality_scores"]) / len(google_data["quality_scores"])
        kosimcse_avg_quality = sum(kosimcse_data["quality_scores"]) / len(kosimcse_data["quality_scores"])

        print(f"\n🎯 평균 검색 품질:")
        print(f"  Google:   {google_avg_quality:.3f}")
        print(f"  KoSimCSE: {kosimcse_avg_quality:.3f}")

        if google_avg_quality > kosimcse_avg_quality:
            print(f"  🏆 Google이 {((google_avg_quality/kosimcse_avg_quality - 1) * 100):.1f}% 더 정확함")
        else:
            print(f"  🏆 KoSimCSE가 {((kosimcse_avg_quality/google_avg_quality - 1) * 100):.1f}% 더 정확함")

        # 승률 계산
        google_wins = sum(1 for g, k in zip(google_data["quality_scores"], kosimcse_data["quality_scores"]) if g > k)
        kosimcse_wins = sum(1 for g, k in zip(google_data["quality_scores"], kosimcse_data["quality_scores"]) if k > g)
        ties = len(google_data["quality_scores"]) - google_wins - kosimcse_wins

        print(f"\n🏁 승부 결과:")
        print(f"  Google 승리: {google_wins}회")
        print(f"  KoSimCSE 승리: {kosimcse_wins}회")
        print(f"  무승부: {ties}회")

        # 총평
        print(f"\n📝 총평:")
        if google_avg_quality > kosimcse_avg_quality and google_avg_time < kosimcse_avg_time:
            print("  🌍 Google 임베딩이 속도와 정확도 모두에서 우세합니다.")
        elif kosimcse_avg_quality > google_avg_quality and kosimcse_avg_time < google_avg_time:
            print("  🇰🇷 KoSimCSE가 속도와 정확도 모두에서 우세합니다.")
        elif kosimcse_avg_quality > google_avg_quality:
            print("  🇰🇷 KoSimCSE가 정확도에서 우세하지만, Google이 속도에서 앞섭니다.")
        elif google_avg_quality > kosimcse_avg_quality:
            print("  🌍 Google이 정확도에서 우세하지만, KoSimCSE가 속도에서 앞섭니다.")
        else:
            print("  🤝 두 모델의 성능이 비슷합니다.")

        print("\n💡 추천:")
        if kosimcse_avg_quality > google_avg_quality * 1.1:
            print("  한국어 문서가 많다면 KoSimCSE 사용을 추천합니다.")
        elif google_avg_time < kosimcse_avg_time * 0.8:
            print("  빠른 응답이 중요하다면 Google 임베딩 사용을 추천합니다.")
        else:
            print("  용도에 따라 선택하세요. 정확도 우선이면 KoSimCSE, 속도 우선이면 Google.")

    def save_results(self, results: Dict[str, Any], output_file: str = "benchmark_results.json"):
        """결과를 JSON 파일로 저장"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 상세 결과가 {output_file}에 저장되었습니다.")


def main():
    """메인 함수"""
    vault_path = "/Users/mrbluesky/Documents/memo"  # 옵시디언 볼트 경로

    if not os.path.exists(vault_path):
        print(f"❌ 옵시디언 볼트를 찾을 수 없습니다: {vault_path}")
        print("VAULT_PATH를 확인해주세요.")
        return

    # 벤치마크 실행
    benchmark = EmbeddingBenchmark(vault_path)
    results = benchmark.run_benchmark()

    # 결과 출력 및 저장
    benchmark.print_summary(results)
    benchmark.save_results(results)

    print("\n🎉 벤치마크 완료!")


if __name__ == "__main__":
    main()