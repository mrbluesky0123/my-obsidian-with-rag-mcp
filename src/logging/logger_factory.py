import logging
import os
from datetime import datetime
from typing import Optional


class LoggerFactory:
    """스프링부트 스타일 로거 팩토리"""

    _configured = False
    _log_dir = "logs"
    _log_level = logging.INFO

    @classmethod
    def configure_logging(cls, log_dir: str = None, log_level: int = logging.INFO):
        """로깅 시스템 전역 설정"""
        if cls._configured:
            return

        # 로그 디렉토리 경로 결정 (쓰기 가능한 위치 우선)
        if log_dir is None:
            # 쓰기 가능한 디렉토리 찾기 (Claude Desktop 샌드박스 환경 고려)
            possible_dirs = [
                os.path.expanduser("~/logs/obsidian-rag"),  # 홈 디렉토리
                "/tmp/obsidian-rag-logs",  # 임시 디렉토리
                "logs"  # 현재 디렉토리 (마지막 시도)
            ]

            for dir_path in possible_dirs:
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    # 쓰기 테스트
                    test_file = os.path.join(dir_path, "test_write.tmp")
                    with open(test_file, 'w') as f:
                        f.write("test")
                    os.remove(test_file)
                    log_dir = dir_path
                    break
                except (OSError, PermissionError):
                    continue
            else:
                # 모든 디렉토리가 실패한 경우 콘솔 로깅만 사용
                log_dir = None

        cls._log_dir = log_dir
        cls._log_level = log_level

        # 로그 디렉토리 생성 (가능한 경우에만)
        if log_dir:
            try:
                os.makedirs(log_dir, exist_ok=True)
            except (OSError, PermissionError) as e:
                print(f"로그 디렉토리 생성 실패, 콘솔 로깅만 사용: {e}")
                cls._log_dir = None

        # 로그 디렉토리 권한을 777으로 설정 (가능한 경우에만)
        if cls._log_dir:
            try:
                import stat
                os.chmod(cls._log_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            except Exception as e:
                print(f"로그 디렉토리 권한 설정 실패: {e}")

        # 전역 로그 설정
        cls._setup_global_logging()
        cls._configured = True

    @classmethod
    def _setup_global_logging(cls):
        """전역 로깅 설정"""
        handlers = []

        # 커스텀 포맷터 (밀리초 표시)
        class SpringBootFormatter(logging.Formatter):
            def formatTime(self, record, datefmt=None):
                dt = datetime.fromtimestamp(record.created)
                return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # 밀리초까지만

        spring_formatter = SpringBootFormatter(
            fmt='%(asctime)s %(levelname)-5s [%(name)s,%(thread)d] --- %(message)s'
        )

        # 파일 핸들러 (가능한 경우에만)
        if cls._log_dir:
            try:
                today = datetime.now().strftime("%Y-%m-%d")
                log_file = os.path.join(cls._log_dir, f"obsidian-rag-{today}.log")

                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(cls._log_level)
                file_handler.setFormatter(spring_formatter)
                handlers.append(file_handler)

                # 로그 파일 권한을 666으로 설정
                try:
                    import stat
                    os.chmod(log_file, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)
                except Exception as e:
                    print(f"로그 파일 권한 설정 실패: {e}")

            except Exception as e:
                print(f"파일 핸들러 생성 실패, 콘솔 로깅만 사용: {e}")

        # 콘솔 핸들러 (항상 추가)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(cls._log_level)
        console_handler.setFormatter(spring_formatter)
        handlers.append(console_handler)

        # 루트 로거 설정
        root_logger = logging.getLogger()
        root_logger.setLevel(cls._log_level)

        # 기존 핸들러 제거
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # 새 핸들러 추가
        for handler in handlers:
            root_logger.addHandler(handler)

    @classmethod
    def get_logger(cls, name: Optional[str] = None) -> logging.Logger:
        """로거 인스턴스 생성 (스프링부트 @Slf4j 스타일)"""
        if not cls._configured:
            cls.configure_logging()

        if name is None:
            # 호출한 모듈명을 자동으로 추출
            import inspect
            frame = inspect.currentframe().f_back
            name = frame.f_globals.get('__name__', 'unknown')

        logger = logging.getLogger(name)
        return logger

    @classmethod
    def get_class_logger(cls, class_obj) -> logging.Logger:
        """클래스 기반 로거 생성"""
        class_name = f"{class_obj.__module__}.{class_obj.__name__}"
        return cls.get_logger(class_name)


# 애플리케이션 시작 시 로깅 초기화
def init_logging():
    """애플리케이션 로깅 초기화"""
    LoggerFactory.configure_logging(
        log_dir=None,  # 자동으로 쓰기 가능한 디렉토리 찾기
        log_level=logging.INFO
    )

    logger = LoggerFactory.get_logger("obsidian_rag.startup")
    logger.info("로깅 시스템 초기화 완료")
    if LoggerFactory._log_dir:
        logger.info(f"로그 파일 위치: {LoggerFactory._log_dir}")
    else:
        logger.info("콘솔 로깅만 사용 (파일 쓰기 불가)")