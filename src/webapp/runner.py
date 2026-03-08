"""
FastAPI 서버 실행 스크립트
- uvicorn으로 LAH RL API 서버 실행
- 포트, 호스트, 리로드 설정 지원
"""

from __future__ import annotations

import os
import sys
import argparse
import logging

# 프로젝트 루트를 PYTHONPATH에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def main():
    """서버 실행 엔트리포인트."""
    parser = argparse.ArgumentParser(description="LAH RL API 서버 실행")
    parser.add_argument("--host", default="0.0.0.0", help="서버 호스트 (기본: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="서버 포트 (기본: 8000)")
    parser.add_argument("--reload", action="store_true", help="개발 모드: 파일 변경 시 자동 재시작")
    parser.add_argument("--workers", type=int, default=1, help="워커 수 (프로덕션용)")
    parser.add_argument("--log-level", default="info",
                        choices=["debug", "info", "warning", "error"],
                        help="로그 레벨")
    args = parser.parse_args()

    # 로깅 설정
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    import uvicorn

    logger.info(f"LAH RL API 서버 시작: http://{args.host}:{args.port}")
    logger.info(f"API 문서: http://{args.host}:{args.port}/docs")
    logger.info(f"리로드 모드: {args.reload}")

    uvicorn.run(
        "src.webapp.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=args.log_level,
        app_dir=project_root,
    )


if __name__ == "__main__":
    main()
