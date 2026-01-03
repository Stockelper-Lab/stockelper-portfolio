import logging
import os
import sys

import dotenv
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.base import router as base_router
from routers.portfolio import router as portfolio_router

# 환경 변수 로딩을 최우선으로 처리
dotenv.load_dotenv(override=True)

DEBUG = os.getenv("DEBUG", "false").strip().lower() in {"1", "true", "yes"}
HOST = os.getenv("HOST", "0.0.0.0").strip()
PORT = int(os.getenv("PORT", "21010"))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# FastAPI 애플리케이션 생성
app = FastAPI(debug=DEBUG)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(base_router)
app.include_router(portfolio_router)

if __name__ == "__main__":
    try:
        print("🚀 Starting Stockelper Portfolio Server...")
        print(f"📍 Server will run on http://{HOST}:{PORT}")
        print(f"🔧 Debug mode: {DEBUG}")

        uvicorn.run(
            app,
            host=HOST,
            port=PORT,
            reload=DEBUG,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)