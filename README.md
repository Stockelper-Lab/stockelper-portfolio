# stockelper-portfolio

`stockelper-llm`(LangGraph/LangChain 기반)에서 **포트폴리오 도메인**을 분리한 로컬 실행 서비스입니다.

## 실행

- **Python**: 3.12+
- **uv** 사용

```bash
cd /Users/oldman/Library/CloudStorage/OneDrive-개인/001_Documents/001_TelePIX/000_workspace/03_PseudoLab/Stockelper-Lab/stockelper-portfolio
uv sync --dev
uv run python src/main.py
```

- 기본 포트: **21010**
- 포트 변경: `PORT=21010` 환경변수로 제어 (예: `PORT=21020 uv run python src/main.py`)

## 주요 엔드포인트

- **Health**
  - `GET /health`
- **투자 성향 기반 추천(기존 LLM 구현 이관)**
  - `POST /portfolio/recommendations`
- **포트폴리오 매수/매도 워크플로우(LangGraph)**
  - `POST /portfolio/buy`
  - `POST /portfolio/sell`

## 환경변수(요약)

### `/portfolio/recommendations` (DB 기반 사용자/KIS 정보 조회)

- **필수**
  - `ASYNC_DATABASE_URL`: 사용자 테이블(`users`) 접근용 async DB URL
- **선택**
  - `ASYNC_DATABASE_URL_KSIC`: 산업분류(KSIC) DB(없으면 업종명 조회 비활성)
  - `OPEN_DART_API_KEY`: DART 기업정보 조회(OpenDartReader)

> 주의: `multi_agent/portfolio_analysis_agent/tools/portfolio.py`는 호출 시점에 `ASYNC_DATABASE_URL`을 확인합니다. 서버 기동은 되지만, 값이 없으면 해당 API 호출 시 에러가 납니다.

### `/portfolio/buy`, `/portfolio/sell` (portfolio_multi_agent)

- **LLM(OpenRouter)**
  - `OPENROUTER_API_KEY` (기본 base_url: `https://openrouter.ai/api/v1`)
- **DART**
  - `OPEN_DART_API_KEY`
- **KIS(모의투자/조회/주문)**
  - `APP_KEY`
  - `APP_SECRET`
  - `ACCESS_TOKEN` (보유종목 조회/주문 노드에서 사용)
  - `ACCOUNT_NO` (예: `12345678-01`)
  - `KIS_MAX_REQUESTS_PER_SECOND` (선택, 기본 20)

## 비고

- 이 레포는 **서비스 분리**(Repo split)를 위한 1차 이관 상태입니다.
- 프론트엔드/LLM 서버와의 연결(프록시/호출부)은 별도 레포(`stockelper-fe`, `stockelper-llm`)에서 환경변수 기반으로 맞추는 것을 전제로 합니다.