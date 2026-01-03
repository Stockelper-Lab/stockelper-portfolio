# Stockelper Portfolio Service

LangGraph 기반 포트폴리오 추천 및 자동 매매 서비스입니다.

## 🚀 주요 기능

- 투자 성향 기반 포트폴리오 추천
- LangGraph 다중 에이전트 매수/매도 워크플로우
- Black-Litterman 모델 기반 포트폴리오 최적화
- 한국투자증권 (KIS) API 연동 실제 거래
- 다중 지표 종목 랭킹 시스템

## 📋 기술 스택

- Python 3.12+
- FastAPI 0.111
- LangGraph (상태 그래프 기반 워크플로우)
- LangChain 1.0+
- PostgreSQL (asyncpg, psycopg)
- OpenRouter API (Perplexity, GPT-4.5.1)
- Korea Investment & Securities (KIS) API
- OpenDartReader (한국 금융감독원 DART)

## 🔌 API 엔드포인트

### 기본
- `GET /` - 루트 엔드포인트
- `GET /health` - 헬스 체크

### 포트폴리오
- `POST /portfolio/recommendations` - 투자 성향 기반 추천
- `POST /portfolio/buy` - 매수 워크플로우 (LangGraph)
- `POST /portfolio/sell` - 매도 워크플로우 (LangGraph)

## 🤖 LangGraph 워크플로우

### 매수 워크플로우

```
Ranking (11개 지표 기반)
  ↓
Analysis (병렬 3개)
  ├─ WebSearch (Perplexity)
  ├─ FinancialStatement (재무제표)
  └─ TechnicalIndicator (기술적 지표)
  ↓
ViewGenerator (Black-Litterman 뷰 생성)
  ↓
PortfolioBuilder (포트폴리오 최적화)
  ↓
PortfolioTrader (매수 주문 실행)
```

### 매도 워크플로우

```
GetPortfolioHoldings (보유 종목 조회)
  ↓
Analysis (병렬 3개)
  ├─ WebSearch
  ├─ FinancialStatement
  └─ TechnicalIndicator
  ↓
SellDecisionMaker (매도 결정)
  ↓
PortfolioSeller (매도 주문 실행)
```

## 📊 종목 랭킹 시스템

11개 랭킹 함수:
- 거래 활동성
- 영업 이익률
- 성장률
- 부채 수준
- 상승률
- 안정성
- 순이익
- 하락률
- 시가총액

## ⚙️ 환경 변수

```bash
# 서버 설정
HOST=0.0.0.0
PORT=21010
DEBUG=false

# 데이터베이스 (필수)
ASYNC_DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db
ASYNC_DATABASE_URL_KSIC=postgresql+asyncpg://user:pass@host:5432/ksic  # 선택

# KIS API (매수/매도 필수)
APP_KEY=
APP_SECRET=
ACCESS_TOKEN=
ACCOUNT_NO=12345678-01
KIS_MAX_REQUESTS_PER_SECOND=20

# 외부 API
OPEN_DART_API_KEY=
OPENROUTER_API_KEY=
```

## 🚀 빠른 시작

### 로컬 실행

```bash
# 의존성 설치
uv sync --dev

# 서버 실행
PORT=21010 uv run python src/main.py
```

### Docker 실행

```bash
# 빌드 및 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f stockelper-portfolio-server
```

## 📝 API 사용 예시

### 포트폴리오 추천

```bash
curl -X POST http://localhost:21010/portfolio/recommendations   -H "Content-Type: application/json"   -d '{
    "user_id": 1,
    "investor_type": "안정형"
  }'
```

### 매수 워크플로우

```bash
curl -X POST http://localhost:21010/portfolio/buy   -H "Content-Type: application/json"   -d '{
    "max_portfolio_size": 10,
    "rank_weight": {...},
    "portfolio_list": [...],
    "risk_free_rate": 0.03
  }'
```

### 매도 워크플로우

```bash
curl -X POST http://localhost:21010/portfolio/sell   -H "Content-Type: application/json"   -d '{
    "loss_threshold": -0.05,
    "profit_threshold": 0.15
  }'
```

## 🗄️ 데이터베이스

### users 테이블
- id, kis_app_key, kis_app_secret
- kis_access_token, account_no
- investor_type
- created_at, updated_at

### industy 테이블 (KSIC DB)
- industy_code (5자리 코드)
- industy_name (산업 분류명)

## 🔒 보안

- 모든 API 키 환경 변수 관리
- KIS 토큰 자동 갱신 (DB 저장)
- Rate limiting (초당 20 요청)
- `.env` 파일 커밋 금지

## 🐳 Docker 구성

### 서비스
- **stockelper-portfolio-server** (포트: 21010)
  - FastAPI 애플리케이션
  - 헬스체크: `/health`

### 네트워크
- `stockelper` 브리지 네트워크

## 📄 라이선스

MIT License
