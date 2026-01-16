import os
import asyncio
import pandas as pd
from typing import Any, List, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from portfolio_multi_agent.state import Stock, AnalysisResult, MarketData
import mojito


class InputState(BaseModel):
    # user context (LoadUserContext에서 채워짐)
    user_id: int | None = Field(default=None, description="사용자 ID")
    kis_app_key: str | None = Field(default=None, description="KIS App Key")
    kis_app_secret: str | None = Field(default=None, description="KIS App Secret")
    kis_access_token: str | None = Field(default=None, description="KIS Access Token")
    account_no: str | None = Field(default=None, description="계좌번호")

    portfolio_list: list[Stock] = Field(default_factory=list)


class OutputState(BaseModel):
    analysis_results: list[AnalysisResult] = Field(default_factory=list)
    market_data_list: list[MarketData] = Field(default_factory=list)


class TechnicalIndicator:
    name = "TechnicalIndicator"

    def __init__(self):
        """
        기술적 지표 분석 및 시장 데이터 로더
        (최근 180일 데이터 사용)
        """
        pass

    async def __call__(self, state: InputState) -> OutputState:
        """
        여러 종목에 대한 기술적 지표 분석 및 시장 데이터 조회를 병렬로 실행 (Rate Limit 적용)

        Args:
            state: 입력 상태 (종목 리스트)

        Returns:
            분석 결과 리스트 및 시장 데이터 리스트
        """
        if not state.portfolio_list:
            return {"analysis_results": [], "market_data_list": []}

        app_key = getattr(state, "kis_app_key", None)
        app_secret = getattr(state, "kis_app_secret", None)
        acc_no = getattr(state, "account_no", None)

        if not all([app_key, app_secret, acc_no]):
            # 자격증명이 없으면 종목별 오류만 반환
            analysis_results = [
                AnalysisResult(
                    code=s.code,
                    name=s.name,
                    type="technical_indicator",
                    result="오류: 사용자 KIS 자격증명/계좌(account_no)가 없어 기술적 지표를 조회할 수 없습니다.",
                )
                for s in state.portfolio_list
            ]
            return {"analysis_results": analysis_results, "market_data_list": []}

        batch_size = self._get_batch_size()
        all_results = []

        # 배치 단위로 처리
        for i in range(0, len(state.portfolio_list), batch_size):
            batch = state.portfolio_list[i : i + batch_size]

            batch_results = await self._execute_batch(batch, app_key, app_secret, acc_no)
            all_results.extend(batch_results)

            # 마지막 배치가 아니면 대기 (Rate Limit 준수)
            if i + batch_size < len(state.portfolio_list):
                await asyncio.sleep(1.0)

        # 결과 필터링 및 분리
        analysis_results, market_data_list = self._filter_and_split_results(
            all_results, state.portfolio_list
        )

        return {
            "analysis_results": analysis_results,
            "market_data_list": market_data_list,
        }

    def _get_batch_size(self) -> int:
        """환경변수에서 배치 크기(초당 최대 요청 수) 조회"""
        return int(os.getenv("KIS_MAX_REQUESTS_PER_SECOND", "20"))

    async def _execute_batch(
        self, batch: List[Stock], app_key: str, app_secret: str, acc_no: str
    ) -> List[Tuple[AnalysisResult, MarketData | None]]:
        """배치 단위로 기술적 지표 분석 실행 (병렬 처리)"""
        tasks = [
            asyncio.to_thread(self.analyze_single_stock, stock, app_key, app_secret, acc_no)
            for stock in batch
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def _filter_and_split_results(
        self,
        all_results: List[Any],
        portfolio_list: List[Stock],
    ) -> Tuple[List[AnalysisResult], List[MarketData]]:
        """실행 결과 필터링 및 분리"""
        analysis_results = []
        market_data_list = []

        for i, result in enumerate(all_results):
            if isinstance(result, Exception):
                continue

            # result is Tuple[AnalysisResult, MarketData | None]
            an_res, mk_data = result
            analysis_results.append(an_res)
            if mk_data:
                market_data_list.append(mk_data)

        return analysis_results, market_data_list

    def get_stock_data(
        self, stock_code: str, app_key: str, app_secret: str, acc_no: str
    ) -> pd.DataFrame | None:
        """
        한국투자증권 API를 사용하여 주식 일별 데이터 조회 (mojito 라이브러리 사용, 최근 180일)

        - KIS 기간별 시세 API는 1회 호출에 최대 100건까지 내려올 수 있어,
          오래된 데이터는 end_day를 뒤로 옮기며 여러 번 호출해 누적합니다.
        """
        if not all([app_key, app_secret, acc_no]):
            return None

        # mojito 브로커 객체 생성 (모의투자 기준)
        broker = mojito.KoreaInvestment(
            api_key=app_key,
            api_secret=app_secret,
            acc_no=acc_no,
            mock=True,
        )

        target_days = 180
        max_pages = 5  # 100건 * 2면 180일 커버 가능하나 안전장치로 여유를 둠
        end_day = datetime.now().strftime("%Y%m%d")
        start_day = "19800104"  # 충분히 과거

        rows: list[dict] = []
        seen_dates: set[str] = set()

        for _ in range(max_pages):
            resp = broker.fetch_ohlcv_domestic(
                stock_code,
                timeframe="D",
                start_day=start_day,
                end_day=end_day,
                adj_price=True,
            )

            if not isinstance(resp, dict) or str(resp.get("rt_cd", "")).strip() != "0":
                return None

            output2 = resp.get("output2") or []
            if not output2:
                break

            # 누적 (중복 일자 제거)
            for item in output2:
                d = str(item.get("stck_bsop_date", "") or "").strip()
                if not d or d in seen_dates:
                    continue
                seen_dates.add(d)
                rows.append(item)

            # 이미 충분히 모이면 종료
            if len(seen_dates) >= target_days + 20:
                break

            # 다음 페이지: 이번 배치에서 가장 오래된 날짜 이전으로 end_day 이동
            dates = [
                str(item.get("stck_bsop_date", "") or "").strip()
                for item in output2
                if item.get("stck_bsop_date")
            ]
            if not dates:
                break

            oldest = min(dates)
            try:
                prev = (datetime.strptime(oldest, "%Y%m%d") - timedelta(days=1)).strftime(
                    "%Y%m%d"
                )
            except Exception:
                break

            if prev >= end_day:
                break
            end_day = prev

        if not rows:
            return None

        # DataFrame 생성
        df_data = []
        for item in rows:
            try:
                date = pd.to_datetime(item["stck_bsop_date"])
                open_price = float(item.get("stck_oprc", 0) or 0)
                high_price = float(item.get("stck_hgpr", 0) or 0)
                low_price = float(item.get("stck_lwpr", 0) or 0)
                close_price = float(item.get("stck_clpr", 0) or 0)
                volume = int(float(item.get("acml_vol", 0) or 0))
            except Exception:
                continue

            df_data.append(
                {
                    "Date": date,
                    "Open": open_price,
                    "High": high_price,
                    "Low": low_price,
                    "Close": close_price,
                    "Volume": volume,
                }
            )

        if not df_data:
            return None

        df = pd.DataFrame(df_data).set_index("Date").sort_index()
        df = df[~df.index.duplicated(keep="last")]

        # 최근 180개로 제한
        if len(df) > target_days:
            df = df.iloc[-target_days:]

        return df

    def calculate_returns(self, df: pd.DataFrame) -> list[float]:
        """
        일별 수익률 계산
        """
        if df is None or df.empty or "Close" not in df.columns:
            return []

        # 일별 수익률 계산: (당일 종가 - 전일 종가) / 전일 종가
        returns = df["Close"].pct_change()

        # 첫 번째 값은 NaN이므로 제거
        returns = returns.dropna()

        # 리스트로 변환
        return returns.tolist()

    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_stochastic(
        self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3
    ):
        """스토캐스틱 계산"""
        low_min = data["Low"].rolling(window=k_period).min()
        high_max = data["High"].rolling(window=k_period).max()

        k = 100 * ((data["Close"] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_period).mean()
        return k, d

    def calculate_indicators(self, df: pd.DataFrame):
        """기술적 지표 계산"""
        # MACD
        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()

        # 볼린저 밴드
        bb_period = 20
        bb_middle = df["Close"].rolling(window=bb_period).mean()
        bb_std = df["Close"].rolling(window=bb_period).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)

        # RSI
        rsi = self.calculate_rsi(df, 14)

        # 스토캐스틱
        k, d = self.calculate_stochastic(df, 14, 3)

        # 이동평균선
        ma20 = df["Close"].rolling(window=20).mean()
        ma60 = df["Close"].rolling(window=60).mean()
        ma120 = df["Close"].rolling(window=120).mean()

        # 최근 데이터 (가장 최신)
        if df.empty:
            return {}

        latest = df.iloc[-1]
        latest_rsi = rsi.iloc[-1]
        latest_k = k.iloc[-1]
        latest_d = d.iloc[-1]
        latest_macd = macd.iloc[-1]
        latest_signal = signal.iloc[-1]
        latest_bb_upper = bb_upper.iloc[-1]
        latest_bb_middle = bb_middle.iloc[-1]
        latest_bb_lower = bb_lower.iloc[-1]
        latest_ma20 = ma20.iloc[-1]
        latest_ma60 = ma60.iloc[-1]
        latest_ma120 = ma120.iloc[-1]

        def _fmt_ma(x: float) -> str:
            return f"{x:.2f}원" if pd.notna(x) else "N/A"

        indicators = {
            "현재가": f"{latest['Close']:.2f}원",
            "RSI(14)": f"{latest_rsi:.2f}",
            "스토캐스틱 K": f"{latest_k:.2f}",
            "스토캐스틱 D": f"{latest_d:.2f}",
            "MACD": f"{latest_macd:.2f}",
            "MACD Signal": f"{latest_signal:.2f}",
            "MACD Histogram": f"{(latest_macd - latest_signal):.2f}",
            "볼린저밴드 상단": f"{latest_bb_upper:.2f}원",
            "볼린저밴드 중간": f"{latest_bb_middle:.2f}원",
            "볼린저밴드 하단": f"{latest_bb_lower:.2f}원",
            "이동평균(20일)": _fmt_ma(latest_ma20),
            "이동평균(60일)": _fmt_ma(latest_ma60),
            "이동평균(120일)": _fmt_ma(latest_ma120),
            "거래량": f"{latest['Volume']:,}",
        }

        return indicators

    def analyze_single_stock(
        self, stock: Stock, app_key: str, app_secret: str, acc_no: str
    ) -> Tuple[AnalysisResult, MarketData | None]:
        """
        단일 종목에 대한 기술적 지표 분석 및 시장 데이터 생성

        Args:
            stock: 종목 정보

        Returns:
            (AnalysisResult, MarketData) 튜플
        """
        # 주식 데이터 조회
        df = self.get_stock_data(stock.code, app_key, app_secret, acc_no)

        if df is None or df.empty:
            error_result = AnalysisResult(
                code=stock.code,
                name=stock.name,
                type="technical_indicator",
                result=f"오류: 종목코드 {stock.code}({stock.name})의 주식 데이터를 조회하지 못했습니다.",
            )
            return error_result, None

        # 1. MarketData 생성 (수익률 계산)
        returns = self.calculate_returns(df)
        market_data = None
        if returns:
            market_data = MarketData(
                code=stock.code,
                name=stock.name,
                returns=returns,
            )

        # 2. 기술적 지표 분석
        # 기술적 지표 계산
        indicators = self.calculate_indicators(df)

        # 지표를 텍스트로 포맷팅
        indicator_text = (
            f"{stock.name}({stock.code}) 기술적 지표 (최근 {len(df)}일 데이터 기준, 최대 180일):\n\n"
        )
        if indicators:
            for key, value in indicators.items():
                indicator_text += f"- {key}: {value}\n"
        else:
            indicator_text += "지표 계산 실패\n"

        analysis_result = AnalysisResult(
            code=stock.code,
            name=stock.name,
            type="technical_indicator",
            result=indicator_text,
        )

        return analysis_result, market_data
