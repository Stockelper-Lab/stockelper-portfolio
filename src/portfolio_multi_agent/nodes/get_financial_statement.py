import contextlib
import io
import os
import asyncio
import logging
import threading
from pydantic import BaseModel, Field
from portfolio_multi_agent.state import Stock, AnalysisResult
import OpenDartReader
from datetime import datetime

from multi_agent.dart import (
    is_dart_quota_exceeded_error,
    mask_api_key,
    parse_open_dart_api_keys,
)

logger = logging.getLogger(__name__)


class InputState(BaseModel):
    portfolio_list: list[Stock] = Field(default_factory=list)


class OutputState(BaseModel):
    analysis_results: list[AnalysisResult] = Field(default_factory=list)


class FinancialStatement:
    name = "FinancialStatement"

    def __init__(self):
        # 워크플로우 컴파일 단계에서 외부 API(네트워크) 호출로 서버가 죽지 않도록
        # DART 클라이언트는 실제 분석 호출 시점에 지연 초기화합니다.
        self.api_keys = parse_open_dart_api_keys()
        self.dart = None
        self._dart_api_key = None
        self._dart_key_index = 0
        self._dart_exhausted_keys: set[str] = set()
        self._dart_lock = threading.Lock()

        if not self.api_keys:
            logger.warning(
                "OPEN_DART_API_KEY(S) is not set; FinancialStatement analysis will be disabled."
            )

    def _invalidate_current_key(self):
        """현재 키를 '소진'으로 마킹하고 클라이언트를 무효화합니다."""
        with self._dart_lock:
            if self._dart_api_key:
                self._dart_exhausted_keys.add(self._dart_api_key)
            self.dart = None
            self._dart_api_key = None

    def _get_dart(self):
        """DART 클라이언트를 지연 초기화하여 반환합니다."""
        keys = parse_open_dart_api_keys()
        if keys != self.api_keys:
            # env 변경 반영
            self.api_keys = keys
            self._dart_key_index = 0
            self._dart_exhausted_keys = set()
            self.dart = None
            self._dart_api_key = None

        if self.dart is not None and self._dart_api_key in (self.api_keys or []):
            return self.dart

        if not self.api_keys:
            raise ValueError("OPEN_DART_API_KEY 또는 OPEN_DART_API_KEYS 환경변수가 설정되어 있지 않습니다.")

        with self._dart_lock:
            # double-check
            if self.dart is not None and self._dart_api_key in (self.api_keys or []):
                return self.dart

            keys = list(self.api_keys or [])
            start_idx = int(self._dart_key_index or 0) % len(keys)
            idx = start_idx
            tried = 0

            while tried < len(keys):
                api_key = keys[idx]
                idx = (idx + 1) % len(keys)
                tried += 1

                if api_key in self._dart_exhausted_keys:
                    continue

                try:
                    self.dart = OpenDartReader(api_key)
                    self._dart_api_key = api_key
                    self._dart_key_index = idx
                    return self.dart
                except Exception as e:
                    if is_dart_quota_exceeded_error(e):
                        self._dart_exhausted_keys.add(api_key)
                        logger.warning(
                            "OpenDartReader quota exceeded(status=020). rotate key: %s",
                            mask_api_key(api_key),
                        )
                        continue

                    self._dart_exhausted_keys.add(api_key)
                    logger.warning(
                        "Failed to initialize OpenDartReader (key=%s): %s",
                        mask_api_key(api_key),
                        e,
                    )
                    continue

            raise ValueError("OpenDartReader 초기화 실패: 사용 가능한 DART API Key가 없습니다.")

    def _finstate_all_with_capture(self, dart, stock_code: str, year: int):
        """`dart.finstate_all()` 호출 시 출력(표준출력/표준에러)을 캡처합니다.

        OpenDartReader는 일부 오류(예: status=020)를 예외로 던지지 않고,
        콘솔에 출력한 뒤 빈 DataFrame을 반환할 수 있습니다. 이를 감지해 키 로테이션에 활용합니다.

        주의: OpenDartReader 인스턴스/표준출력은 스레드 세이프하지 않을 수 있어,
        `_dart_lock`으로 호출을 직렬화합니다.
        """
        buf = io.StringIO()
        with self._dart_lock, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            df = dart.finstate_all(stock_code, year)
        return df, buf.getvalue()

    async def __call__(self, state: InputState) -> OutputState:
        """
        여러 종목에 대한 재무제표 분석을 병렬로 실행

        Args:
            state: 입력 상태 (종목 리스트)

        Returns:
            분석 결과 리스트
        """
        if not state.portfolio_list:
            return {"analysis_results": []}

        # 모든 종목에 대해 병렬로 재무제표 분석
        # DART API는 동기 함수이므로 asyncio.to_thread 사용
        tasks = [
            asyncio.to_thread(self.analyze_single_stock, stock)
            for stock in state.portfolio_list
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 성공한 결과만 필터링
        analysis_results = []
        for i, result in enumerate(results):
            if isinstance(result, AnalysisResult):
                analysis_results.append(result)
            elif isinstance(result, Exception):
                continue

        return {"analysis_results": analysis_results}

    def calculater(self, financial_statement_all):
        """
        재무제표 데이터로부터 주요 재무 지표를 계산합니다.

        Args:
            financial_statement_all: OpenDartReader로부터 가져온 재무제표 DataFrame

        Returns:
            dict: 계산된 재무 지표들 (한글 키, 포맷팅된 값)
        """
        df_55 = financial_statement_all.iloc[:]

        # 사용할 계정 ID 목록
        financial_data = df_55[
            df_55["account_id"].isin(
                [
                    "ifrs-full_CurrentAssets",
                    "ifrs-full_CurrentLiabilities",
                    "ifrs-full_Liabilities",
                    "ifrs-full_Equity",
                    "ifrs-full_SharePremium",
                    "ifrs-full_RetainedEarnings",
                    "ifrs-full_IssuedCapital",
                    "dart_OperatingIncomeLoss",
                    "dart_OtherGains",
                    "dart_OtherLosses",
                    "ifrs-full_ProfitLoss",
                    "ifrs-full_Revenue",
                    "ifrs-full_FinanceCosts",
                ]
            )
        ]

        # 계산을 위한 재무 데이터 딕셔너리 생성
        financial_dict = financial_data.set_index("account_id")[
            "thstrm_amount"
        ].to_dict()

        # 계산을 위해 모든 값을 float로 변환
        for key in financial_dict:
            financial_dict[key] = float(financial_dict[key])

        results_dict = {}

        # 1. 유동비율 = (유동자산 / 유동부채) * 100
        try:
            current_ratio = (
                financial_dict["ifrs-full_CurrentAssets"]
                / financial_dict["ifrs-full_CurrentLiabilities"]
            ) * 100
            results_dict["유동비율"] = f"{current_ratio:.2f}%"
        except:
            pass

        # 2. 부채비율 = (부채총계 / 자본총계) * 100
        try:
            debt_ratio = (
                financial_dict["ifrs-full_Liabilities"]
                / financial_dict["ifrs-full_Equity"]
            ) * 100
            results_dict["부채비율"] = f"{debt_ratio:.2f}%"
        except:
            pass

        # 3. 유보율 = (자본잉여금 + 이익잉여금) / 납입자본금 * 100
        try:
            reserve_ratio = (
                (
                    financial_dict["ifrs-full_SharePremium"]
                    + financial_dict["ifrs-full_RetainedEarnings"]
                )
                / financial_dict["ifrs-full_IssuedCapital"]
            ) * 100
            results_dict["유보율"] = f"{reserve_ratio:.2f}%"
        except:
            pass

        # 4. 자본잠식률 = {(자본금 - 자본총계) / 자본금} * 100
        try:
            capital_impairment_ratio = (
                (
                    financial_dict["ifrs-full_IssuedCapital"]
                    - financial_dict["ifrs-full_Equity"]
                )
                / financial_dict["ifrs-full_IssuedCapital"]
            ) * 100
            results_dict["자본잠식률"] = f"{capital_impairment_ratio:.2f}%"
        except:
            pass

        # 5. 경상이익 = 영업이익 + 영업외수익 - 영업외비용
        ordinary_income = 0
        try:
            ordinary_income = (
                financial_dict["dart_OperatingIncomeLoss"]
                + financial_dict["dart_OtherGains"]
                - financial_dict["dart_OtherLosses"]
            )
            results_dict["경상이익"] = f"{ordinary_income:.2f}원"
        except:
            pass

        # 6. 매출액경상이익률 = 경상이익 / 매출액 * 100
        try:
            ordinary_income_ratio = (
                ordinary_income / financial_dict["ifrs-full_Revenue"]
            ) * 100
            results_dict["매출액경상이익률"] = f"{ordinary_income_ratio:.2f}%"
        except:
            pass

        # 7. 이자보상배율 = 영업이익 / 이자비용 * 100
        try:
            interest_coverage_ratio = (
                financial_dict["dart_OperatingIncomeLoss"]
                / financial_dict["ifrs-full_FinanceCosts"]
            ) * 100
            results_dict["이자보상배율"] = f"{interest_coverage_ratio:.2f}%"
        except:
            pass

        # 8. 자기자본이익률 = 당기순이익 / 자본총액 * 100
        try:
            roe = (
                financial_dict["ifrs-full_ProfitLoss"]
                / financial_dict["ifrs-full_Equity"]
            ) * 100
            results_dict["자기자본이익률"] = f"{roe:.2f}%"
        except:
            pass

        return results_dict

    def analyze_single_stock(self, stock: Stock) -> AnalysisResult:
        """
        단일 종목에 대한 재무제표 분석

        Args:
            stock: 종목 정보

        Returns:
            AnalysisResult 객체
        """
        # DART 클라이언트 준비(키 누락/무효시 에러 메시지 반환)
        try:
            dart = self._get_dart()
        except Exception as e:
            return AnalysisResult(
                code=stock.code,
                name=stock.name,
                type="financial_statement",
                result=f"오류: DART API 초기화 실패 - {e}",
            )

        # 최근 5년 내 재무제표 데이터 조회
        current_year = datetime.now().year
        financial_statement_all = None

        for offset in range(5):
            year = current_year - offset
            try:
                df, out = self._finstate_all_with_capture(dart, stock.code, year)
                if df is not None and not df.empty:
                    financial_statement_all = df
                    break

                # OpenDartReader가 예외 대신 출력+빈 DF로 신호를 주는 케이스(status=020 등)
                if is_dart_quota_exceeded_error(out):
                    logger.warning(
                        "DART finstate_all quota exceeded(status=020) for %s(%s); rotate key and retry once",
                        stock.name,
                        stock.code,
                    )
                    self._invalidate_current_key()
                    try:
                        dart = self._get_dart()
                        df2, out2 = self._finstate_all_with_capture(dart, stock.code, year)
                        if df2 is not None and not df2.empty:
                            financial_statement_all = df2
                            break
                        if is_dart_quota_exceeded_error(out2):
                            # 다음 호출에서 다른 키를 강제하도록 소진 마킹
                            self._invalidate_current_key()
                    except Exception:
                        pass

            except Exception as e:
                if is_dart_quota_exceeded_error(e):
                    logger.warning(
                        "DART finstate_all quota exceeded(status=020) for %s(%s); rotate key and retry once",
                        stock.name,
                        stock.code,
                    )
                    self._invalidate_current_key()
                    try:
                        dart = self._get_dart()
                        df2, out2 = self._finstate_all_with_capture(dart, stock.code, year)
                        if df2 is not None and not df2.empty:
                            financial_statement_all = df2
                            break
                        if is_dart_quota_exceeded_error(out2):
                            self._invalidate_current_key()
                    except Exception:
                        pass
                continue

        if financial_statement_all is None or financial_statement_all.empty:
            return AnalysisResult(
                code=stock.code,
                name=stock.name,
                type="financial_statement",
                result=f"오류: 종목코드 {stock.code}({stock.name})의 최근 5년 내 재무제표를 찾지 못했습니다. DART에 등록된 종목인지 확인해주세요.",
            )

        # 재무 지표 계산
        analysis_result = self.calculater(financial_statement_all)

        # 결과를 문자열로 포맷팅
        if not analysis_result:
            result_str = "재무제표 데이터를 분석했으나 계산 가능한 지표가 없습니다."
        else:
            result_lines = [f"{stock.name}({stock.code}) 재무제표 분석 결과:\n"]
            for key, value in analysis_result.items():
                result_lines.append(f"- {key}: {value}")
            result_str = "\n".join(result_lines)

        return AnalysisResult(
            code=stock.code,
            name=stock.name,
            type="financial_statement",
            result=result_str,
        )
