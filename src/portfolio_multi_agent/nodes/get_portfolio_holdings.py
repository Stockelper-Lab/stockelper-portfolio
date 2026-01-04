import os
import asyncio
import requests
from pydantic import BaseModel, Field
from portfolio_multi_agent.state import HoldingStock, Stock


class InputState(BaseModel):
    # user context (LoadUserContext에서 채워짐)
    user_id: int = Field(description="사용자 ID")
    kis_app_key: str | None = Field(default=None)
    kis_app_secret: str | None = Field(default=None)
    kis_access_token: str | None = Field(default=None)
    account_no: str | None = Field(default=None)


class OutputState(BaseModel):
    holding_stocks: list[HoldingStock] = Field(default_factory=list)


class GetPortfolioHoldings:
    name = "GetPortfolioHoldings"

    def __init__(self):
        """
        보유 종목 조회 노드
        """
        pass

    async def fetch_holdings(
        self, app_key: str, app_secret: str, access_token: str, account_no: str
    ) -> list[HoldingStock]:
        """
        계좌 보유 종목 조회

        Args:
            app_key: API 키
            app_secret: API 시크릿
            access_token: 액세스 토큰
            account_no: 계좌번호 (예: "12345678-01")

        Returns:
            보유 종목 리스트
        """
        url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-stock/v1/trading/inquire-balance"
        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {access_token}",
            "appKey": app_key,
            "appSecret": app_secret,
            "tr_id": "VTTC8434R",  # 모의투자 계좌 잔고 조회
            "custtype": "P",  # 고객타입 - P: 개인
        }
        params = {
            "CANO": account_no.split("-")[0],  # 계좌번호
            "ACNT_PRDT_CD": account_no.split("-")[1],
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "01",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

        async with asyncio.timeout(30):
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: requests.get(url, headers=headers, params=params)
            )
            # KIS는 토큰 만료 등에서 HTTP 500 + JSON(msg1) 형태로 내려오기도 하므로,
            # raise_for_status() 이전에 본문(JSON)을 먼저 확인합니다.
            try:
                res_data = response.json()
            except Exception:
                response.raise_for_status()
                raise

            # NOTE:
            # 본 프로젝트는 LoadUserContext에서 매 요청마다 access_token을 발급/DB에 저장한 뒤 사용합니다.
            # 따라서 여기서는 token 자동 갱신을 별도로 수행하지 않습니다.

            # 응답 코드 확인
            if res_data.get("rt_cd") != "0":
                raise ValueError(f"API 오류: {res_data.get('msg1', '알 수 없는 오류')}")

            # output1에 보유 종목 정보가 있음
            output1 = res_data.get("output1", [])

            holdings = []
            for item in output1:
                # 보유 수량이 0보다 큰 종목만 추가
                quantity = int(item.get("hldg_qty", "0"))
                if quantity <= 0:
                    continue

                code = item.get("pdno", "")  # 종목 코드
                name = item.get("prdt_name", "")  # 종목명
                avg_buy_price = float(item.get("pchs_avg_pric", "0"))  # 매입 평균가
                current_price = float(item.get("prpr", "0"))  # 현재가
                evaluated_amount = float(item.get("evlu_amt", "0"))  # 평가 금액
                profit_loss = float(item.get("evlu_pfls_amt", "0"))  # 평가 손익

                # 수익률 계산
                if avg_buy_price > 0:
                    return_rate = (current_price - avg_buy_price) / avg_buy_price
                else:
                    return_rate = 0.0

                holdings.append(
                    HoldingStock(
                        code=code,
                        name=name,
                        quantity=quantity,
                        avg_buy_price=avg_buy_price,
                        current_price=current_price,
                        return_rate=return_rate,
                        evaluated_amount=evaluated_amount,
                        profit_loss=profit_loss,
                    )
                )

            return holdings

    async def __call__(self, state: InputState) -> OutputState:
        """
        보유 종목 조회

        Args:
            state: 입력 상태

        Returns:
            보유 종목 리스트
        """
        # 사용자별 KIS 자격증명/계좌 (LoadUserContext에서 세팅)
        app_key = getattr(state, "kis_app_key", None)
        app_secret = getattr(state, "kis_app_secret", None)
        access_token = getattr(state, "kis_access_token", None)
        account_no = getattr(state, "account_no", None)

        if not all([app_key, app_secret, access_token, account_no]):
            print(
                "[GetPortfolioHoldings] 사용자 KIS 자격증명/계좌가 없습니다. user_id 기반 DB 로드가 필요합니다."
            )
            return {"holding_stocks": []}

        print(f"[GetPortfolioHoldings] 보유 종목 조회 시작")

        # 보유 종목 조회
        holdings = await self.fetch_holdings(
            app_key, app_secret, access_token, account_no
        )

        print(f"[GetPortfolioHoldings] 보유 종목 {len(holdings)}개 조회 완료")
        for holding in holdings:
            print(
                f"  - {holding.name}({holding.code}): {holding.quantity}주, "
                f"수익률 {holding.return_rate*100:.2f}%, "
                f"평가손익 {holding.profit_loss:,.0f}원"
            )

        return {"holding_stocks": holdings}
