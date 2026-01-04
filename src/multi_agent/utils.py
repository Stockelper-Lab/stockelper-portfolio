from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Optional

import aiohttp
import requests
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy import Column, Integer, Text, TIMESTAMP, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

load_dotenv(override=True)

Base = declarative_base()

# 사용자 테이블 모델 정의
class User(Base):
    # NOTE: stockelper_web DB 의 `users` 테이블을 참조
    __tablename__ = "users"
    __table_args__ = {"schema": os.getenv("STOCKELPER_WEB_SCHEMA", "public")}

    id = Column(Integer, primary_key=True)
    kis_app_key = Column(Text, nullable=False)
    kis_app_secret = Column(Text, nullable=False)
    kis_access_token = Column(Text, nullable=True)
    account_no = Column(Text, nullable=True)  # ex) "50132452-01"


# 설문 테이블 모델 정의 (stockelper_web schema)
class Survey(Base):
    __tablename__ = "survey"
    __table_args__ = {"schema": os.getenv("STOCKELPER_WEB_SCHEMA", "public")}

    # 컬럼 구조는 프로젝트 DB에 따라 달라질 수 있어, 여기서는 필요한 최소 컬럼만 매핑합니다.
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    answer = Column(JSONB, nullable=True)

# 산업분류 테이블 모델 정의
class Industy(Base):
    __tablename__ = "industy"

    industy_code = Column(Text, primary_key=True)  # 5자리 패딩된 산업분류코드
    industy_name = Column(Text, nullable=False)  # 산업분류명

# 사용자 정보 조회 함수
async def get_user_kis_credentials(async_engine: object, user_id: int):
    async with AsyncSession(async_engine) as session:
        stmt = select(User).where(User.id == user_id)
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()

        if user:
            return {
                "id": user.id,
                "kis_app_key": user.kis_app_key,
                "kis_app_secret": user.kis_app_secret,
                "kis_access_token": user.kis_access_token,
                "account_no": user.account_no,
            }
        else:
            return None


async def update_user_kis_credentials(
    async_engine: object, user_id: int, access_token: str
):
    async with AsyncSession(async_engine) as session:
        stmt = select(User).where(User.id == user_id)
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()
        if user is None:
            # 사용자 없음: 업데이트 불가
            return False
        user.kis_access_token = access_token
        await session.commit()
        return True


async def get_user_survey_answer(async_engine: object, user_id: int) -> Optional[dict]:
    """stockelper_web.survey.answer(JSON)에서 user_id의 설문 응답을 조회합니다."""
    async with AsyncSession(async_engine) as session:
        # 일반적으로 유저당 1건이라고 가정하고, 여러 건이면 최신(id desc)을 선택
        stmt = select(Survey).where(Survey.user_id == user_id).order_by(Survey.id.desc())
        result = await session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is None:
            return None

        answer = row.answer
        if answer is None:
            return None
        if isinstance(answer, dict):
            return answer
        # 안전하게 문자열 JSON도 지원
        try:
            return json.loads(str(answer))
        except Exception:
            return None


# =====================
# 투자 성향(설문) 매핑
# =====================

_Q1_MAP = {
    1: "미성년자",
    2: "20대",
    3: "30대",
    4: "40대",
    5: "50대",
    6: "60대 이상",
}
_Q2_MAP = {
    1: "30백만원 이하",
    2: "50백만원 이하",
    3: "70백만원 이하",
    4: "90백만원 이하",
    5: "90백만원 초과",
}
_Q3_MAP = {
    1: "5% 이하",
    2: "10% 이하",
    3: "15% 이하",
    4: "20% 이하",
    5: "20% 초과",
}
_Q4_MAP = {
    1: "현재 일정한 수입이 있고, 앞으로 유지되거나 늘어날 것 같아요",
    2: "현재 일정한 수입이 있지만, 앞으로 줄어들거나 불안정해질 것 같아요",
    3: "일정한 수입이 없거나, 주로 연금으로 생활하고 있어요",
}
_Q5_MAP = {
    1: "예금/적금/국채/MMF 등(안전)",
    2: "금융채/우량회사채/채권형펀드/원금보장형 ELS(비교적 안전)",
    3: "중간등급 회사채/부분 원금보장 ELS/혼합형펀드(중간 위험)",
    4: "저신용 회사채/주식/원금 비보장 ELS/주식형펀드(위험)",
    5: "ELW/선물옵션/고수익 주식형펀드/파생상품/신용거래(고위험)",
}
_Q6_MAP = {
    1: "파생상품 포함 대부분 구조/위험을 잘 이해",
    2: "주식/채권/펀드 구조/위험을 깊이 이해",
    3: "주식/채권/펀드 기본 특징을 앎",
    4: "투자 경험 없음",
}
_Q7_MAP = {
    1: "높은 수익 위해 원금 손실 위험도 감수",
    2: "원금 20% 미만 손실 감수 가능",
    3: "원금 10% 미만 손실 감수 가능",
    4: "원금 반드시 보전",
}
_Q8_MAP = {
    1: "3년 이상",
    2: "2~3년",
    3: "1~2년",
    4: "6개월~1년",
    5: "6개월 미만",
}


def survey_answer_to_investor_type(answer: dict) -> str:
    """survey.answer(JSON) 기반으로 투자 성향을 5단계로 산출합니다.

    반환값: "안정형" | "안정추구형" | "위험중립형" | "적극투자형" | "공격투자형"
    """
    # 안전한 기본값
    if not isinstance(answer, dict):
        return "위험중립형"

    q1 = int(answer.get("q1", 0) or 0)
    q2 = int(answer.get("q2", 0) or 0)
    q3 = int(answer.get("q3", 0) or 0)
    q4 = int(answer.get("q4", 0) or 0)
    q6 = int(answer.get("q6", 0) or 0)
    q7 = int(answer.get("q7", 0) or 0)
    q8 = int(answer.get("q8", 0) or 0)

    q5_raw = answer.get("q5", [])
    if isinstance(q5_raw, list) and q5_raw:
        q5 = int(max(q5_raw))
    elif isinstance(q5_raw, (int, float, str)) and str(q5_raw).isdigit():
        q5 = int(q5_raw)
    else:
        q5 = 0

    # 0~1로 정규화(1이 더 공격적)
    # q1(연령): 20대가 가장 공격적, 60대 이상이 가장 보수적
    age_score_map = {1: 0.7, 2: 1.0, 3: 0.85, 4: 0.7, 5: 0.5, 6: 0.3}
    age_score = age_score_map.get(q1, 0.6)

    income_score = (q2 - 1) / 4 if 1 <= q2 <= 5 else 0.5
    asset_score = (q3 - 1) / 4 if 1 <= q3 <= 5 else 0.5

    # q4(수입 전망): 1(안정/증가)이 공격적, 3(연금/무수입)이 보수적
    income_outlook_map = {1: 1.0, 2: 0.5, 3: 0.0}
    income_outlook_score = income_outlook_map.get(q4, 0.5)

    exp_score = (q5 - 1) / 4 if 1 <= q5 <= 5 else 0.0

    knowledge_map = {1: 1.0, 2: 0.7, 3: 0.4, 4: 0.0}
    knowledge_score = knowledge_map.get(q6, 0.3)

    loss_tol_map = {1: 1.0, 2: 0.7, 3: 0.4, 4: 0.0}
    loss_tol_score = loss_tol_map.get(q7, 0.2)

    horizon_map = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.3, 5: 0.0}
    horizon_score = horizon_map.get(q8, 0.4)

    # 가중합 (손실감수/경험/기간/이해도 중심)
    risk_score = (
        0.25 * loss_tol_score
        + 0.15 * exp_score
        + 0.15 * horizon_score
        + 0.15 * knowledge_score
        + 0.10 * income_outlook_score
        + 0.10 * age_score
        + 0.05 * income_score
        + 0.05 * asset_score
    )

    if risk_score < 0.2:
        return "안정형"
    if risk_score < 0.4:
        return "안정추구형"
    if risk_score < 0.6:
        return "위험중립형"
    if risk_score < 0.8:
        return "적극투자형"
    return "공격투자형"


def format_survey_answer_korean(answer: dict) -> str:
    """디버깅/로그용: survey.answer를 사람이 읽을 수 있게 변환합니다."""
    if not isinstance(answer, dict):
        return ""
    q5 = answer.get("q5", [])
    if isinstance(q5, list):
        q5_text = ", ".join(_Q5_MAP.get(int(x), str(x)) for x in q5)
    else:
        q5_text = _Q5_MAP.get(int(q5), str(q5))

    parts = [
        f"q1(연령대): {_Q1_MAP.get(int(answer.get('q1', 0)), 'N/A')}",
        f"q2(연간소득): {_Q2_MAP.get(int(answer.get('q2', 0)), 'N/A')}",
        f"q3(금융자산비중): {_Q3_MAP.get(int(answer.get('q3', 0)), 'N/A')}",
        f"q4(수입전망): {_Q4_MAP.get(int(answer.get('q4', 0)), 'N/A')}",
        f"q5(투자경험): {q5_text or 'N/A'}",
        f"q6(이해도): {_Q6_MAP.get(int(answer.get('q6', 0)), 'N/A')}",
        f"q7(손실감수): {_Q7_MAP.get(int(answer.get('q7', 0)), 'N/A')}",
        f"q8(기간): {_Q8_MAP.get(int(answer.get('q8', 0)), 'N/A')}",
    ]
    return "\n".join(parts)


async def get_access_token(app_key, app_secret):
    """접근 토큰(access token) 발급"""
    url = "https://openapivts.koreainvestment.com:29443/oauth2/tokenP"
    headers = {
        "content-type": "application/json"
    }
    body = {
        "grant_type": "client_credentials",
        "appkey": app_key,
        "appsecret": app_secret
    }
    
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=headers, json=body) as res:
            if res.status == 200:
                token_data = await res.json()
                return token_data["access_token"]
            else:
                text = await res.text()
                print(f"토큰 발급 실패: {res.status} - {text}")
                return None
    

async def check_account_balance(app_key, app_secret, access_token, account_no):
    """계좌 잔고 조회"""
    url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-stock/v1/trading/inquire-balance"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "VTTC8434R",  # 모의투자 계좌 잔고 조회 / 실전투자 : TTTC8434R 
        "custtype": "P"  # 고객타입 - P: 개인 
    }
    params = {
        "CANO": account_no.split('-')[0],  # 계좌번호
        "ACNT_PRDT_CD": account_no.split('-')[1],
        "AFHR_FLPR_YN": "N",  # 시간외 단일가 포함 여부 
        "OFL_YN": "",  # 오프라인 여부 
        "INQR_DVSN": "01",  # 조회 구분 - 01:대출일별 / 02 : 종목별
        "UNPR_DVSN": "01",  # 단가 구분 - 01:현재가
        "FUND_STTL_ICLD_YN": "N",  # 펀드 결제 포함 여부 
        "FNCG_AMT_AUTO_RDPT_YN": "N",  # 금융금액 자동 상환 여부 
        "PRCS_DVSN": "01",  # 처리 구분 - 	00 : 전일매매포함 / 01 : 전일매매미포함
        "CTX_AREA_FK100": "",  # 연속조회검색조건100
        "CTX_AREA_NK100": ""  # 연속조회검색키100
    }
    
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get(url, headers=headers, params=params, timeout=30) as res:
                if res.status == 200:
                    res_data = await res.json()
                    if res_data.get('rt_cd') == '0':  # 응답 성공
                        output = res_data.get('output2', {})[0]
                        cash = output.get('dnca_tot_amt')  # 예수금
                        total_eval = output.get('tot_evlu_amt')  # 총 평가금액
                        return {'cash': cash, 'total_eval': total_eval}
                    else:
                        print(f"잔고 조회 실패: {res_data.get('msg1')}")
                        return None
                else:
                    text = await res.text()
                    print(f"잔고 조회 요청 실패: {res.status} - {text}")
                    try:
                        res_data = await res.json()
                        return res_data['msg1']
                    except:
                        return f"오류: {text}"
        except asyncio.TimeoutError:
            print("잔고 조회 요청 시간 초과 (timeout)")
            return None


def get_hashkey(app_key, app_secret, body_data, url_base):
    url = f"{url_base}/uapi/hashkey"
    headers = {
        'content-type': 'application/json',
        'appkey': app_key,
        'appsecret': app_secret
    }
    res = requests.post(url, headers=headers, data=json.dumps(body_data))
    if res.status_code == 200:
        return res.json()['HASH']
    else:
        print(f"Hashkey 요청 실패: {res.status_code} - {res.text}")
        return None
        

def place_order(stock_code:str, order_side:str, order_type:str, order_price:float, order_quantity:int, account_no:str = None, kis_app_key:str = None, kis_app_secret:str = None, kis_access_token:str = None, **kwargs) -> dict:
    """
    국내주식 모의투자 매수 또는 매도 주문을 실행.

    Parameters:
    - token (str): API 접근 토큰
    - stock_code (str): 주문할 주식의 종목 코드 (예: "005930" - 삼성전자)
    - order_qty (int): 주문 수량 (예: 1주)
    - order_price (float): 주문 단가 (예: 60000원)
    - ord_type (str, 기본값: "buy"): 주문 유형 ("buy" 또는 "sell")

    Returns:
    - dict: 주문 결과를 포함하는 JSON 응답 데이터
    """

    url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-stock/v1/trading/order-cash"

    # 주문 유형에 따라 SLL_BUY_DVSN_CD와 tr_id 설정
    if order_side == "buy":
        tr_id = "VTTC0802U"  # 매수 tr_id
    elif order_side == "sell":
        tr_id = "VTTC0011U"  # 매도 tr_id
    else:
        print("주문 유형이 잘못되었습니다. 'buy' 또는 'sell'을 선택하세요.")
        return "주문 요청 실패"
    
    if order_type == "market":
        order_dvsn = "01"
    elif order_type == "limit":
        order_dvsn = "00"
    else:
        print("주문 유형이 잘못되었습니다. 'market' 또는 'limit'을 선택하세요.")
        return "주문 요청 실패"

    body = {
        "CANO": account_no.split('-')[0],
        "ACNT_PRDT_CD": account_no.split('-')[1],
        "PDNO": stock_code,
        "ORD_DVSN": order_dvsn,  # 00: 지정가 / 01: 시장가
        "ORD_QTY": str(order_quantity),  # 주문수량
        # 시장가 주문은 0, 지정가는 전달된 가격을 문자열로
        "ORD_UNPR": "0" if order_dvsn == "01" else str(order_price),
    }

    # hashkey 생성 (주문 엔드포인트 필수)
    try:
        hashkey = get_hashkey(kis_app_key, kis_app_secret, body, "https://openapivts.koreainvestment.com:29443")
    except Exception as e:
        return f"hashkey 생성 실패: {str(e)}"

    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {kis_access_token}",
        "appkey": kis_app_key,
        "appsecret": kis_app_secret,
        "tr_id": tr_id,  # 모의투자 매수 - VTTC0802U / 모의투자 매도 - VTTC0011U
        "custtype": "P",
        "hashkey": hashkey,
    }
    try:
        res = requests.post(url, headers=headers, data=json.dumps(body), timeout=30)
        res.raise_for_status()
        data = res.json()
        # 표준 메시지 우선 반환하되, 없으면 전체 응답 반환
        return data.get('msg1', data)
    except Exception as e:
        try:
            # 가능한 경우 서버 메시지 노출
            data = res.json()
            return data.get('msg1', str(e))
        except Exception:
            return f"주문 요청 실패: {str(e)}"
    

def custom_add_messages(existing: list, update: list):
    for message in update:
        if not isinstance(message, BaseMessage):
            if message["role"] == "user":
                existing.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                existing.append(AIMessage(content=message["content"]))
            else:
                raise ValueError(f"Invalid message type: {type(message)}")
        else:
            existing.append(message)
    return existing[-10:]
