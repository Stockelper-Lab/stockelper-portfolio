from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import create_async_engine

from multi_agent.utils import (
    get_access_token,
    get_user_kis_credentials,
    update_user_kis_credentials,
)


_ENGINE = None


def _to_async_db_url(url: Optional[str]) -> Optional[str]:
    """Convert Prisma-style DATABASE_URL to SQLAlchemy async URL if needed."""
    if not url:
        return None
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


def _get_engine():
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    async_db_url = os.getenv("ASYNC_DATABASE_URL") or _to_async_db_url(
        os.getenv("DATABASE_URL")
    )
    if not async_db_url:
        raise RuntimeError("ASYNC_DATABASE_URL 또는 DATABASE_URL 이 설정되어 있지 않습니다.")
    # NOTE: DB가 idle 커넥션을 끊는 경우, 풀에서 꺼낸 커넥션이 closed일 수 있어 pre-ping을 켭니다.
    _ENGINE = create_async_engine(async_db_url, echo=False, pool_pre_ping=True)
    return _ENGINE


class InputState(BaseModel):
    user_id: int = Field(description="사용자 ID")


class OutputState(BaseModel):
    kis_app_key: str
    kis_app_secret: str
    kis_access_token: str
    account_no: str


class LoadUserContext:
    """DB(stockelper_web.user)에서 사용자 KIS 자격증명을 로드하고 토큰을 발급/저장합니다."""

    name = "LoadUserContext"

    async def __call__(self, state: InputState) -> OutputState:
        engine = _get_engine()
        user_info = await get_user_kis_credentials(engine, state.user_id)
        if not user_info:
            raise ValueError(f"user_id={state.user_id} 사용자를 DB에서 찾지 못했습니다.")

        if not user_info.get("account_no"):
            raise ValueError("user.account_no가 비어있습니다.")

        # DB에 저장된 토큰이 있으면 그대로 사용하고,
        # 없으면 app_key/app_secret으로 access_token을 발급받아 DB에 저장합니다.
        access_token = user_info.get("kis_access_token")
        if not access_token:
            access_token = await get_access_token(
                user_info["kis_app_key"], user_info["kis_app_secret"]
            )
            if not access_token:
                raise ValueError("KIS access token 발급 실패 (app_key/app_secret 확인 필요)")
            await update_user_kis_credentials(engine, state.user_id, access_token)

        return {
            "kis_app_key": user_info["kis_app_key"],
            "kis_app_secret": user_info["kis_app_secret"],
            "kis_access_token": access_token,
            "account_no": user_info["account_no"],
        }


