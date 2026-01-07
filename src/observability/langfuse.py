from __future__ import annotations

import os
from typing import Any


def _is_langfuse_configured() -> bool:
    """Langfuse는 키가 없으면 동작하지 않도록 '옵트인'으로 둡니다."""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def build_langfuse_runnable_config(
    *,
    run_name: str,
    trace_id: str | None = None,
    user_id: str | int | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    update_trace: bool = True,
) -> dict[str, Any]:
    """Langfuse 트레이싱을 위한 RunnableConfig를 생성합니다.

    - Langfuse 설정이 없으면(키 미설정) 빈 config를 반환합니다.
    - Langfuse CallbackHandler는 `metadata`에 아래 키가 있으면 trace 속성으로 승격합니다.
      - `langfuse_user_id`: str
      - `langfuse_session_id`: str
      - `langfuse_tags`: list[str]
    """
    if not _is_langfuse_configured():
        return {"run_name": run_name} if run_name else {}

    # 런타임에서 langfuse가 설치되지 않았더라도 서버가 죽지 않게 처리합니다.
    try:
        from langfuse.langchain import CallbackHandler
    except Exception:
        return {"run_name": run_name} if run_name else {}

    handler_kwargs: dict[str, Any] = {"update_trace": update_trace}
    if trace_id:
        handler_kwargs["trace_context"] = {"trace_id": trace_id}

    handler = CallbackHandler(**handler_kwargs)

    merged_metadata: dict[str, Any] = dict(metadata or {})
    if session_id:
        merged_metadata["langfuse_session_id"] = str(session_id)
    if user_id is not None:
        merged_metadata["langfuse_user_id"] = str(user_id)
    if tags:
        merged_metadata["langfuse_tags"] = list(tags)

    config: dict[str, Any] = {
        "callbacks": [handler],
        "metadata": merged_metadata,
    }
    if tags:
        config["tags"] = list(tags)
    if run_name:
        config["run_name"] = run_name

    return config


