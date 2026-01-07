from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig


class JsonOutputParser(StrOutputParser):
    response_schema: object

    def __init__(self, response_schema: object):
        super().__init__(response_schema=response_schema)

    def parse(self, llm_output: str):
        text = super().parse(llm_output)

        # </think> 이전의 모든 문자열 제거
        if "</think>" in text:
            think_pattern = r".*</think>"
            text = re.sub(think_pattern, "", text, flags=re.DOTALL)

        if "```json" in text:
            pattern = r"```json(.*?)```"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                text = match.group(1)

        text = text.replace("\\'", "'")
        structured_output = json.loads(text)
        self.response_schema.model_validate(structured_output)
        return structured_output


def with_runnable_config(
    config: RunnableConfig | None,
    *,
    run_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> RunnableConfig | None:
    """기존 RunnableConfig에 run_name/metadata/tags를 안전하게 합쳐 반환합니다.

    - 원본 config를 변형하지 않습니다.
    - config가 없고 추가할 값도 없으면 None을 그대로 반환합니다.
    """
    if config is None and run_name is None and not metadata and not tags:
        return None

    merged: dict[str, Any] = dict(config or {})

    if run_name:
        merged["run_name"] = run_name

    if metadata:
        md = dict(merged.get("metadata") or {})
        md.update(metadata)
        merged["metadata"] = md

    if tags:
        existing = list(merged.get("tags") or [])
        for t in tags:
            if t not in existing:
                existing.append(t)
        merged["tags"] = existing

    return merged