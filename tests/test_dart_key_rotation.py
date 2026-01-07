from __future__ import annotations

import os

import pytest

from multi_agent.dart import is_dart_quota_exceeded_error, parse_open_dart_api_keys


def test_parse_open_dart_api_keys_from_plural(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPEN_DART_API_KEYS", "k1,k2  k3\nk2")
    monkeypatch.delenv("OPEN_DART_API_KEY", raising=False)
    assert parse_open_dart_api_keys() == ["k1", "k2", "k3"]


def test_parse_open_dart_api_keys_from_singular(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPEN_DART_API_KEYS", raising=False)
    monkeypatch.setenv("OPEN_DART_API_KEY", "k1,k2")
    assert parse_open_dart_api_keys() == ["k1", "k2"]


def test_is_dart_quota_exceeded_error_dict():
    assert is_dart_quota_exceeded_error({"status": "020", "message": "사용한도를 초과하였습니다."}) is True


def test_is_dart_quota_exceeded_error_stringified_dict():
    err = Exception("{'status': '020', 'message': '사용한도를 초과하였습니다.'}")
    assert is_dart_quota_exceeded_error(err) is True


