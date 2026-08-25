import os
from typing import Any, Dict, Optional

import requests


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
DEFAULT_TIMEOUT = int(os.getenv("API_TIMEOUT", "60"))


class ApiError(RuntimeError):
    pass


def _request(method: str, path: str, *, timeout: int = DEFAULT_TIMEOUT, **kwargs) -> Any:
    url = f"{API_BASE_URL}{path}"
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        raise ApiError(f"无法连接后端：{exc}") from exc

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ApiError(f"后端返回 {response.status_code}：{detail}")

    if response.status_code == 204:
        return None
    return response.json()


def get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    return _request("GET", path, params=params)


def post(path: str, json: Optional[Dict[str, Any]] = None, files=None, timeout: int = DEFAULT_TIMEOUT) -> Any:
    return _request("POST", path, json=json, files=files, timeout=timeout)


def patch(path: str, json: Dict[str, Any]) -> Any:
    return _request("PATCH", path, json=json)


def delete(path: str) -> None:
    _request("DELETE", path)
