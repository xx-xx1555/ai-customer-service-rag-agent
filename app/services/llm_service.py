import json
import logging
import re
from typing import Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

from app.core.config import settings

logger = logging.getLogger(__name__)


def llm_available() -> bool:
    return bool(OpenAI is not None and settings.DEEPSEEK_API_KEY)


def get_llm_client() -> Optional["OpenAI"]:
    if not llm_available():
        return None
    return OpenAI(api_key=settings.DEEPSEEK_API_KEY, base_url=settings.DEEPSEEK_BASE_URL)


def chat_completion(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    """统一封装 LLM 调用，后续要换 OpenAI/通义/智谱，只改这里。"""
    client = get_llm_client()
    if client is None:
        return ""

    try:
        response = client.chat.completions.create(
            model=settings.DEEPSEEK_MODEL,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:  # 线上项目应该细分超时、限流、认证错误
        logger.exception("LLM 调用失败：%s", exc)
        return ""


def extract_json(text: str) -> Dict:
    """从模型输出中提取 JSON，避免模型偶尔包一层 markdown。"""
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```json\s*(.*?)\s*```", text, re.S)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return {}

    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    return {}


def json_chat_completion(messages: List[Dict[str, str]], temperature: float = 0.0) -> Dict:
    content = chat_completion(messages=messages, temperature=temperature)
    return extract_json(content)
