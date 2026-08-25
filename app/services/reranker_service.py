import logging
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class RerankerService:
    """Cross-Encoder 二阶段排序器；加载失败时可按配置回退到基础检索排序。"""

    def __init__(self) -> None:
        self._model = None
        self._load_error: str | None = None

    @property
    def enabled(self) -> bool:
        return settings.RERANKER_ENABLED

    @property
    def model_name(self) -> str:
        return settings.RERANKER_MODEL

    def _get_model(self):
        if not self.enabled:
            return None
        if self._model is not None:
            return self._model
        if self._load_error and settings.RERANKER_FAIL_OPEN:
            return None

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                device=settings.RERANKER_DEVICE,
            )
            return self._model
        except Exception as exc:  # 模型下载、显存、依赖错误统一降级
            self._load_error = str(exc)
            logger.exception("Reranker 加载失败，将回退到基础检索排序：%s", exc)
            if settings.RERANKER_FAIL_OPEN:
                return None
            raise RuntimeError(f"Reranker 加载失败：{exc}") from exc

    def score_pairs(self, question: str, passages: Sequence[str]) -> np.ndarray:
        model = self._get_model()
        if model is None or not passages:
            return np.asarray([], dtype=np.float32)

        pairs = [[question, passage] for passage in passages]
        scores = model.predict(
            pairs,
            batch_size=settings.RERANKER_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(scores, dtype=np.float32).reshape(-1)

    @staticmethod
    def _normalize(scores: Iterable[float]) -> List[float]:
        values = np.asarray(list(scores), dtype=np.float32)
        if values.size == 0:
            return []
        min_value = float(values.min())
        max_value = float(values.max())
        if abs(max_value - min_value) < 1e-9:
            return [1.0 for _ in values]
        return [float((value - min_value) / (max_value - min_value)) for value in values]

    @staticmethod
    def _build_passage(candidate: Dict) -> str:
        parts = []

        source = str(candidate.get("source") or "").strip()
        title = str(candidate.get("title") or "").strip()
        content = str(candidate.get("content") or "").strip()

        if source:
            parts.append(f"来源：{source}")
        if title:
            parts.append(f"标题：{title}")
        if content:
            parts.append(f"内容：{content}")

        return "\n".join(parts)

    def rerank(self, question: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        if not candidates:
            return []

        base_candidates = [dict(candidate) for candidate in candidates]
        if not self.enabled:
            return self._fallback(base_candidates, top_k, reason="disabled")

        passages = [self._build_passage(candidate) for candidate in base_candidates]
        raw_scores = self.score_pairs(question, passages)
        if len(raw_scores) != len(base_candidates):
            return self._fallback(base_candidates, top_k, reason="unavailable")

        normalized_scores = self._normalize(raw_scores)
        rerank_weight = min(max(settings.RERANKER_WEIGHT, 0.0), 1.0)

        for candidate, raw_score, normalized_score in zip(
            base_candidates,
            raw_scores,
            normalized_scores,
        ):
            base_score = float(candidate.get("base_score", candidate.get("final_score", 0.0)))
            candidate["base_score"] = round(base_score, 4)
            candidate["rerank_raw_score"] = round(float(raw_score), 4)
            candidate["rerank_score"] = round(normalized_score, 4)
            candidate["final_score"] = round(
                rerank_weight * normalized_score + (1 - rerank_weight) * base_score,
                4,
            )
            candidate["rerank_applied"] = True
            candidate["retrieval_mode"] = f"{candidate.get('retrieval_mode', 'hybrid')}+rerank"

        base_candidates.sort(key=lambda item: item["final_score"], reverse=True)
        return base_candidates[:top_k]

    @staticmethod
    def _fallback(candidates: List[Dict], top_k: int, reason: str) -> List[Dict]:
        for candidate in candidates:
            base_score = float(candidate.get("base_score", candidate.get("final_score", 0.0)))
            candidate["base_score"] = round(base_score, 4)
            candidate["rerank_score"] = 0.0
            candidate["rerank_raw_score"] = None
            candidate["rerank_applied"] = False
            candidate["rerank_fallback_reason"] = reason
            candidate["final_score"] = round(base_score, 4)
        candidates.sort(key=lambda item: item["final_score"], reverse=True)
        return candidates[:top_k]


@lru_cache(maxsize=1)
def get_reranker_service() -> RerankerService:
    return RerankerService()
