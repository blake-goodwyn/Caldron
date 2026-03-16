"""Production ML service layer for Caldron.

Provides lazy-loaded, thread-safe access to trained ML models for
ingredient substitution, recipe completion, and affinity scoring.
"""

import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Add research modules to path
_RESEARCH_DIR = os.path.join(os.path.dirname(__file__), '..', 'research', 'phase7')
if _RESEARCH_DIR not in sys.path:
    sys.path.insert(0, _RESEARCH_DIR)


class CulinaryMLService:
    """Singleton service providing ML-backed culinary intelligence.

    Models are loaded lazily on first access and cached. Thread-safe
    for use in FastAPI async contexts. Degrades gracefully when models
    are unavailable.
    """

    _instance: Optional["CulinaryMLService"] = None
    _lock = threading.Lock()

    def __new__(cls, models_dir: Optional[str] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self, models_dir: Optional[str] = None):
        if self._initialized:
            return
        from config import ML_MODELS_DIR, ML_ENABLED
        self._models_dir = Path(models_dir or ML_MODELS_DIR)
        self._enabled = ML_ENABLED
        self._food2vec = None
        self._cf = None
        self._vocab = None
        self._canonical_map = None
        self._model_lock = threading.Lock()
        self._initialized = True

    @classmethod
    def reset(cls):
        """Reset singleton for testing."""
        cls._instance = None

    @property
    def available(self) -> bool:
        """Check if ML models are available."""
        if not self._enabled:
            return False
        return (self._models_dir / "food2vec.model").exists()

    def _load_vocab(self):
        if self._vocab is None:
            with self._model_lock:
                if self._vocab is None:
                    from data_pipeline import IngredientVocab
                    vocab_path = self._models_dir / "vocab.json"
                    if vocab_path.exists():
                        self._vocab = IngredientVocab.load(vocab_path)
                        logger.info(f"Loaded vocab: {self._vocab.size} ingredients")
                    else:
                        logger.warning(f"Vocab not found: {vocab_path}")
        return self._vocab

    def _load_canonical_map(self):
        if self._canonical_map is None:
            with self._model_lock:
                if self._canonical_map is None:
                    from vocab_canonicalize import CanonicalMap
                    cmap_path = self._models_dir / "canonical_map.json"
                    if cmap_path.exists():
                        self._canonical_map = CanonicalMap.load(cmap_path)
                    else:
                        # Return empty map — no canonicalization
                        self._canonical_map = CanonicalMap()
        return self._canonical_map

    def _load_food2vec(self):
        if self._food2vec is None:
            with self._model_lock:
                if self._food2vec is None:
                    from food2vec import Food2Vec
                    model_path = self._models_dir / "food2vec.model"
                    if model_path.exists():
                        self._food2vec = Food2Vec.load(model_path)
                        logger.info(f"Loaded food2vec: {len(self._food2vec.vocabulary)} ingredients")
                    else:
                        logger.warning(f"food2vec model not found: {model_path}")
        return self._food2vec

    def _load_cf(self):
        if self._cf is None:
            with self._model_lock:
                if self._cf is None:
                    vocab = self._load_vocab()
                    if vocab is None:
                        return None
                    ri_path = self._models_dir / "recipe_ingredient.npz"
                    if ri_path.exists():
                        from scipy.sparse import load_npz
                        from affinity_models import IngredientCF
                        ri_matrix = load_npz(ri_path)
                        self._cf = IngredientCF(n_neighbors=20)
                        self._cf.fit(ri_matrix, vocab)
                        logger.info("Loaded collaborative filtering model")
                    else:
                        logger.warning(f"Recipe-ingredient matrix not found: {ri_path}")
        return self._cf

    def _normalize(self, ingredient: str) -> str:
        """Normalize and canonicalize an ingredient name."""
        from data_pipeline import normalize_ingredient
        cmap = self._load_canonical_map()
        return normalize_ingredient(ingredient, canonical_map=cmap)

    def suggest_substitutions(
        self, ingredient: str, n: int = 5
    ) -> list[dict]:
        """Find ingredient substitutions using food2vec embeddings.

        Args:
            ingredient: Ingredient to find substitutes for.
            n: Number of suggestions.

        Returns:
            List of {"name": str, "score": float, "source": "food2vec"} dicts.
        """
        if not self._enabled:
            return []

        model = self._load_food2vec()
        if model is None:
            return []

        normalized = self._normalize(ingredient)
        if not normalized:
            return []

        neighbors = model.most_similar(normalized, topn=n)
        return [
            {"name": name, "score": round(score, 4), "source": "food2vec"}
            for name, score in neighbors
        ]

    def complete_recipe(
        self, ingredients: list[str], n: int = 5
    ) -> list[dict]:
        """Suggest ingredients to complete a recipe using collaborative filtering.

        Args:
            ingredients: Current ingredient list.
            n: Number of suggestions.

        Returns:
            List of {"name": str, "score": float, "source": "collaborative_filtering"} dicts.
        """
        if not self._enabled:
            return []

        cf = self._load_cf()
        if cf is None:
            return []

        normalized = [self._normalize(ing) for ing in ingredients]
        normalized = [n for n in normalized if n]

        if len(normalized) < 1:
            return []

        suggestions = cf.suggest_ingredients(normalized, topn=n)
        return [
            {"name": name, "score": round(score, 4), "source": "collaborative_filtering"}
            for name, score in suggestions
        ]

    def score_affinity(self, ing_a: str, ing_b: str) -> dict:
        """Score how well two ingredients pair together.

        Args:
            ing_a: First ingredient.
            ing_b: Second ingredient.

        Returns:
            {"score": float, "food2vec_score": float, "source": "combined"} dict.
        """
        if not self._enabled:
            return {"score": 0.0, "food2vec_score": 0.0, "source": "unavailable"}

        model = self._load_food2vec()
        if model is None:
            return {"score": 0.0, "food2vec_score": 0.0, "source": "unavailable"}

        a = self._normalize(ing_a)
        b = self._normalize(ing_b)
        if not a or not b:
            return {"score": 0.0, "food2vec_score": 0.0, "source": "unknown_ingredient"}

        f2v_score = model.similarity(a, b)
        return {
            "score": round(f2v_score, 4),
            "food2vec_score": round(f2v_score, 4),
            "source": "food2vec",
        }
