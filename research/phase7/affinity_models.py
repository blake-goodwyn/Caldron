"""
Phase 7 — Affinity Models: Collaborative Filtering + NMF

Collaborative filtering on recipe×ingredient matrix and NMF on
ingredient×technique matrix to discover latent culinary patterns.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.sparse import load_npz, csr_matrix
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


# ── Collaborative Filtering ─────────────────────────────────────────────

class IngredientCF:
    """Item-based collaborative filtering for ingredient affinity.

    Treats recipes as "users" and ingredients as "items".
    Learns which ingredients tend to co-occur across recipes.
    """

    def __init__(self, n_neighbors: int = 20, metric: str = "cosine"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self._nn: Optional[NearestNeighbors] = None
        self._matrix: Optional[csr_matrix] = None
        self._vocab = None

    def fit(self, recipe_ingredient_matrix: csr_matrix, vocab) -> "IngredientCF":
        """Fit the model on a recipe×ingredient matrix.

        Args:
            recipe_ingredient_matrix: Sparse binary matrix (recipes × ingredients).
            vocab: IngredientVocab instance for index↔name mapping.
        """
        self._matrix = recipe_ingredient_matrix
        self._vocab = vocab

        # Item-based: transpose so ingredients are rows
        item_matrix = recipe_ingredient_matrix.T.tocsr()

        self._nn = NearestNeighbors(
            n_neighbors=min(self.n_neighbors + 1, item_matrix.shape[0]),
            metric=self.metric,
            algorithm="brute",
        )
        self._nn.fit(item_matrix)

        logger.info(
            f"CF model fitted: {item_matrix.shape[0]} ingredients, "
            f"{item_matrix.shape[1]} recipes"
        )
        return self

    def similar_ingredients(
        self, ingredient: str, topn: int = 10
    ) -> list[tuple[str, float]]:
        """Find ingredients with similar co-occurrence patterns."""
        idx = self._vocab.encode(ingredient)
        if idx is None:
            logger.warning(f"'{ingredient}' not in vocabulary")
            return []

        item_matrix = self._matrix.T.tocsr()
        n_neighbors = min(topn + 1, item_matrix.shape[0])
        distances, indices = self._nn.kneighbors(
            item_matrix[idx].toarray(), n_neighbors=n_neighbors
        )

        results = []
        for dist, neighbor_idx in zip(distances[0], indices[0]):
            name = self._vocab.decode(int(neighbor_idx))
            if name and name != ingredient:
                # Convert distance to similarity
                sim = 1.0 - dist if self.metric == "cosine" else 1.0 / (1.0 + dist)
                results.append((name, float(sim)))

        return results[:topn]

    def suggest_ingredients(
        self,
        current_ingredients: list[str],
        topn: int = 10,
    ) -> list[tuple[str, float]]:
        """Given a partial ingredient list, suggest what else belongs.

        Aggregates neighbor scores across all current ingredients.
        """
        scores: dict[str, float] = {}

        for ing in current_ingredients:
            neighbors = self.similar_ingredients(ing, topn=20)
            for name, sim in neighbors:
                if name not in current_ingredients:
                    scores[name] = scores.get(name, 0) + sim

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:topn]


# ── NMF Technique-Ingredient Decomposition ───────────────────────────────

class TechniqueNMF:
    """Non-negative Matrix Factorization on ingredient×technique matrix.

    Discovers latent culinary patterns — clusters of ingredients that
    share cooking techniques.
    """

    def __init__(self, n_components: int = 20, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self._model: Optional[NMF] = None
        self._W: Optional[np.ndarray] = None  # ingredient loadings
        self._H: Optional[np.ndarray] = None  # technique loadings
        self._ingredient_names: list[str] = []
        self._technique_names: list[str] = []

    def fit(
        self,
        matrix: np.ndarray,
        ingredient_names: list[str],
        technique_names: list[str],
    ) -> "TechniqueNMF":
        """Fit NMF on an ingredient×technique co-occurrence matrix.

        Args:
            matrix: Dense or sparse matrix (ingredients × techniques).
            ingredient_names: Row labels.
            technique_names: Column labels.
        """
        self._ingredient_names = ingredient_names
        self._technique_names = technique_names

        self._model = NMF(
            n_components=self.n_components,
            random_state=self.random_state,
            max_iter=500,
        )
        self._W = self._model.fit_transform(matrix)
        self._H = self._model.components_

        logger.info(
            f"NMF fitted: {len(ingredient_names)} ingredients × "
            f"{len(technique_names)} techniques → {self.n_components} components, "
            f"reconstruction error: {self._model.reconstruction_err_:.2f}"
        )
        return self

    def inspect_components(self, top_n: int = 10) -> list[dict]:
        """Show top ingredients and techniques for each component.

        Returns list of dicts, one per component.
        """
        components = []
        for k in range(self.n_components):
            # Top ingredients for this component
            ing_scores = self._W[:, k]
            top_ing_idx = np.argsort(ing_scores)[::-1][:top_n]
            top_ingredients = [
                (self._ingredient_names[i], float(ing_scores[i]))
                for i in top_ing_idx
            ]

            # Top techniques for this component
            tech_scores = self._H[k, :]
            top_tech_idx = np.argsort(tech_scores)[::-1][:top_n]
            top_techniques = [
                (self._technique_names[i], float(tech_scores[i]))
                for i in top_tech_idx
            ]

            components.append({
                "component": k,
                "top_ingredients": top_ingredients,
                "top_techniques": top_techniques,
            })

        return components

    def ingredient_embedding(self, ingredient: str) -> Optional[np.ndarray]:
        """Get the NMF latent vector for an ingredient."""
        if ingredient in self._ingredient_names:
            idx = self._ingredient_names.index(ingredient)
            return self._W[idx]
        return None

    def similar_by_technique_profile(
        self, ingredient: str, topn: int = 10
    ) -> list[tuple[str, float]]:
        """Find ingredients with similar technique profiles."""
        vec = self.ingredient_embedding(ingredient)
        if vec is None:
            return []

        sims = cosine_similarity(vec.reshape(1, -1), self._W)[0]
        top_idx = np.argsort(sims)[::-1][1:topn + 1]  # skip self

        return [
            (self._ingredient_names[i], float(sims[i]))
            for i in top_idx
        ]


# ── Technique extraction from recipe text ────────────────────────────────

COOKING_TECHNIQUES = [
    "bake", "roast", "grill", "broil", "fry", "deep fry", "saute",
    "stir fry", "pan fry", "boil", "simmer", "poach", "steam",
    "braise", "stew", "smoke", "blanch", "sear", "toast", "caramelize",
    "reduce", "deglaze", "emulsify", "whisk", "fold", "knead",
    "marinate", "cure", "ferment", "pickle", "blend", "puree",
    "chop", "dice", "mince", "julienne", "zest", "grate", "cream",
    "whip", "beat", "melt", "proof", "rest",
]


def extract_techniques_from_instructions(instructions: str) -> list[str]:
    """Extract cooking techniques mentioned in recipe instructions."""
    import unicodedata
    # Normalize accented chars (é → e) for matching
    instructions_normalized = unicodedata.normalize("NFKD", instructions.lower())
    instructions_normalized = "".join(
        c for c in instructions_normalized if not unicodedata.combining(c)
    )
    found = []
    for technique in COOKING_TECHNIQUES:
        import re
        if re.search(rf"\b{re.escape(technique)}\w*\b", instructions_normalized):
            found.append(technique)
    return found


def build_ingredient_technique_matrix(
    recipes: list[dict],
    vocab,
    instructions_key: str = "instructions",
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build ingredient×technique co-occurrence matrix from recipes.

    Args:
        recipes: List of recipe dicts with 'ingredients' and 'instructions'.
        vocab: IngredientVocab for filtering.
        instructions_key: Key for recipe instructions text.

    Returns:
        (matrix, ingredient_names, technique_names)
    """
    technique_counts: dict[tuple[str, str], int] = {}

    for recipe in recipes:
        instructions = recipe.get(instructions_key, "")
        if not instructions:
            continue

        techniques = extract_techniques_from_instructions(instructions)
        ingredients = recipe.get("ingredients", [])

        for ing in ingredients:
            if vocab.encode(ing) is not None:
                for tech in techniques:
                    key = (ing, tech)
                    technique_counts[key] = technique_counts.get(key, 0) + 1

    # Build matrix
    ingredient_names = sorted(set(k[0] for k in technique_counts))
    technique_names = sorted(set(k[1] for k in technique_counts))

    ing_idx = {name: i for i, name in enumerate(ingredient_names)}
    tech_idx = {name: i for i, name in enumerate(technique_names)}

    matrix = np.zeros((len(ingredient_names), len(technique_names)), dtype=np.float32)
    for (ing, tech), count in technique_counts.items():
        matrix[ing_idx[ing], tech_idx[tech]] = count

    logger.info(
        f"Ingredient×technique matrix: {matrix.shape[0]}×{matrix.shape[1]}, "
        f"{np.count_nonzero(matrix)} non-zero entries"
    )
    return matrix, ingredient_names, technique_names


# ── Combined affinity score ──────────────────────────────────────────────

class CombinedAffinity:
    """Combines food2vec similarity + CF score for ingredient affinity."""

    def __init__(self, food2vec, cf_model: IngredientCF, alpha: float = 0.5):
        """
        Args:
            food2vec: Trained Food2Vec model.
            cf_model: Trained IngredientCF model.
            alpha: Weight for food2vec (1-alpha for CF).
        """
        self.food2vec = food2vec
        self.cf = cf_model
        self.alpha = alpha

    def affinity(self, ing_a: str, ing_b: str) -> float:
        """Combined affinity score between two ingredients."""
        f2v_score = self.food2vec.similarity(ing_a, ing_b)

        # Get CF similarity (search through neighbors)
        cf_neighbors = self.cf.similar_ingredients(ing_a, topn=50)
        cf_score = 0.0
        for name, sim in cf_neighbors:
            if name == ing_b:
                cf_score = sim
                break

        return self.alpha * f2v_score + (1 - self.alpha) * cf_score

    def top_affinities(
        self, ingredient: str, topn: int = 10
    ) -> list[tuple[str, float]]:
        """Get top affinity scores combining both models."""
        # Get candidates from both models
        f2v_neighbors = dict(self.food2vec.most_similar(ingredient, topn=30))
        cf_neighbors = dict(self.cf.similar_ingredients(ingredient, topn=30))

        # Union of candidates
        all_candidates = set(f2v_neighbors) | set(cf_neighbors)
        scores = []
        for candidate in all_candidates:
            f2v = f2v_neighbors.get(candidate, 0.0)
            cf = cf_neighbors.get(candidate, 0.0)
            combined = self.alpha * f2v + (1 - self.alpha) * cf
            scores.append((candidate, combined))

        scores.sort(key=lambda x: -x[1])
        return scores[:topn]


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("Affinity models module loaded. Use from notebooks or pipeline.")
    print("Available: IngredientCF, TechniqueNMF, CombinedAffinity")
