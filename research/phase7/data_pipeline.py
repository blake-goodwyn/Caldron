"""
Phase 7 — Culinary ML Data Pipeline

Downloads, parses, and normalizes recipe data from public sources.
Produces clean ingredient lists and co-occurrence matrices for ML experiments.
"""

import csv
import json
import re
import ast
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.sparse import lil_matrix, save_npz, load_npz

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

# ── Ingredient normalization ──────────────────────────────────────────────

# Common measurement words to strip
UNITS = {
    "cup", "cups", "tablespoon", "tablespoons", "tbsp", "teaspoon",
    "teaspoons", "tsp", "ounce", "ounces", "oz", "pound", "pounds", "lb",
    "lbs", "gram", "grams", "g", "kg", "kilogram", "ml", "milliliter",
    "liter", "litre", "quart", "pint", "gallon", "dash", "pinch", "bunch",
    "clove", "cloves", "slice", "slices", "piece", "pieces", "can", "cans",
    "package", "packet", "stick", "sticks", "head", "heads", "sprig",
    "sprigs", "handful", "large", "medium", "small", "fresh", "dried",
    "ground", "chopped", "minced", "diced", "sliced", "crushed", "whole",
    "optional", "to", "taste", "for", "garnish", "divided", "softened",
    "melted", "room", "temperature", "finely", "coarsely", "thinly",
    "thick", "thin", "about", "approximately", "plus", "more", "extra",
}

# Fraction unicode chars
FRACTIONS = {"½": 0.5, "⅓": 0.33, "⅔": 0.67, "¼": 0.25, "¾": 0.75,
             "⅛": 0.125, "⅜": 0.375, "⅝": 0.625, "⅞": 0.875}


def normalize_ingredient(raw: str) -> str:
    """Normalize an ingredient string to a canonical name.

    Strips quantities, units, preparation instructions, and punctuation.
    Returns lowercased, whitespace-collapsed ingredient name.
    """
    text = raw.lower().strip()

    # Remove parenthetical notes
    text = re.sub(r"\(.*?\)", "", text)

    # Remove fractions (unicode and ASCII)
    for frac in FRACTIONS:
        text = text.replace(frac, "")
    text = re.sub(r"\d+/\d+", "", text)

    # Remove numbers
    text = re.sub(r"\d+\.?\d*", "", text)

    # Remove punctuation except hyphens (keep compound names like "all-purpose")
    text = re.sub(r"[^\w\s-]", "", text)

    # Tokenize and remove unit/descriptor words
    tokens = [t for t in text.split() if t not in UNITS and len(t) > 1]

    return " ".join(tokens).strip()


# ── RecipeNLG parser ─────────────────────────────────────────────────────

def parse_recipenlg_row(row: dict) -> Optional[dict]:
    """Parse a single RecipeNLG CSV row into a clean recipe dict."""
    try:
        ingredients_raw = ast.literal_eval(row.get("NER", "[]"))
        if not ingredients_raw:
            # Fall back to ingredients column
            ingredients_raw = ast.literal_eval(row.get("ingredients", "[]"))

        ingredients = []
        for ing in ingredients_raw:
            normalized = normalize_ingredient(ing)
            if normalized and len(normalized) > 1:
                ingredients.append(normalized)

        if len(ingredients) < 2:
            return None

        return {
            "title": row.get("title", "").strip(),
            "ingredients": list(dict.fromkeys(ingredients)),  # dedup, preserve order
            "source": row.get("source", ""),
        }
    except (ValueError, SyntaxError):
        return None


def load_recipenlg(path: Path, limit: Optional[int] = None) -> list[dict]:
    """Load and parse RecipeNLG CSV file.

    Args:
        path: Path to full_dataset.csv
        limit: Max recipes to load (None = all)

    Returns:
        List of parsed recipe dicts with normalized ingredients.
    """
    recipes = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            parsed = parse_recipenlg_row(row)
            if parsed:
                recipes.append(parsed)

    logger.info(f"Loaded {len(recipes)} recipes from RecipeNLG")
    return recipes


# ── Vocabulary and co-occurrence ─────────────────────────────────────────

class IngredientVocab:
    """Manages ingredient vocabulary with frequency filtering."""

    def __init__(self, min_count: int = 5):
        self.min_count = min_count
        self.counter: Counter = Counter()
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}

    def fit(self, recipes: list[dict]) -> "IngredientVocab":
        """Build vocabulary from recipe ingredient lists."""
        for recipe in recipes:
            self.counter.update(recipe["ingredients"])

        # Filter by min_count and build index
        idx = 0
        for word, count in self.counter.most_common():
            if count >= self.min_count:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

        logger.info(
            f"Vocabulary: {len(self.word2idx)} ingredients "
            f"(filtered from {len(self.counter)} with min_count={self.min_count})"
        )
        return self

    def encode(self, ingredient: str) -> Optional[int]:
        return self.word2idx.get(ingredient)

    def decode(self, idx: int) -> Optional[str]:
        return self.idx2word.get(idx)

    @property
    def size(self) -> int:
        return len(self.word2idx)

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump({
                "word2idx": self.word2idx,
                "min_count": self.min_count,
                "counts": dict(self.counter.most_common()),
            }, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "IngredientVocab":
        with open(path) as f:
            data = json.load(f)
        vocab = cls(min_count=data["min_count"])
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(v): k for k, v in vocab.word2idx.items()}
        vocab.counter = Counter(data.get("counts", {}))
        return vocab


def build_cooccurrence_matrix(
    recipes: list[dict],
    vocab: IngredientVocab,
    window: Optional[int] = None,
) -> "np.ndarray":
    """Build ingredient co-occurrence matrix from recipes.

    Args:
        recipes: List of recipe dicts with 'ingredients' key.
        vocab: Fitted IngredientVocab.
        window: If None, all ingredients in a recipe co-occur.
                If set, only ingredients within `window` positions co-occur.

    Returns:
        Symmetric co-occurrence matrix (vocab.size × vocab.size).
    """
    n = vocab.size
    matrix = lil_matrix((n, n), dtype=np.float32)

    for recipe in recipes:
        indices = [vocab.encode(ing) for ing in recipe["ingredients"]]
        indices = [i for i in indices if i is not None]

        for i, idx_a in enumerate(indices):
            neighbors = indices if window is None else indices[max(0, i - window):i + window + 1]
            for idx_b in neighbors:
                if idx_a != idx_b:
                    matrix[idx_a, idx_b] += 1

    # Symmetrize
    matrix = matrix + matrix.T
    logger.info(f"Co-occurrence matrix: {n}×{n}, {matrix.nnz} non-zero entries")
    return matrix.tocsr()


# ── Recipe-ingredient binary matrix (for collaborative filtering) ────────

def build_recipe_ingredient_matrix(
    recipes: list[dict],
    vocab: IngredientVocab,
) -> "np.ndarray":
    """Build binary recipe×ingredient matrix for collaborative filtering.

    Returns:
        Sparse binary matrix (num_recipes × vocab.size).
    """
    n_recipes = len(recipes)
    n_ingredients = vocab.size
    matrix = lil_matrix((n_recipes, n_ingredients), dtype=np.float32)

    for i, recipe in enumerate(recipes):
        for ing in recipe["ingredients"]:
            idx = vocab.encode(ing)
            if idx is not None:
                matrix[i, idx] = 1.0

    logger.info(
        f"Recipe-ingredient matrix: {n_recipes}×{n_ingredients}, "
        f"density={matrix.nnz / (n_recipes * n_ingredients):.4f}"
    )
    return matrix.tocsr()


# ── FlavorDB loader ──────────────────────────────────────────────────────

def load_flavordb(path: Path) -> dict[str, list[str]]:
    """Load FlavorDB compound data.

    Expected format: JSON with ingredient → [compound1, compound2, ...] mapping.

    Returns:
        Dict mapping ingredient names to lists of flavor compound names.
    """
    with open(path) as f:
        data = json.load(f)

    # Normalize ingredient names to match our vocab
    normalized = {}
    for ingredient, compounds in data.items():
        key = normalize_ingredient(ingredient)
        if key:
            normalized[key] = compounds

    logger.info(f"Loaded FlavorDB: {len(normalized)} ingredients, "
                f"{len(set(c for cs in normalized.values() for c in cs))} unique compounds")
    return normalized


def compound_overlap_score(
    ing_a: str,
    ing_b: str,
    flavordb: dict[str, list[str]],
) -> float:
    """Jaccard similarity of flavor compounds between two ingredients."""
    compounds_a = set(flavordb.get(ing_a, []))
    compounds_b = set(flavordb.get(ing_b, []))
    if not compounds_a or not compounds_b:
        return 0.0
    return len(compounds_a & compounds_b) / len(compounds_a | compounds_b)


# ── Convenience: full pipeline ───────────────────────────────────────────

def run_pipeline(
    recipenlg_path: Path,
    limit: int = 100_000,
    min_count: int = 10,
    output_dir: Optional[Path] = None,
) -> dict:
    """Run the full data pipeline: load → normalize → build matrices.

    Args:
        recipenlg_path: Path to RecipeNLG full_dataset.csv
        limit: Max recipes to process
        min_count: Minimum ingredient frequency for vocabulary
        output_dir: Where to save outputs (default: DATA_DIR)

    Returns:
        Dict with recipes, vocab, cooccurrence, and recipe_ingredient matrices.
    """
    output_dir = output_dir or DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and parse
    recipes = load_recipenlg(recipenlg_path, limit=limit)

    # Build vocabulary
    vocab = IngredientVocab(min_count=min_count)
    vocab.fit(recipes)
    vocab.save(output_dir / "vocab.json")

    # Build matrices
    cooccurrence = build_cooccurrence_matrix(recipes, vocab)
    recipe_ingredient = build_recipe_ingredient_matrix(recipes, vocab)

    save_npz(output_dir / "cooccurrence.npz", cooccurrence)
    save_npz(output_dir / "recipe_ingredient.npz", recipe_ingredient)

    # Save recipe metadata (titles + ingredient lists)
    meta = [{"title": r["title"], "ingredients": r["ingredients"]} for r in recipes]
    with open(output_dir / "recipes_meta.json", "w") as f:
        json.dump(meta, f)

    logger.info(f"Pipeline complete. Outputs saved to {output_dir}")

    return {
        "recipes": recipes,
        "vocab": vocab,
        "cooccurrence": cooccurrence,
        "recipe_ingredient": recipe_ingredient,
    }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python data_pipeline.py <path_to_recipenlg_csv> [limit]")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000

    run_pipeline(csv_path, limit=limit)
