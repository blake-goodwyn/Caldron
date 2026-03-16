"""
Phase 7 -- Run Milestone A+B: Data Pipeline + food2vec Training

Loads RecipeNLG from HuggingFace, builds vocabulary and matrices,
trains food2vec embeddings, and evaluates results.

Usage:
    python run_experiment.py                   # lite dataset (~7K recipes)
    python run_experiment.py --full            # full dataset (~2.2M recipes)
    python run_experiment.py --full --limit N  # full dataset, first N recipes
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from data_pipeline import (
    IngredientVocab,
    build_cooccurrence_matrix,
    build_recipe_ingredient_matrix,
    normalize_ingredient,
)
from food2vec import Food2Vec, evaluate_neighbors
from affinity_models import IngredientCF, CombinedAffinity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


def load_from_huggingface(full: bool = False, limit: int = None) -> list[dict]:
    """Load RecipeNLG from HuggingFace datasets."""
    from datasets import load_dataset

    dataset_name = "Zappandy/recipe_nlg"
    logger.info(f"Loading RecipeNLG from HuggingFace ({dataset_name})...")
    if not full and not limit:
        limit = 50_000  # Default lite mode to 50K recipes
    ds = load_dataset(dataset_name, split="train")

    recipes = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break

        # NER column is comma-separated string of ingredient names
        ner_raw = row.get("NER", "") or ""
        ner = [n.strip() for n in ner_raw.split(",") if n.strip()] if ner_raw else []
        title = row.get("title", "") or ""

        # Normalize ingredients from NER
        ingredients = []
        for ing in ner:
            normalized = normalize_ingredient(ing)
            if normalized and len(normalized) > 1:
                ingredients.append(normalized)

        # Deduplicate, preserve order
        ingredients = list(dict.fromkeys(ingredients))

        # Capture directions for technique extraction
        directions_raw = row.get("directions", "") or ""
        # Zappandy uses <extra_id_99> as step delimiter
        directions = directions_raw.replace("<extra_id_99>", "\n").strip()

        if len(ingredients) >= 2:
            recipes.append({
                "title": title,
                "ingredients": ingredients,
                "directions": directions,
            })

    logger.info(f"Loaded {len(recipes)} valid recipes")
    return recipes


def run_pipeline(recipes: list[dict], min_count: int = 3) -> dict:
    """Run data pipeline: build vocab + matrices."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Build vocabulary
    vocab = IngredientVocab(min_count=min_count)
    vocab.fit(recipes)
    vocab.save(DATA_DIR / "vocab.json")

    # Build matrices
    cooc = build_cooccurrence_matrix(recipes, vocab)
    ri_matrix = build_recipe_ingredient_matrix(recipes, vocab)

    from scipy.sparse import save_npz
    save_npz(DATA_DIR / "cooccurrence.npz", cooc)
    save_npz(DATA_DIR / "recipe_ingredient.npz", ri_matrix)

    # Save recipe metadata (including directions for technique extraction)
    meta = [{"title": r["title"], "ingredients": r["ingredients"], "directions": r.get("directions", "")} for r in recipes]
    with open(DATA_DIR / "recipes_meta.json", "w") as f:
        json.dump(meta, f)

    return {"vocab": vocab, "cooccurrence": cooc, "recipe_ingredient": ri_matrix}


def train_food2vec(recipes: list[dict], min_count: int = 3) -> Food2Vec:
    """Train food2vec model."""
    model = Food2Vec(
        vector_size=100,
        window=10,
        min_count=min_count,
        epochs=30,
        seed=42,
    )
    model.train(recipes)
    model.save(DATA_DIR / "food2vec.model")
    return model


def train_cf(ri_matrix, vocab) -> IngredientCF:
    """Train collaborative filtering model."""
    cf = IngredientCF(n_neighbors=20)
    cf.fit(ri_matrix, vocab)
    return cf


def evaluate_all(model: Food2Vec, cf: IngredientCF, vocab) -> dict:
    """Run evaluations on trained models."""
    results = {}

    # Test ingredients for neighbor evaluation
    test_items = [
        "garlic", "butter", "basil", "soy sauce", "cinnamon",
        "chicken", "lemon", "olive oil", "ginger", "cream cheese",
        "cumin", "coconut milk", "mozzarella", "rice", "honey",
        "salmon", "avocado", "cilantro", "paprika", "chocolate",
    ]
    # Filter to ingredients in vocabulary
    test_items = [t for t in test_items if t in model.vocabulary]

    # 1. food2vec nearest neighbors
    print("\n" + "=" * 70)
    print("FOOD2VEC -- NEAREST NEIGHBORS")
    print("=" * 70)
    neighbors = evaluate_neighbors(model, test_items, topn=5)
    results["food2vec_neighbors"] = {
        k: [(n, round(s, 4)) for n, s in v] for k, v in neighbors.items()
    }

    # 2. Analogy tests
    print("\n" + "=" * 70)
    print("FOOD2VEC -- ANALOGY TESTS")
    print("=" * 70)
    analogies = [
        # (positive, negative, description)
        (["soy sauce", "italian"], ["asian"], "soy sauce - asian + italian = ?"),
        (["butter", "asian"], ["european"], "butter - european + asian = ?"),
        (["cinnamon", "savory"], ["sweet"], "cinnamon - sweet + savory = ?"),
        (["chicken", "vegetarian"], ["meat"], "chicken - meat + vegetarian = ?"),
    ]

    analogy_results = []
    for positive, negative, desc in analogies:
        # Only try if all terms are in vocab
        all_terms = positive + negative
        if all(t in model.vocabulary for t in all_terms):
            result = model.analogy(positive, negative, topn=3)
            print(f"  {desc}")
            for name, score in result:
                print(f"    -> {name} ({score:.3f})")
            analogy_results.append({"query": desc, "results": result})
        else:
            missing = [t for t in all_terms if t not in model.vocabulary]
            print(f"  {desc} -- skipped (missing: {missing})")

    results["analogies"] = analogy_results

    # 3. CF suggestions
    print("\n" + "=" * 70)
    print("COLLABORATIVE FILTERING -- INGREDIENT SUGGESTIONS")
    print("=" * 70)
    test_combos = [
        ["garlic", "butter", "parsley"],
        ["soy sauce", "ginger", "rice"],
        ["chocolate", "butter", "sugar"],
        ["tomato", "basil", "mozzarella"],
    ]

    cf_results = []
    for combo in test_combos:
        valid_combo = [c for c in combo if vocab.encode(c) is not None]
        if len(valid_combo) >= 2:
            suggestions = cf.suggest_ingredients(valid_combo, topn=5)
            print(f"\n  Given: {valid_combo}")
            for name, score in suggestions:
                print(f"    -> {name} ({score:.3f})")
            cf_results.append({"given": valid_combo, "suggestions": suggestions})

    results["cf_suggestions"] = cf_results

    # 4. Combined affinity
    print("\n" + "=" * 70)
    print("COMBINED AFFINITY -- TOP PAIRINGS")
    print("=" * 70)
    combined = CombinedAffinity(model, cf, alpha=0.5)

    pair_tests = ["garlic", "chocolate", "salmon", "basil"]
    pair_tests = [p for p in pair_tests if p in model.vocabulary and vocab.encode(p) is not None]

    combined_results = []
    for ing in pair_tests:
        top = combined.top_affinities(ing, topn=5)
        print(f"\n  {ing}:")
        for name, score in top:
            print(f"    -> {name} ({score:.3f})")
        combined_results.append({"ingredient": ing, "top_affinities": top})

    results["combined_affinities"] = combined_results

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Phase 7 ML experiment")
    parser.add_argument("--full", action="store_true", help="Use full 2.2M dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of recipes")
    args = parser.parse_args()

    # Determine min_count based on dataset size
    if args.full:
        min_count = 10
    else:
        min_count = 3  # Lower threshold for lite dataset

    # Step 1: Load data
    recipes = load_from_huggingface(full=args.full, limit=args.limit)

    # Step 2: Build pipeline artifacts
    logger.info("Building vocabulary and matrices...")
    pipeline = run_pipeline(recipes, min_count=min_count)

    # Step 3: Train food2vec
    logger.info("Training food2vec...")
    model = train_food2vec(recipes, min_count=min_count)

    # Step 4: Train collaborative filtering
    logger.info("Training collaborative filtering...")
    cf = train_cf(pipeline["recipe_ingredient"], pipeline["vocab"])

    # Step 5: Evaluate
    logger.info("Running evaluations...")
    results = evaluate_all(model, cf, pipeline["vocab"])

    # Save results
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(DATA_DIR / "experiment_results.json", "w") as f:
        json.dump(results, f, indent=2, default=convert)

    print("\n" + "=" * 70)
    print(f"SUMMARY")
    print("=" * 70)
    print(f"  Recipes processed:     {len(recipes):,}")
    print(f"  Vocabulary size:       {pipeline['vocab'].size:,}")
    print(f"  food2vec vocab:        {len(model.vocabulary):,}")
    print(f"  Vector dimensions:     {model.vector_size}")
    print(f"  Results saved to:      {DATA_DIR / 'experiment_results.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
