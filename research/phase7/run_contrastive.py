"""
Phase 7 -- Run Contrastive Relationship Classifier Experiment

Mines training pairs from food2vec + CF, trains the MLP classifier,
and evaluates on held-out pairs.

Usage:
    python run_contrastive.py
"""

import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from data_pipeline import IngredientVocab
from food2vec import Food2Vec
from affinity_models import IngredientCF
from contrastive_model import (
    mine_training_pairs,
    train_classifier,
    classify_relationship,
    LABELS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


def main():
    # Load models
    logger.info("Loading models...")
    vocab = IngredientVocab.load(DATA_DIR / "vocab.json")
    food2vec = Food2Vec.load(DATA_DIR / "food2vec.model")

    from scipy.sparse import load_npz
    ri_matrix = load_npz(DATA_DIR / "recipe_ingredient.npz")
    cf = IngredientCF(n_neighbors=20)
    cf.fit(ri_matrix, vocab)

    # Mine training pairs
    print("\n" + "=" * 70)
    print("MINING TRAINING PAIRS")
    print("=" * 70)
    pairs = mine_training_pairs(
        food2vec, cf, vocab,
        f2v_threshold=0.45,
        cf_threshold=0.25,
        max_pairs_per_class=1500,
    )

    if not pairs:
        print("No pairs mined. Exiting.")
        return

    # Show samples
    from collections import Counter
    label_counts = Counter(label for _, _, label in pairs)
    print(f"\n  Total pairs: {len(pairs)}")
    for label, count in label_counts.most_common():
        print(f"    {label}: {count}")

    print("\n  Sample substitutes:")
    for a, b, l in pairs[:20]:
        if l == "substitute":
            print(f"    {a} <-> {b}")

    print("\n  Sample pairings:")
    for a, b, l in pairs[:50]:
        if l == "pairs_with":
            print(f"    {a} + {b}")

    # Train classifier
    print("\n" + "=" * 70)
    print("TRAINING CLASSIFIER")
    print("=" * 70)
    result = train_classifier(
        pairs, food2vec,
        embedding_dim=100,
        hidden_dim=128,
        epochs=50,
        batch_size=64,
    )

    if result["model"] is None:
        print("Training failed.")
        return

    # Save model
    torch.save(result["model"].state_dict(), DATA_DIR / "relation_classifier.pt")

    # Print metrics
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n  Weighted F1: {result['f1']:.3f}")
    for label in LABELS:
        if label in result["metrics"]:
            m = result["metrics"][label]
            print(f"  {label:15s}: precision={m['precision']:.3f}, "
                  f"recall={m['recall']:.3f}, f1={m['f1-score']:.3f}")

    # Spot-check predictions
    print("\n" + "=" * 70)
    print("SPOT-CHECK PREDICTIONS")
    print("=" * 70)
    test_pairs = [
        ("butter", "margarine", "should be substitute"),
        ("garlic", "butter", "should be pairing"),
        ("chocolate", "salmon", "should be unrelated"),
        ("olive oil", "vegetable oil", "should be substitute"),
        ("basil", "oregano", "could be either"),
        ("soy sauce", "oyster sauce", "should be substitute"),
        ("cinnamon", "sugar", "should be pairing"),
        ("lemon", "lime", "should be substitute"),
    ]

    model = result["model"]
    for a, b, expected in test_pairs:
        pred = classify_relationship(model, food2vec, a, b)
        status = pred["relationship"]
        conf = pred["confidence"]
        print(f"  {a} + {b}: {status} ({conf:.2f}) -- {expected}")

    # Save results
    save_results = {
        "f1": result["f1"],
        "metrics": {k: v for k, v in result["metrics"].items() if isinstance(v, dict)},
        "history": result["history"],
        "n_pairs": len(pairs),
    }
    with open(DATA_DIR / "contrastive_results.json", "w") as f:
        json.dump(save_results, f, indent=2)

    print(f"\n  Results saved to {DATA_DIR / 'contrastive_results.json'}")


if __name__ == "__main__":
    main()
