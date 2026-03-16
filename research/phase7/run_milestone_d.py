"""
Phase 7 -- Milestone D: Knowledge Graph + GNN Experiment

Builds a food knowledge graph, trains RotatE embeddings, trains a GCN
for link prediction, and compares results against food2vec baseline.

Usage:
    python run_milestone_d.py           # full pipeline
    python run_milestone_d.py --kg-only # just build KG + train RotatE
    python run_milestone_d.py --gnn-only # just train GCN (assumes data exists)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import load_npz

sys.path.insert(0, str(Path(__file__).parent))

from data_pipeline import IngredientVocab
from food2vec import Food2Vec
from knowledge_graph import FoodKnowledgeGraph, train_kg_embeddings, KGQueryEngine
from gnn_model import train_gnn, evaluate_gnn_neighbors, compare_with_food2vec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

TEST_INGREDIENTS = [
    "garlic", "butter", "basil", "soy sauce", "cinnamon",
    "chicken", "lemon", "olive oil", "ginger", "chocolate",
    "cumin", "cilantro", "mozzarella", "salmon", "honey",
]


def build_knowledge_graph(recipes, vocab) -> FoodKnowledgeGraph:
    """Build the food knowledge graph from recipe data."""
    print("\n" + "=" * 70)
    print("BUILDING FOOD KNOWLEDGE GRAPH")
    print("=" * 70)

    kg = FoodKnowledgeGraph()

    # Pairing triples from co-occurrence
    kg.add_pairing_triples(recipes, vocab, min_cooccurrence=5)

    # Cuisine triples
    kg.add_cuisine_triples(recipes, vocab)

    # Variant triples
    kg.add_variant_triples(vocab)

    kg.save(DATA_DIR / "food_kg.json")

    print(f"\n  Total triples:   {kg.num_triples:,}")
    print(f"  Total entities:  {len(kg.entities):,}")
    print(f"  Total relations: {len(kg.relations)}")

    # Breakdown by relation type
    from collections import Counter
    rel_counts = Counter(r for _, r, _ in kg.triples)
    for rel, count in rel_counts.most_common():
        print(f"    {rel}: {count:,}")

    return kg


def run_kg_embeddings(kg: FoodKnowledgeGraph) -> dict:
    """Train RotatE embeddings on the knowledge graph."""
    print("\n" + "=" * 70)
    print("TRAINING KG EMBEDDINGS (RotatE)")
    print("=" * 70)

    results = train_kg_embeddings(
        kg,
        model_name="RotatE",
        embedding_dim=64,
        num_epochs=50,
        batch_size=512,
        lr=0.01,
    )

    print(f"\n  Hits@1:  {results['metrics']['hits_at_1']:.3f}")
    print(f"  Hits@3:  {results['metrics']['hits_at_3']:.3f}")
    print(f"  Hits@10: {results['metrics']['hits_at_10']:.3f}")
    print(f"  MRR:     {results['metrics']['mean_reciprocal_rank']:.3f}")
    print(f"  MeanRank:{results['metrics']['mean_rank']:.1f}")

    # Query examples
    print("\n  Sample predictions:")
    engine = KGQueryEngine(results["result"], results["training_factory"])

    queries = [
        ("garlic", "pairs_with"),
        ("chocolate", "pairs_with"),
        ("soy sauce", "pairs_with"),
        ("basil", "same_cuisine"),
    ]

    for head, relation in queries:
        preds = engine.predict_tail(head, relation, topn=5)
        if preds:
            pred_str = ", ".join(f"{t} ({s:.2f})" for t, s in preds[:5])
            print(f"    ({head}, {relation}, ?) -> {pred_str}")

    return results


def run_gnn(vocab, food2vec_model) -> dict:
    """Train GCN on co-occurrence graph."""
    print("\n" + "=" * 70)
    print("TRAINING GCN FOR LINK PREDICTION")
    print("=" * 70)

    cooc = load_npz(DATA_DIR / "cooccurrence.npz")

    gnn_results = train_gnn(
        cooc_matrix=cooc,
        vocab=vocab,
        food2vec_model=food2vec_model,
        hidden_dim=64,
        output_dim=32,
        lr=0.01,
        epochs=200,
        edge_threshold=5.0,
    )

    # Evaluate neighbors
    test_items = [t for t in TEST_INGREDIENTS if vocab.encode(t) is not None]

    print("\n  GNN Nearest Neighbors:")
    gnn_neighbors = evaluate_gnn_neighbors(
        gnn_results["embeddings"], vocab, test_items, topn=5
    )

    # Compare with food2vec
    if food2vec_model:
        print("\n  GNN vs food2vec Comparison:")
        comparison = compare_with_food2vec(
            gnn_results["embeddings"], food2vec_model, vocab, test_items, topn=10
        )

        for ing, comp in comparison.items():
            overlap = comp["overlap_ratio"]
            print(f"    {ing}: {overlap:.0%} overlap with food2vec top-10")

    return gnn_results


def main():
    parser = argparse.ArgumentParser(description="Run Milestone D experiment")
    parser.add_argument("--kg-only", action="store_true", help="Only build KG + train RotatE")
    parser.add_argument("--gnn-only", action="store_true", help="Only train GCN")
    args = parser.parse_args()

    # Load prerequisites
    logger.info("Loading data...")
    vocab = IngredientVocab.load(DATA_DIR / "vocab.json")

    food2vec_path = DATA_DIR / "food2vec.model"
    food2vec_model = Food2Vec.load(food2vec_path) if food2vec_path.exists() else None

    with open(DATA_DIR / "recipes_meta.json") as f:
        recipes = json.load(f)

    results = {}

    if not args.gnn_only:
        # Build KG + train embeddings
        kg = build_knowledge_graph(recipes, vocab)
        kg_results = run_kg_embeddings(kg)
        results["kg_metrics"] = kg_results["metrics"]

    if not args.kg_only:
        # Train GCN
        gnn_results = run_gnn(vocab, food2vec_model)
        results["gnn_history"] = gnn_results["history"]

        # Save GNN embeddings
        torch.save(gnn_results["embeddings"], DATA_DIR / "gnn_embeddings.pt")
        torch.save(gnn_results["model"].state_dict(), DATA_DIR / "gnn_model.pt")

    # Save combined results
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(DATA_DIR / "milestone_d_results.json", "w") as f:
        json.dump(results, f, indent=2, default=convert)

    print("\n" + "=" * 70)
    print("MILESTONE D COMPLETE")
    print("=" * 70)
    print(f"  Results saved to: {DATA_DIR / 'milestone_d_results.json'}")


if __name__ == "__main__":
    main()
