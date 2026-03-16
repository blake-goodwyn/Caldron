"""
Phase 7 — food2vec: Ingredient Embedding Model

Trains Word2Vec on recipe ingredient lists to learn dense ingredient
representations where co-occurring ingredients are close in vector space.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


class Food2Vec:
    """Ingredient embedding model based on Word2Vec.

    Treats each recipe's ingredient list as a "sentence" and each
    ingredient as a "word". Learns dense vectors where culinary
    partners are close in embedding space.
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 10,
        min_count: int = 5,
        epochs: int = 30,
        seed: int = 42,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.seed = seed
        self.model = None

    def train(self, recipes: list[dict]) -> "Food2Vec":
        """Train Word2Vec on recipe ingredient lists.

        Args:
            recipes: List of recipe dicts, each with an 'ingredients' key.
        """
        from gensim.models import Word2Vec

        sentences = [recipe["ingredients"] for recipe in recipes]

        logger.info(
            f"Training food2vec: {len(sentences)} recipes, "
            f"dim={self.vector_size}, window={self.window}, epochs={self.epochs}"
        )

        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            seed=self.seed,
            workers=4,
            sg=1,  # Skip-gram (better for small datasets)
        )

        logger.info(f"Trained on {len(self.model.wv)} ingredients")
        return self

    def most_similar(self, ingredient: str, topn: int = 10) -> list[tuple[str, float]]:
        """Find most similar ingredients by embedding distance."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        try:
            return self.model.wv.most_similar(ingredient, topn=topn)
        except KeyError:
            logger.warning(f"'{ingredient}' not in vocabulary")
            return []

    def similarity(self, ing_a: str, ing_b: str) -> float:
        """Cosine similarity between two ingredient embeddings."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        try:
            return float(self.model.wv.similarity(ing_a, ing_b))
        except KeyError:
            return 0.0

    def analogy(
        self,
        positive: list[str],
        negative: list[str],
        topn: int = 5,
    ) -> list[tuple[str, float]]:
        """Ingredient analogy: positive - negative ≈ ?

        Example: analogy(["soy sauce", "french"], ["asian"]) → "worcestershire"
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        try:
            return self.model.wv.most_similar(
                positive=positive, negative=negative, topn=topn
            )
        except KeyError as e:
            logger.warning(f"Analogy failed: {e}")
            return []

    def get_vector(self, ingredient: str) -> Optional[np.ndarray]:
        """Get the embedding vector for an ingredient."""
        if self.model is None:
            return None
        try:
            return self.model.wv[ingredient]
        except KeyError:
            return None

    @property
    def vocabulary(self) -> list[str]:
        """All ingredients in the model vocabulary."""
        if self.model is None:
            return []
        return list(self.model.wv.key_to_index.keys())

    def save(self, path: Path):
        """Save trained model."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        self.model.save(str(path))
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "Food2Vec":
        """Load a trained model."""
        from gensim.models import Word2Vec
        instance = cls()
        instance.model = Word2Vec.load(str(path))
        logger.info(f"Model loaded from {path}")
        return instance


# ── Evaluation utilities ─────────────────────────────────────────────────

def evaluate_neighbors(
    model: Food2Vec,
    test_ingredients: list[str],
    topn: int = 10,
) -> dict[str, list[tuple[str, float]]]:
    """Get nearest neighbors for a list of test ingredients.

    Returns dict mapping each ingredient to its neighbors + scores.
    """
    results = {}
    for ing in test_ingredients:
        neighbors = model.most_similar(ing, topn=topn)
        results[ing] = neighbors
        if neighbors:
            neighbor_str = ", ".join(f"{n} ({s:.2f})" for n, s in neighbors[:5])
            logger.info(f"  {ing}: {neighbor_str}")
    return results


def evaluate_affinity_vs_compounds(
    model: Food2Vec,
    flavordb: dict[str, list[str]],
    sample_size: int = 200,
    topn: int = 10,
) -> dict:
    """Evaluate: do food2vec neighbors share more flavor compounds?

    Compares compound overlap for top-N neighbors vs. random pairs.
    """
    from data_pipeline import compound_overlap_score

    # Get ingredients that exist in both model and FlavorDB
    shared = [ing for ing in model.vocabulary if ing in flavordb]
    if len(shared) < 20:
        logger.warning(f"Only {len(shared)} shared ingredients — results may be unreliable")

    rng = np.random.RandomState(42)
    sample = rng.choice(shared, size=min(sample_size, len(shared)), replace=False)

    neighbor_overlaps = []
    random_overlaps = []

    for ing in sample:
        neighbors = model.most_similar(ing, topn=topn)
        for neighbor, _ in neighbors:
            if neighbor in flavordb:
                score = compound_overlap_score(ing, neighbor, flavordb)
                neighbor_overlaps.append(score)

        # Random comparison
        randoms = rng.choice(shared, size=topn, replace=False)
        for rand_ing in randoms:
            if rand_ing != ing:
                score = compound_overlap_score(ing, rand_ing, flavordb)
                random_overlaps.append(score)

    result = {
        "mean_neighbor_overlap": float(np.mean(neighbor_overlaps)) if neighbor_overlaps else 0,
        "mean_random_overlap": float(np.mean(random_overlaps)) if random_overlaps else 0,
        "n_neighbor_pairs": len(neighbor_overlaps),
        "n_random_pairs": len(random_overlaps),
        "lift": 0.0,
    }
    if result["mean_random_overlap"] > 0:
        result["lift"] = result["mean_neighbor_overlap"] / result["mean_random_overlap"]

    logger.info(
        f"Compound overlap — neighbors: {result['mean_neighbor_overlap']:.4f}, "
        f"random: {result['mean_random_overlap']:.4f}, "
        f"lift: {result['lift']:.2f}x"
    )
    return result


# ── Visualization ────────────────────────────────────────────────────────

def plot_embeddings(
    model: Food2Vec,
    ingredients: Optional[list[str]] = None,
    n_ingredients: int = 200,
    method: str = "umap",
    output_path: Optional[Path] = None,
):
    """Visualize ingredient embeddings with dimensionality reduction.

    Args:
        model: Trained Food2Vec model.
        ingredients: Specific ingredients to plot. If None, uses top-N by frequency.
        n_ingredients: Number of ingredients if `ingredients` is None.
        method: "umap" or "tsne".
        output_path: Save plot to file. If None, displays interactively.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if ingredients is None:
        ingredients = model.vocabulary[:n_ingredients]

    vectors = []
    labels = []
    for ing in ingredients:
        vec = model.get_vector(ing)
        if vec is not None:
            vectors.append(vec)
            labels.append(ing)

    vectors = np.array(vectors)

    if method == "umap":
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)

    coords = reducer.fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.scatter(coords[:, 0], coords[:, 1], s=10, alpha=0.6)

    # Label a subset to avoid clutter
    step = max(1, len(labels) // 50)
    for i in range(0, len(labels), step):
        ax.annotate(labels[i], (coords[i, 0], coords[i, 1]),
                    fontsize=7, alpha=0.8)

    ax.set_title(f"food2vec Ingredient Embeddings ({method.upper()})")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to {output_path}")
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python food2vec.py train <recipenlg_csv> [limit]")
        print("  python food2vec.py similar <ingredient>")
        print("  python food2vec.py eval")
        sys.exit(1)

    cmd = sys.argv[1]
    model_path = DATA_DIR / "food2vec.model"

    if cmd == "train":
        from data_pipeline import load_recipenlg

        csv_path = Path(sys.argv[2])
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else 100_000

        recipes = load_recipenlg(csv_path, limit=limit)
        model = Food2Vec(vector_size=100, window=10, min_count=10, epochs=30)
        model.train(recipes)
        model.save(model_path)

        # Quick evaluation
        test_items = [
            "garlic", "butter", "basil", "soy sauce", "cinnamon",
            "chicken", "lemon", "olive oil", "ginger", "cream cheese",
            "cumin", "coconut milk", "mozzarella", "rice", "honey",
            "salmon", "avocado", "cilantro", "paprika", "chocolate",
        ]
        print("\n── Nearest Neighbors ──")
        evaluate_neighbors(model, test_items, topn=5)

    elif cmd == "similar":
        model = Food2Vec.load(model_path)
        ingredient = " ".join(sys.argv[2:])
        neighbors = model.most_similar(ingredient, topn=10)
        print(f"\nMost similar to '{ingredient}':")
        for name, score in neighbors:
            print(f"  {name:30s} {score:.4f}")

    elif cmd == "eval":
        model = Food2Vec.load(model_path)
        print(f"Vocabulary size: {len(model.vocabulary)}")
        print(f"Vector dimensionality: {model.vector_size}")

        test_items = [
            "garlic", "butter", "basil", "soy sauce", "cinnamon",
            "chicken", "lemon", "olive oil", "ginger", "chocolate",
        ]
        print("\n── Nearest Neighbors ──")
        evaluate_neighbors(model, test_items, topn=5)
