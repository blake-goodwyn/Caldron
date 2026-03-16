"""
Phase 7 — Knowledge Graph Construction & Embedding

Builds a food knowledge graph from recipe data and FlavorDB compounds,
then trains KG embeddings (RotatE) via PyKEEN for multi-relational
ingredient reasoning.
"""

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


# ── Relation types ───────────────────────────────────────────────────────

RELATIONS = [
    "pairs_with",        # ingredients that co-occur frequently
    "same_cuisine",      # ingredients common to the same cuisine cluster
    "same_technique",    # ingredients that share cooking techniques
    "shares_compound",   # ingredients that share flavor compounds (FlavorDB)
    "variant_of",        # ingredient variants (e.g., butter / unsalted butter)
    "cooked_by",         # ingredient → technique (directional)
    "shares_flavor_profile",  # ingredients sharing flavor descriptors
]


# ── Cuisine detection (heuristic from ingredient clusters) ───────────────

CUISINE_MARKERS = {
    "asian": ["soy sauce", "sesame oil", "ginger", "rice vinegar", "hoisin sauce",
              "oyster sauce", "fish sauce", "tofu", "bamboo shoots", "water chestnuts",
              "teriyaki sauce", "miso", "sake", "nori", "wasabi"],
    "italian": ["basil", "oregano", "mozzarella", "parmesan", "prosciutto",
                "olive oil", "balsamic vinegar", "capers", "romano", "ricotta",
                "pesto", "marinara", "polenta", "arborio rice"],
    "mexican": ["cilantro", "cumin", "jalapeno", "tortilla", "black beans",
                "avocado", "salsa", "chipotle", "queso", "lime",
                "enchilada sauce", "taco seasoning", "cotija"],
    "indian": ["turmeric", "garam masala", "cumin", "coriander", "cardamom",
               "curry", "ghee", "basmati", "naan", "paneer",
               "chutney", "tamarind", "fenugreek"],
    "french": ["shallot", "tarragon", "dijon mustard", "cream", "wine",
               "butter", "thyme", "bouquet garni", "cognac", "gruyere",
               "brie", "herbes de provence"],
    "southern_us": ["buttermilk", "cornmeal", "okra", "collard greens",
                    "grits", "fatback", "bourbon", "pecan", "sweet potato"],
    "baking": ["flour", "sugar", "baking powder", "baking soda", "vanilla",
               "yeast", "shortening", "confectioners sugar", "cream of tartar"],
}


def detect_cuisines(recipe_ingredients: list[str]) -> list[str]:
    """Detect likely cuisine(s) for a recipe based on marker ingredients."""
    scores = {}
    for cuisine, markers in CUISINE_MARKERS.items():
        matches = sum(1 for ing in recipe_ingredients if any(m in ing for m in markers))
        if matches >= 2:
            scores[cuisine] = matches
    return sorted(scores, key=scores.get, reverse=True)


# ── Knowledge Graph builder ─────────────────────────────────────────────

class FoodKnowledgeGraph:
    """Builds and manages a food knowledge graph as (head, relation, tail) triples."""

    def __init__(self):
        self.triples: list[tuple[str, str, str]] = []
        self._entity_set: set[str] = set()
        self._relation_set: set[str] = set()

    @property
    def entities(self) -> list[str]:
        return sorted(self._entity_set)

    @property
    def relations(self) -> list[str]:
        return sorted(self._relation_set)

    @property
    def num_triples(self) -> int:
        return len(self.triples)

    def add_triple(self, head: str, relation: str, tail: str):
        """Add a single triple to the graph."""
        self.triples.append((head, relation, tail))
        self._entity_set.update([head, tail])
        self._relation_set.add(relation)

    def add_pairing_triples(
        self,
        recipes: list[dict],
        vocab,
        min_cooccurrence: int = 10,
    ):
        """Add pairs_with triples from ingredient co-occurrence.

        Only adds pairs that co-occur in at least `min_cooccurrence` recipes.
        """
        pair_counts: Counter = Counter()
        for recipe in recipes:
            ingredients = [
                ing for ing in recipe["ingredients"]
                if vocab.encode(ing) is not None
            ]
            for i, a in enumerate(ingredients):
                for b in ingredients[i + 1:]:
                    pair = tuple(sorted([a, b]))
                    pair_counts[pair] += 1

        added = 0
        for (a, b), count in pair_counts.items():
            if count >= min_cooccurrence:
                self.add_triple(a, "pairs_with", b)
                added += 1

        logger.info(f"Added {added} pairs_with triples (min_cooccurrence={min_cooccurrence})")

    def add_cuisine_triples(self, recipes: list[dict], vocab):
        """Add same_cuisine triples for ingredients sharing a cuisine context."""
        cuisine_ingredients: dict[str, set[str]] = defaultdict(set)

        for recipe in recipes:
            ingredients = [
                ing for ing in recipe["ingredients"]
                if vocab.encode(ing) is not None
            ]
            cuisines = detect_cuisines(ingredients)
            for cuisine in cuisines:
                for ing in ingredients:
                    cuisine_ingredients[cuisine].add(ing)

        # Add cuisine entity nodes and link ingredients
        added = 0
        for cuisine, ings in cuisine_ingredients.items():
            cuisine_entity = f"cuisine:{cuisine}"
            for ing in ings:
                self.add_triple(ing, "same_cuisine", cuisine_entity)
                added += 1

        logger.info(f"Added {added} same_cuisine triples across {len(cuisine_ingredients)} cuisines")

    def add_technique_triples(self, recipes: list[dict], vocab):
        """Add same_technique triples from recipe instructions."""
        from affinity_models import extract_techniques_from_instructions

        technique_ingredients: dict[str, set[str]] = defaultdict(set)

        for recipe in recipes:
            instructions = recipe.get("instructions", "")
            if not instructions:
                continue
            techniques = extract_techniques_from_instructions(instructions)
            ingredients = [
                ing for ing in recipe["ingredients"]
                if vocab.encode(ing) is not None
            ]
            for tech in techniques:
                for ing in ingredients:
                    technique_ingredients[tech].add(ing)

        added = 0
        for tech, ings in technique_ingredients.items():
            tech_entity = f"technique:{tech}"
            for ing in ings:
                self.add_triple(ing, "same_technique", tech_entity)
                added += 1

        logger.info(f"Added {added} same_technique triples across {len(technique_ingredients)} techniques")

    def add_cooked_by_triples(self, recipes: list[dict], vocab, min_count: int = 5):
        """Add cooked_by triples: ingredient → technique (directional).

        Only adds triples where an ingredient-technique pair appears in
        at least `min_count` recipes.
        """
        from affinity_models import extract_techniques_from_instructions
        from collections import Counter

        pair_counts: Counter = Counter()

        for recipe in recipes:
            directions = recipe.get("directions", "")
            if not directions:
                continue
            techniques = extract_techniques_from_instructions(directions)
            ingredients = [
                ing for ing in recipe["ingredients"]
                if vocab.encode(ing) is not None
            ]
            for ing in ingredients:
                for tech in techniques:
                    pair_counts[(ing, tech)] += 1

        added = 0
        for (ing, tech), count in pair_counts.items():
            if count >= min_count:
                self.add_triple(ing, "cooked_by", f"technique:{tech}")
                added += 1

        logger.info(f"Added {added} cooked_by triples (min_count={min_count})")

    def add_compound_triples(self, flavordb: dict[str, list[str]], vocab):
        """Add shares_compound triples from FlavorDB data."""
        # Build compound -> ingredients mapping
        compound_to_ings: dict[str, set[str]] = defaultdict(set)
        for ing, compounds in flavordb.items():
            if vocab.encode(ing) is not None:
                for compound in compounds:
                    compound_to_ings[compound].add(ing)

        # Add triples for ingredients sharing compounds
        added = 0
        for compound, ings in compound_to_ings.items():
            ings_list = sorted(ings)
            if len(ings_list) < 2:
                continue
            compound_entity = f"compound:{compound}"
            for ing in ings_list:
                self.add_triple(ing, "shares_compound", compound_entity)
                added += 1

        logger.info(f"Added {added} shares_compound triples")

    def add_variant_triples(self, vocab):
        """Add variant_of triples for ingredient variants.

        Heuristic: if ingredient A's name is a substring of ingredient B,
        they may be variants (e.g., 'butter' and 'unsalted butter').
        """
        ingredients = [vocab.idx2word[i] for i in range(vocab.size)]
        added = 0

        for i, a in enumerate(ingredients):
            for b in ingredients[i + 1:]:
                # Check if one is a variant of the other
                if a != b and len(a) > 2 and len(b) > 2:
                    if a in b and len(b) - len(a) < 15:
                        self.add_triple(b, "variant_of", a)
                        added += 1
                    elif b in a and len(a) - len(b) < 15:
                        self.add_triple(a, "variant_of", b)
                        added += 1

        logger.info(f"Added {added} variant_of triples")

    def to_triples_factory(self):
        """Convert to PyKEEN TriplesFactory for training."""
        from pykeen.triples import TriplesFactory

        # Convert to numpy array of strings
        triples_array = np.array(self.triples, dtype=str)

        factory = TriplesFactory.from_labeled_triples(triples_array)
        logger.info(
            f"TriplesFactory: {factory.num_entities} entities, "
            f"{factory.num_relations} relations, {factory.num_triples} triples"
        )
        return factory

    def save(self, path: Path):
        """Save knowledge graph to JSON."""
        data = {
            "triples": self.triples,
            "num_entities": len(self._entity_set),
            "num_relations": len(self._relation_set),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"KG saved to {path}: {len(self.triples)} triples")

    @classmethod
    def load(cls, path: Path) -> "FoodKnowledgeGraph":
        """Load knowledge graph from JSON."""
        with open(path) as f:
            data = json.load(f)
        kg = cls()
        for h, r, t in data["triples"]:
            kg.add_triple(h, r, t)
        return kg


# ── KG Embedding training ───────────────────────────────────────────────

def train_kg_embeddings(
    kg: FoodKnowledgeGraph,
    model_name: str = "RotatE",
    embedding_dim: int = 64,
    num_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 0.01,
    output_dir: Optional[Path] = None,
) -> dict:
    """Train KG embeddings using PyKEEN.

    Args:
        kg: Constructed FoodKnowledgeGraph.
        model_name: PyKEEN model name (RotatE, TransE, ComplEx).
        embedding_dim: Embedding dimensionality.
        num_epochs: Training epochs.
        batch_size: Training batch size.
        lr: Learning rate.
        output_dir: Where to save model and results.

    Returns:
        Dict with model, results, and training metrics.
    """
    from pykeen.pipeline import pipeline

    output_dir = output_dir or DATA_DIR / "kg_model"

    factory = kg.to_triples_factory()

    # Split into train/test
    training, testing = factory.split([0.8, 0.2], random_state=42)

    logger.info(
        f"Training {model_name} with dim={embedding_dim}, "
        f"epochs={num_epochs}, lr={lr}"
    )

    result = pipeline(
        training=training,
        testing=testing,
        model=model_name,
        model_kwargs={"embedding_dim": embedding_dim},
        optimizer_kwargs={"lr": lr},
        training_kwargs={
            "num_epochs": num_epochs,
            "batch_size": batch_size,
        },
        random_seed=42,
        device="cpu",
    )

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    result.save_to_directory(str(output_dir))
    logger.info(f"KG model saved to {output_dir}")

    # Extract metrics
    metrics = {
        "hits_at_1": float(result.metric_results.get_metric("hits_at_1")),
        "hits_at_3": float(result.metric_results.get_metric("hits_at_3")),
        "hits_at_10": float(result.metric_results.get_metric("hits_at_10")),
        "mean_rank": float(result.metric_results.get_metric("mean_rank")),
        "mean_reciprocal_rank": float(result.metric_results.get_metric("mean_reciprocal_rank")),
    }

    logger.info(
        f"Results: Hits@10={metrics['hits_at_10']:.3f}, "
        f"MRR={metrics['mean_reciprocal_rank']:.3f}, "
        f"MeanRank={metrics['mean_rank']:.1f}"
    )

    return {
        "result": result,
        "metrics": metrics,
        "training_factory": training,
        "testing_factory": testing,
    }


# ── KG query utilities ──────────────────────────────────────────────────

class KGQueryEngine:
    """Query trained KG embeddings for ingredient relationships."""

    def __init__(self, result, factory):
        self.model = result.model
        self.factory = factory

    def predict_tail(
        self, head: str, relation: str, topn: int = 10
    ) -> list[tuple[str, float]]:
        """Predict: (head, relation, ?) -> ranked tails."""
        from pykeen.predict import predict_target

        try:
            predictions = predict_target(
                model=self.model,
                head=head,
                relation=relation,
                triples_factory=self.factory,
            )
            df = predictions.df
            results = []
            for _, row in df.head(topn).iterrows():
                results.append((row["tail_label"], float(row["score"])))
            return results
        except (KeyError, ValueError) as e:
            logger.warning(f"Prediction failed for ({head}, {relation}, ?): {e}")
            return []

    def predict_relation(
        self, head: str, tail: str, topn: int = 5
    ) -> list[tuple[str, float]]:
        """Predict: (head, ?, tail) -> ranked relations."""
        from pykeen.predict import predict_target

        try:
            predictions = predict_target(
                model=self.model,
                head=head,
                tail=tail,
                triples_factory=self.factory,
            )
            df = predictions.df
            results = []
            for _, row in df.head(topn).iterrows():
                results.append((row["relation_label"], float(row["score"])))
            return results
        except (KeyError, ValueError) as e:
            logger.warning(f"Prediction failed for ({head}, ?, {tail}): {e}")
            return []

    def get_entity_embedding(self, entity: str) -> Optional[np.ndarray]:
        """Get the learned embedding vector for an entity."""
        try:
            entity_id = self.factory.entity_to_id[entity]
            with torch.no_grad():
                emb = self.model.entity_representations[0](
                    torch.tensor([entity_id])
                )
            return emb.numpy().flatten()
        except (KeyError, IndexError):
            return None


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python knowledge_graph.py build   -- build KG from recipe data")
        print("  python knowledge_graph.py train   -- train RotatE embeddings")
        print("  python knowledge_graph.py query <head> <relation>  -- predict tails")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "build":
        from data_pipeline import IngredientVocab

        # Load vocab and recipes
        vocab = IngredientVocab.load(DATA_DIR / "vocab.json")
        with open(DATA_DIR / "recipes_meta.json") as f:
            recipes = json.load(f)

        # Build KG
        kg = FoodKnowledgeGraph()
        kg.add_pairing_triples(recipes, vocab, min_cooccurrence=5)
        kg.add_cuisine_triples(recipes, vocab)
        kg.add_variant_triples(vocab)

        kg.save(DATA_DIR / "food_kg.json")
        print(f"\nKG built: {kg.num_triples} triples, "
              f"{len(kg.entities)} entities, {len(kg.relations)} relations")

    elif cmd == "train":
        kg = FoodKnowledgeGraph.load(DATA_DIR / "food_kg.json")
        results = train_kg_embeddings(kg, num_epochs=100)
        print(f"\nTraining complete. Metrics: {results['metrics']}")

    elif cmd == "query":
        head = sys.argv[2]
        relation = sys.argv[3]
        print(f"\nPredicting ({head}, {relation}, ?)...")
        # Load and query would go here
