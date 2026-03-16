"""
Phase 7 -- Contrastive Relationship Classifier

Learns to distinguish 'substitute', 'pairs_with', and 'unrelated'
relationships between ingredients using a lightweight MLP on top of
frozen food2vec embeddings.

Key insight: substitutes have high embedding similarity but low
co-occurrence (butter/margarine), while pairings have high co-occurrence
(garlic/butter).
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

LABELS = ["substitute", "pairs_with", "unrelated"]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}

DATA_DIR = Path(__file__).parent / "data"


# ── Pair mining ──────────────────────────────────────────────────────────

def mine_training_pairs(
    food2vec,
    cf,
    vocab,
    f2v_threshold: float = 0.5,
    cf_threshold: float = 0.3,
    max_pairs_per_class: int = 2000,
    seed: int = 42,
) -> list[tuple[str, str, str]]:
    """Mine labeled training pairs from food2vec and CF models.

    Heuristic labeling:
    - SUBSTITUTE: high food2vec similarity + low CF co-occurrence
      (butter/margarine: similar but rarely in same recipe)
    - PAIRS_WITH: high CF co-occurrence
      (garlic/butter: always together)
    - UNRELATED: low scores on both

    Args:
        food2vec: Trained Food2Vec model.
        cf: Trained IngredientCF model.
        vocab: IngredientVocab.
        f2v_threshold: Min food2vec similarity for substitute candidates.
        cf_threshold: CF score boundary between substitute and pairing.
        max_pairs_per_class: Max pairs per class to keep balanced.
        seed: Random seed.

    Returns:
        List of (ingredient_a, ingredient_b, label) tuples.
    """
    rng = np.random.RandomState(seed)

    # Get all ingredients with embeddings
    ingredients = [
        ing for ing in food2vec.vocabulary
        if vocab.encode(ing) is not None
    ]

    substitutes = []
    pairings = []
    unrelated = []

    # Build CF neighbor lookup for faster access
    cf_neighbors_cache = {}

    for ing in ingredients:
        # Get food2vec neighbors
        f2v_neighbors = food2vec.most_similar(ing, topn=20)

        # Get CF neighbors
        if ing not in cf_neighbors_cache:
            cf_neighbors_cache[ing] = dict(cf.similar_ingredients(ing, topn=30))
        cf_scores = cf_neighbors_cache[ing]

        for neighbor, f2v_score in f2v_neighbors:
            if neighbor not in food2vec.vocabulary:
                continue

            cf_score = cf_scores.get(neighbor, 0.0)

            if f2v_score >= f2v_threshold and cf_score < cf_threshold:
                # High similarity, low co-occurrence -> substitute
                substitutes.append((ing, neighbor, "substitute"))
            elif cf_score >= cf_threshold:
                # High co-occurrence -> pairing
                pairings.append((ing, neighbor, "pairs_with"))

    # Sample unrelated pairs (low on both signals)
    n_unrelated = max(max_pairs_per_class, len(substitutes), len(pairings))
    attempts = 0
    while len(unrelated) < n_unrelated and attempts < n_unrelated * 10:
        a = rng.choice(ingredients)
        b = rng.choice(ingredients)
        attempts += 1
        if a == b:
            continue
        f2v_score = food2vec.similarity(a, b)
        if f2v_score < 0.2:
            unrelated.append((a, b, "unrelated"))

    # Balance classes
    n = min(max_pairs_per_class, len(substitutes), len(pairings), len(unrelated))
    if n == 0:
        logger.warning("No training pairs mined. Check thresholds.")
        return []

    rng.shuffle(substitutes)
    rng.shuffle(pairings)
    rng.shuffle(unrelated)

    pairs = substitutes[:n] + pairings[:n] + unrelated[:n]
    rng.shuffle(pairs)

    logger.info(
        f"Mined {len(pairs)} training pairs: "
        f"{n} substitutes, {n} pairings, {n} unrelated"
    )
    return pairs


# ── Dataset ──────────────────────────────────────────────────────────────

class IngredientPairDataset(Dataset):
    """Dataset of ingredient pairs with relationship labels."""

    def __init__(self, pairs: list[tuple[str, str, str]], food2vec):
        self.pairs = []
        self.labels = []
        self.food2vec = food2vec

        for a, b, label in pairs:
            vec_a = food2vec.get_vector(a)
            vec_b = food2vec.get_vector(b)
            if vec_a is not None and vec_b is not None:
                self.pairs.append((vec_a, vec_b))
                self.labels.append(LABEL_TO_IDX[label])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        vec_a, vec_b = self.pairs[idx]
        # Concatenate embeddings + element-wise difference + product
        combined = np.concatenate([
            vec_a, vec_b,
            vec_a - vec_b,
            vec_a * vec_b,
        ])
        return (
            torch.tensor(combined, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ── Classifier ───────────────────────────────────────────────────────────

class RelationClassifier(nn.Module):
    """2-layer MLP for ingredient relationship classification.

    Input: concatenated food2vec embeddings (4 * embedding_dim features).
    Output: 3-class probability (substitute, pairs_with, unrelated).
    """

    def __init__(self, embedding_dim: int = 100, hidden_dim: int = 128):
        super().__init__()
        input_dim = embedding_dim * 4  # concat + diff + product
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, len(LABELS))
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# ── Training ─────────────────────────────────────────────────────────────

def train_classifier(
    pairs: list[tuple[str, str, str]],
    food2vec,
    embedding_dim: int = 100,
    hidden_dim: int = 128,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
    test_split: float = 0.2,
    seed: int = 42,
) -> dict:
    """Train the relationship classifier.

    Args:
        pairs: Labeled training pairs from mine_training_pairs().
        food2vec: Trained Food2Vec model.
        embedding_dim: food2vec vector dimensionality.
        hidden_dim: MLP hidden layer size.
        epochs: Training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        test_split: Fraction of data for testing.
        seed: Random seed.

    Returns:
        Dict with model, history, and test metrics.
    """
    torch.manual_seed(seed)

    # Split data
    n_test = int(len(pairs) * test_split)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(pairs))
    test_pairs = [pairs[i] for i in indices[:n_test]]
    train_pairs = [pairs[i] for i in indices[n_test:]]

    train_dataset = IngredientPairDataset(train_pairs, food2vec)
    test_dataset = IngredientPairDataset(test_pairs, food2vec)

    if len(train_dataset) == 0:
        logger.warning("No valid training samples. Check food2vec vocabulary.")
        return {"model": None, "history": {}, "metrics": {}}

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = RelationClassifier(embedding_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"loss": [], "accuracy": [], "test_accuracy": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total if total > 0 else 0

        # Evaluate
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for features, labels in test_loader:
                    outputs = model(features)
                    preds = outputs.argmax(dim=1)
                    test_correct += (preds == labels).sum().item()
                    test_total += labels.size(0)

            test_acc = test_correct / test_total if test_total > 0 else 0
            history["loss"].append(total_loss / len(train_loader))
            history["accuracy"].append(train_acc)
            history["test_accuracy"].append(test_acc)

            logger.info(
                f"  Epoch {epoch + 1:3d}/{epochs}: "
                f"loss={total_loss / len(train_loader):.4f}, "
                f"train_acc={train_acc:.3f}, test_acc={test_acc:.3f}"
            )

    # Final metrics
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    from sklearn.metrics import classification_report, f1_score
    report = classification_report(
        all_labels, all_preds,
        target_names=LABELS,
        output_dict=True,
    )
    f1 = f1_score(all_labels, all_preds, average="weighted")

    logger.info(f"  Final weighted F1: {f1:.3f}")

    return {
        "model": model,
        "history": history,
        "metrics": report,
        "f1": f1,
    }


# ── Inference ────────────────────────────────────────────────────────────

def classify_relationship(
    model: RelationClassifier,
    food2vec,
    ing_a: str,
    ing_b: str,
) -> dict:
    """Classify the relationship between two ingredients.

    Returns:
        {"relationship": str, "confidence": float, "scores": dict}
    """
    vec_a = food2vec.get_vector(ing_a)
    vec_b = food2vec.get_vector(ing_b)

    if vec_a is None or vec_b is None:
        return {"relationship": "unknown", "confidence": 0.0, "scores": {}}

    combined = np.concatenate([vec_a, vec_b, vec_a - vec_b, vec_a * vec_b])
    x = torch.tensor(combined, dtype=torch.float32).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]

    scores = {label: round(float(probs[i]), 4) for i, label in enumerate(LABELS)}
    predicted_idx = probs.argmax().item()

    return {
        "relationship": LABELS[predicted_idx],
        "confidence": round(float(probs[predicted_idx]), 4),
        "scores": scores,
    }
