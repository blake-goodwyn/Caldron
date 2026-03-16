"""
Phase 7 -- GNN: Graph Neural Network for Ingredient Link Prediction

Trains a 2-layer GCN on the ingredient co-occurrence graph to predict
whether two ingredients belong together in a recipe.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


# ── GCN Model ────────────────────────────────────────────────────────────

class IngredientGCN(nn.Module):
    """2-layer Graph Convolutional Network for ingredient embeddings.

    Learns ingredient representations by aggregating neighbor information
    through the co-occurrence graph. Used for link prediction: do these
    two ingredients belong together?
    """

    def __init__(self, num_nodes: int, input_dim: int, hidden_dim: int = 64, output_dim: int = 32):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim

        # Node feature embedding (if no external features)
        self.node_embedding = nn.Embedding(num_nodes, input_dim)

        # GCN layers (manual implementation to avoid torch-geometric version issues)
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(0.3)

    def normalize_adjacency(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute symmetric normalized adjacency D^{-1/2} A D^{-1/2}."""
        # Build sparse adjacency
        row, col = edge_index
        # Add self-loops
        self_loops = torch.arange(num_nodes, device=edge_index.device)
        row = torch.cat([row, self_loops])
        col = torch.cat([col, self_loops])

        # Compute degree
        deg = torch.zeros(num_nodes, device=edge_index.device)
        deg.scatter_add_(0, row, torch.ones(row.size(0), device=edge_index.device))

        # D^{-1/2}
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # Normalized edge weights
        weights = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Return as sparse tensor
        adj = torch.sparse_coo_tensor(
            torch.stack([row, col]), weights, (num_nodes, num_nodes)
        )
        return adj

    def forward(self, edge_index: torch.Tensor, node_features: Optional[torch.Tensor] = None):
        """Forward pass through 2-layer GCN.

        Args:
            edge_index: [2, num_edges] tensor of edge indices.
            node_features: Optional [num_nodes, input_dim] feature matrix.
                          If None, uses learned embeddings.

        Returns:
            Node embeddings [num_nodes, output_dim].
        """
        if node_features is None:
            x = self.node_embedding.weight
        else:
            x = node_features

        adj = self.normalize_adjacency(edge_index, self.num_nodes)

        # Layer 1: aggregate + transform + ReLU
        x = torch.sparse.mm(adj, x)
        x = self.W1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2: aggregate + transform
        x = torch.sparse.mm(adj, x)
        x = self.W2(x)

        return x

    def predict_link(self, embeddings: torch.Tensor, node_a: int, node_b: int) -> float:
        """Predict link probability between two nodes via dot product."""
        with torch.no_grad():
            score = torch.dot(embeddings[node_a], embeddings[node_b])
            return torch.sigmoid(score).item()


# ── Training utilities ───────────────────────────────────────────────────

def build_edge_index_from_cooccurrence(cooc_matrix, threshold: float = 5.0):
    """Convert co-occurrence matrix to edge_index tensor.

    Args:
        cooc_matrix: Sparse co-occurrence matrix.
        threshold: Minimum co-occurrence count to create an edge.

    Returns:
        edge_index [2, num_edges] tensor.
    """
    from scipy.sparse import coo_matrix

    coo = coo_matrix(cooc_matrix)
    mask = coo.data >= threshold
    rows = torch.tensor(coo.row[mask], dtype=torch.long)
    cols = torch.tensor(coo.col[mask], dtype=torch.long)

    edge_index = torch.stack([rows, cols])
    logger.info(f"Edge index: {edge_index.shape[1]} edges (threshold={threshold})")
    return edge_index


def sample_negative_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_negatives: int,
    rng: Optional[np.random.RandomState] = None,
) -> torch.Tensor:
    """Sample negative (non-existent) edges for contrastive training."""
    rng = rng or np.random.RandomState(42)

    # Build set of existing edges for fast lookup
    existing = set()
    for i in range(edge_index.shape[1]):
        existing.add((edge_index[0, i].item(), edge_index[1, i].item()))

    # Cap negatives at available non-edges
    max_possible = num_nodes * (num_nodes - 1) - len(existing)
    num_negatives = min(num_negatives, max_possible)

    neg_edges = []
    attempts = 0
    max_attempts = num_negatives * 20
    while len(neg_edges) < num_negatives and attempts < max_attempts:
        a = rng.randint(0, num_nodes)
        b = rng.randint(0, num_nodes)
        attempts += 1
        if a != b and (a, b) not in existing:
            neg_edges.append((a, b))
            existing.add((a, b))

    neg_index = torch.tensor(neg_edges, dtype=torch.long).T
    return neg_index


def train_gnn(
    cooc_matrix,
    vocab,
    food2vec_model=None,
    hidden_dim: int = 64,
    output_dim: int = 32,
    lr: float = 0.01,
    epochs: int = 200,
    edge_threshold: float = 5.0,
) -> dict:
    """Train GCN for ingredient link prediction.

    Args:
        cooc_matrix: Sparse ingredient co-occurrence matrix.
        vocab: IngredientVocab.
        food2vec_model: Optional Food2Vec model for initial node features.
        hidden_dim: GCN hidden layer dimension.
        output_dim: Output embedding dimension.
        lr: Learning rate.
        epochs: Training epochs.
        edge_threshold: Min co-occurrence for edge creation.

    Returns:
        Dict with model, embeddings, and training history.
    """
    num_nodes = vocab.size

    # Build edges
    edge_index = build_edge_index_from_cooccurrence(cooc_matrix, threshold=edge_threshold)

    # Initialize node features from food2vec if available
    node_features = None
    input_dim = 100
    if food2vec_model is not None:
        features = []
        for i in range(num_nodes):
            name = vocab.decode(i)
            vec = food2vec_model.get_vector(name) if name else None
            if vec is not None:
                features.append(vec)
            else:
                features.append(np.random.randn(food2vec_model.vector_size).astype(np.float32) * 0.01)
        node_features = torch.tensor(np.array(features), dtype=torch.float32)
        input_dim = food2vec_model.vector_size

    # Create model
    model = IngredientGCN(num_nodes, input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Sample negative edges (same count as positive for balanced training)
    num_pos = edge_index.shape[1]
    neg_edge_index = sample_negative_edges(edge_index, num_nodes, num_pos)
    num_neg = neg_edge_index.shape[1]

    # Split positive edges into train/test
    perm = torch.randperm(num_pos)
    split = int(0.8 * num_pos)
    train_pos = edge_index[:, perm[:split]]
    test_pos = edge_index[:, perm[split:]]

    neg_split = int(0.8 * num_neg)
    train_neg = neg_edge_index[:, :neg_split]
    test_neg = neg_edge_index[:, neg_split:]

    # Training loop
    history = {"loss": [], "train_auc": [], "test_auc": []}

    logger.info(f"Training GCN: {num_nodes} nodes, {num_pos} edges, {epochs} epochs")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass (use all edges for message passing)
        embeddings = model(edge_index, node_features)

        # Positive scores
        pos_scores = (embeddings[train_pos[0]] * embeddings[train_pos[1]]).sum(dim=1)
        # Negative scores
        neg_scores = (embeddings[train_neg[0]] * embeddings[train_neg[1]]).sum(dim=1)

        # Binary cross-entropy loss
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores)
        )
        loss = pos_loss + neg_loss

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            # Evaluate
            model.eval()
            with torch.no_grad():
                embeddings = model(edge_index, node_features)

                # Train AUC
                train_pos_scores = torch.sigmoid(
                    (embeddings[train_pos[0]] * embeddings[train_pos[1]]).sum(dim=1)
                )
                train_neg_scores = torch.sigmoid(
                    (embeddings[train_neg[0]] * embeddings[train_neg[1]]).sum(dim=1)
                )
                train_auc = compute_auc(train_pos_scores, train_neg_scores)

                # Test AUC
                if test_pos.shape[1] > 0 and test_neg.shape[1] > 0:
                    test_pos_scores = torch.sigmoid(
                        (embeddings[test_pos[0]] * embeddings[test_pos[1]]).sum(dim=1)
                    )
                    test_neg_scores = torch.sigmoid(
                        (embeddings[test_neg[0]] * embeddings[test_neg[1]]).sum(dim=1)
                    )
                    test_auc = compute_auc(test_pos_scores, test_neg_scores)
                else:
                    test_auc = 0.5

            history["loss"].append(float(loss))
            history["train_auc"].append(train_auc)
            history["test_auc"].append(test_auc)

            logger.info(
                f"  Epoch {epoch + 1:3d}/{epochs}: "
                f"loss={loss:.4f}, train_AUC={train_auc:.3f}, test_AUC={test_auc:.3f}"
            )

    # Final embeddings
    model.eval()
    with torch.no_grad():
        final_embeddings = model(edge_index, node_features)

    return {
        "model": model,
        "embeddings": final_embeddings,
        "edge_index": edge_index,
        "history": history,
    }


def compute_auc(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> float:
    """Compute AUC from positive and negative scores."""
    scores = torch.cat([pos_scores, neg_scores]).numpy()
    labels = np.concatenate([
        np.ones(len(pos_scores)),
        np.zeros(len(neg_scores)),
    ])
    # Simple AUC: fraction of pos > neg pairs
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return 0.5


# ── Evaluation ───────────────────────────────────────────────────────────

def evaluate_gnn_neighbors(
    embeddings: torch.Tensor,
    vocab,
    test_ingredients: list[str],
    topn: int = 10,
) -> dict[str, list[tuple[str, float]]]:
    """Find nearest neighbors in GNN embedding space."""
    results = {}

    # L2-normalize embeddings for more discriminative cosine similarity
    normed = F.normalize(embeddings, p=2, dim=1)

    for ing in test_ingredients:
        idx = vocab.encode(ing)
        if idx is None:
            continue

        # Euclidean distance in normalized space (more discriminative than cosine
        # when embeddings are over-smoothed)
        query = normed[idx].unsqueeze(0)
        dists = torch.cdist(query, normed).squeeze(0)
        dists[idx] = float('inf')  # exclude self

        top_indices = torch.topk(dists, topn, largest=False).indices
        neighbors = []
        for i in top_indices:
            name = vocab.decode(i.item())
            if name:
                # Convert distance to similarity score
                sim = 1.0 / (1.0 + float(dists[i]))
                neighbors.append((name, sim))

        results[ing] = neighbors
        if neighbors:
            neighbor_str = ", ".join(f"{n} ({s:.3f})" for n, s in neighbors[:5])
            logger.info(f"  {ing}: {neighbor_str}")

    return results


def compare_with_food2vec(
    gnn_embeddings: torch.Tensor,
    food2vec_model,
    vocab,
    test_ingredients: list[str],
    topn: int = 10,
) -> dict:
    """Compare GNN vs food2vec neighbor rankings."""
    comparison = {}

    for ing in test_ingredients:
        idx = vocab.encode(ing)
        if idx is None or ing not in food2vec_model.vocabulary:
            continue

        # GNN neighbors
        query = gnn_embeddings[idx].unsqueeze(0)
        sims = F.cosine_similarity(query, gnn_embeddings)
        sims[idx] = -1
        gnn_top = torch.topk(sims, topn).indices
        gnn_names = set()
        for i in gnn_top:
            name = vocab.decode(i.item())
            if name:
                gnn_names.add(name)

        # food2vec neighbors
        f2v_neighbors = food2vec_model.most_similar(ing, topn=topn)
        f2v_names = set(n for n, _ in f2v_neighbors)

        overlap = gnn_names & f2v_names
        comparison[ing] = {
            "gnn_top": list(gnn_names),
            "f2v_top": list(f2v_names),
            "overlap": list(overlap),
            "overlap_ratio": len(overlap) / topn if topn > 0 else 0,
        }

    avg_overlap = np.mean([v["overlap_ratio"] for v in comparison.values()])
    logger.info(f"Average GNN-food2vec overlap: {avg_overlap:.2f}")

    return comparison


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("Usage:")
    print("  python gnn_model.py train   -- train GCN on co-occurrence graph")
    print("  python gnn_model.py eval    -- evaluate trained GCN")
