"""Tests for Phase 7 Knowledge Graph and GNN modules."""

import sys
import os
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'research', 'phase7'))


# ── Cuisine detection ────────────────────────────────────────────────────

class TestCuisineDetection:
    def test_detects_asian(self):
        from knowledge_graph import detect_cuisines
        result = detect_cuisines(["soy sauce", "ginger", "sesame oil", "rice"])
        assert "asian" in result

    def test_detects_italian(self):
        from knowledge_graph import detect_cuisines
        result = detect_cuisines(["basil", "mozzarella", "olive oil", "oregano"])
        assert "italian" in result

    def test_detects_mexican(self):
        from knowledge_graph import detect_cuisines
        result = detect_cuisines(["cilantro", "cumin", "black beans", "avocado"])
        assert "mexican" in result

    def test_no_cuisine_with_generic_ingredients(self):
        from knowledge_graph import detect_cuisines
        result = detect_cuisines(["salt", "water", "flour"])
        assert len(result) == 0

    def test_multiple_cuisines(self):
        from knowledge_graph import detect_cuisines
        # Fusion dish with both Asian and Mexican markers
        result = detect_cuisines([
            "soy sauce", "ginger", "cilantro", "cumin", "lime"
        ])
        assert len(result) >= 2


# ── FoodKnowledgeGraph ───────────────────────────────────────────────────

class TestFoodKnowledgeGraph:
    @pytest.fixture
    def sample_recipes(self):
        return [
            {"ingredients": ["garlic", "butter", "parsley"]},
            {"ingredients": ["garlic", "olive oil", "basil"]},
            {"ingredients": ["garlic", "butter", "lemon"]},
            {"ingredients": ["garlic", "butter", "salt"]},
            {"ingredients": ["soy sauce", "ginger", "garlic"]},
        ] * 5  # Repeat to meet min_cooccurrence

    @pytest.fixture
    def sample_vocab(self, sample_recipes):
        from data_pipeline import IngredientVocab
        return IngredientVocab(min_count=2).fit(sample_recipes)

    def test_add_triple(self):
        from knowledge_graph import FoodKnowledgeGraph
        kg = FoodKnowledgeGraph()
        kg.add_triple("garlic", "pairs_with", "butter")
        assert kg.num_triples == 1
        assert "garlic" in kg.entities
        assert "butter" in kg.entities
        assert "pairs_with" in kg.relations

    def test_add_pairing_triples(self, sample_recipes, sample_vocab):
        from knowledge_graph import FoodKnowledgeGraph
        kg = FoodKnowledgeGraph()
        kg.add_pairing_triples(sample_recipes, sample_vocab, min_cooccurrence=3)
        assert kg.num_triples > 0
        # garlic + butter co-occur in 3+ recipes
        pair_found = any(
            (h == "garlic" and t == "butter") or (h == "butter" and t == "garlic")
            for h, r, t in kg.triples if r == "pairs_with"
        )
        assert pair_found

    def test_add_cuisine_triples(self, sample_recipes, sample_vocab):
        from knowledge_graph import FoodKnowledgeGraph
        kg = FoodKnowledgeGraph()
        kg.add_cuisine_triples(sample_recipes, sample_vocab)
        # May or may not detect cuisines depending on markers
        assert isinstance(kg.num_triples, int)

    def test_add_variant_triples(self, sample_vocab):
        from knowledge_graph import FoodKnowledgeGraph
        kg = FoodKnowledgeGraph()
        kg.add_variant_triples(sample_vocab)
        # Check that variant detection runs without error
        assert isinstance(kg.num_triples, int)

    def test_save_load_roundtrip(self, sample_recipes, sample_vocab, tmp_path):
        from knowledge_graph import FoodKnowledgeGraph
        kg = FoodKnowledgeGraph()
        kg.add_pairing_triples(sample_recipes, sample_vocab, min_cooccurrence=3)

        path = tmp_path / "test_kg.json"
        kg.save(path)

        loaded = FoodKnowledgeGraph.load(path)
        assert loaded.num_triples == kg.num_triples

    def test_to_triples_factory(self, sample_recipes, sample_vocab):
        from knowledge_graph import FoodKnowledgeGraph
        kg = FoodKnowledgeGraph()
        kg.add_pairing_triples(sample_recipes, sample_vocab, min_cooccurrence=3)

        if kg.num_triples > 0:
            factory = kg.to_triples_factory()
            assert factory.num_triples == kg.num_triples


# ── GCN Model ────────────────────────────────────────────────────────────

class TestIngredientGCN:
    @pytest.fixture
    def simple_graph(self):
        """Simple 5-node graph for testing."""
        from gnn_model import IngredientGCN
        num_nodes = 5
        model = IngredientGCN(num_nodes, input_dim=16, hidden_dim=8, output_dim=4)
        # Simple chain: 0-1-2-3-4
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4],
            [1, 0, 2, 1, 3, 2, 4, 3],
        ], dtype=torch.long)
        return model, edge_index, num_nodes

    def test_forward_shape(self, simple_graph):
        model, edge_index, num_nodes = simple_graph
        embeddings = model(edge_index)
        assert embeddings.shape == (num_nodes, 4)

    def test_forward_with_features(self, simple_graph):
        model, edge_index, num_nodes = simple_graph
        features = torch.randn(num_nodes, 16)
        embeddings = model(edge_index, features)
        assert embeddings.shape == (num_nodes, 4)

    def test_predict_link_range(self, simple_graph):
        model, edge_index, _ = simple_graph
        model.eval()
        with torch.no_grad():
            embeddings = model(edge_index)
        score = model.predict_link(embeddings, 0, 1)
        assert 0.0 <= score <= 1.0

    def test_gradient_flow(self, simple_graph):
        model, edge_index, _ = simple_graph
        embeddings = model(edge_index)
        loss = embeddings.sum()
        loss.backward()
        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


# ── Edge utilities ───────────────────────────────────────────────────────

class TestEdgeUtilities:
    def test_build_edge_index_from_cooccurrence(self):
        from gnn_model import build_edge_index_from_cooccurrence
        from scipy.sparse import csr_matrix

        # Simple 3x3 matrix
        data = np.array([[0, 10, 3], [10, 0, 8], [3, 8, 0]], dtype=np.float32)
        matrix = csr_matrix(data)

        edge_index = build_edge_index_from_cooccurrence(matrix, threshold=5.0)
        # Should have edges for pairs with count >= 5: (0,1), (1,0), (1,2), (2,1)
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] == 4  # 4 directed edges

    def test_sample_negative_edges(self):
        from gnn_model import sample_negative_edges

        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        neg = sample_negative_edges(edge_index, num_nodes=10, num_negatives=5)
        assert neg.shape[0] == 2
        assert neg.shape[1] == 5

        # Negatives should not include existing edges
        existing = set()
        for i in range(edge_index.shape[1]):
            existing.add((edge_index[0, i].item(), edge_index[1, i].item()))
        for i in range(neg.shape[1]):
            assert (neg[0, i].item(), neg[1, i].item()) not in existing

    def test_compute_auc(self):
        from gnn_model import compute_auc
        pos = torch.tensor([0.9, 0.8, 0.7])
        neg = torch.tensor([0.2, 0.3, 0.1])
        auc = compute_auc(pos, neg)
        assert auc > 0.9  # Should be high since pos > neg


# ── Integration test: small GCN training ────────────────────────────────

class TestGNNTraining:
    def test_train_gnn_small(self):
        """End-to-end GCN training on a small synthetic graph."""
        from gnn_model import train_gnn
        from data_pipeline import IngredientVocab, build_cooccurrence_matrix
        import random

        random.seed(42)

        # Create a sparser graph with 30 ingredients in small clusters
        all_ingredients = [f"ing_{i}" for i in range(30)]
        recipes = []
        for _ in range(100):
            # Pick a random cluster of 3-5 ingredients
            size = random.randint(3, 5)
            start = random.randint(0, 25)
            ingredients = all_ingredients[start:start + size]
            recipes.append({"ingredients": ingredients})

        vocab = IngredientVocab(min_count=2).fit(recipes)
        cooc = build_cooccurrence_matrix(recipes, vocab)

        results = train_gnn(
            cooc_matrix=cooc,
            vocab=vocab,
            food2vec_model=None,
            hidden_dim=16,
            output_dim=8,
            lr=0.01,
            epochs=20,
            edge_threshold=2.0,
        )

        assert "embeddings" in results
        assert results["embeddings"].shape[0] == vocab.size
        assert results["embeddings"].shape[1] == 8
        assert len(results["history"]["loss"]) > 0
