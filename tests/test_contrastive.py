"""Tests for the contrastive relationship classifier."""

import sys
import os
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'research', 'phase7'))


# ── Mock food2vec for testing ────────────────────────────────────────────

class MockFood2Vec:
    """Minimal mock with controlled embeddings for testing."""

    def __init__(self, dim=32):
        self.dim = dim
        rng = np.random.RandomState(42)
        # Create embeddings where substitutes are close, pairings differ
        self._vectors = {
            "butter": rng.randn(dim).astype(np.float32),
            "margarine": None,  # set below to be close to butter
            "garlic": rng.randn(dim).astype(np.float32),
            "onion": rng.randn(dim).astype(np.float32),
            "chocolate": rng.randn(dim).astype(np.float32),
            "salmon": rng.randn(dim).astype(np.float32),
        }
        # Make margarine close to butter (substitute)
        self._vectors["margarine"] = self._vectors["butter"] + rng.randn(dim).astype(np.float32) * 0.1

    @property
    def vocabulary(self):
        return list(self._vectors.keys())

    @property
    def vector_size(self):
        return self.dim

    def get_vector(self, name):
        return self._vectors.get(name)

    def similarity(self, a, b):
        va, vb = self.get_vector(a), self.get_vector(b)
        if va is None or vb is None:
            return 0.0
        cos = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))
        return float(cos)

    def most_similar(self, name, topn=10):
        scores = []
        for other in self._vectors:
            if other != name:
                scores.append((other, self.similarity(name, other)))
        scores.sort(key=lambda x: -x[1])
        return scores[:topn]


class MockCF:
    """Minimal CF mock with controlled co-occurrence."""

    def __init__(self):
        # garlic+butter co-occur frequently, butter+margarine don't
        self._pairs = {
            "garlic": [("butter", 0.7), ("onion", 0.6)],
            "butter": [("garlic", 0.7), ("flour", 0.5)],
            "margarine": [("flour", 0.4)],
            "chocolate": [("butter", 0.3)],
            "onion": [("garlic", 0.6)],
            "salmon": [],
        }

    def similar_ingredients(self, ing, topn=10):
        return self._pairs.get(ing, [])[:topn]


class MockVocab:
    def __init__(self, words):
        self._words = {w: i for i, w in enumerate(words)}

    def encode(self, word):
        return self._words.get(word)


# ── Pair mining tests ────────────────────────────────────────────────────

class TestPairMining:
    @pytest.fixture
    def models(self):
        f2v = MockFood2Vec(dim=32)
        cf = MockCF()
        vocab = MockVocab(f2v.vocabulary)
        return f2v, cf, vocab

    def test_mines_pairs(self, models):
        from contrastive_model import mine_training_pairs
        f2v, cf, vocab = models
        pairs = mine_training_pairs(
            f2v, cf, vocab,
            f2v_threshold=0.3, cf_threshold=0.3,
            max_pairs_per_class=10,
        )
        assert len(pairs) > 0
        labels = set(label for _, _, label in pairs)
        assert "unrelated" in labels

    def test_pairs_have_correct_format(self, models):
        from contrastive_model import mine_training_pairs
        f2v, cf, vocab = models
        pairs = mine_training_pairs(
            f2v, cf, vocab,
            f2v_threshold=0.3, cf_threshold=0.3,
            max_pairs_per_class=5,
        )
        for a, b, label in pairs:
            assert isinstance(a, str)
            assert isinstance(b, str)
            assert label in ["substitute", "pairs_with", "unrelated"]

    def test_empty_with_impossible_thresholds(self, models):
        from contrastive_model import mine_training_pairs
        f2v, cf, vocab = models
        pairs = mine_training_pairs(
            f2v, cf, vocab,
            f2v_threshold=0.99, cf_threshold=0.99,
            max_pairs_per_class=5,
        )
        # May still get unrelated pairs but no substitutes/pairings
        sub_count = sum(1 for _, _, l in pairs if l == "substitute")
        pair_count = sum(1 for _, _, l in pairs if l == "pairs_with")
        assert sub_count == 0 or pair_count == 0


# ── Dataset tests ────────────────────────────────────────────────────────

class TestIngredientPairDataset:
    def test_creates_dataset(self):
        from contrastive_model import IngredientPairDataset
        f2v = MockFood2Vec(dim=32)
        pairs = [
            ("butter", "margarine", "substitute"),
            ("garlic", "butter", "pairs_with"),
            ("chocolate", "salmon", "unrelated"),
        ]
        ds = IngredientPairDataset(pairs, f2v)
        assert len(ds) == 3

    def test_feature_shape(self):
        from contrastive_model import IngredientPairDataset
        f2v = MockFood2Vec(dim=32)
        pairs = [("butter", "margarine", "substitute")]
        ds = IngredientPairDataset(pairs, f2v)
        features, label = ds[0]
        assert features.shape == (32 * 4,)  # concat + diff + product
        assert label.item() == 0  # substitute = index 0

    def test_skips_unknown_ingredients(self):
        from contrastive_model import IngredientPairDataset
        f2v = MockFood2Vec(dim=32)
        pairs = [
            ("butter", "margarine", "substitute"),
            ("unknown", "butter", "pairs_with"),
        ]
        ds = IngredientPairDataset(pairs, f2v)
        assert len(ds) == 1  # Only the valid pair


# ── Classifier tests ─────────────────────────────────────────────────────

class TestRelationClassifier:
    def test_forward_shape(self):
        from contrastive_model import RelationClassifier
        model = RelationClassifier(embedding_dim=32, hidden_dim=64)
        x = torch.randn(4, 32 * 4)  # batch of 4
        output = model(x)
        assert output.shape == (4, 3)

    def test_output_logits(self):
        from contrastive_model import RelationClassifier
        model = RelationClassifier(embedding_dim=32)
        x = torch.randn(1, 32 * 4)
        output = model(x)
        # Logits can be any real number
        assert not torch.isnan(output).any()

    def test_gradient_flow(self):
        from contrastive_model import RelationClassifier
        model = RelationClassifier(embedding_dim=32)
        x = torch.randn(2, 32 * 4)
        output = model(x)
        loss = output.sum()
        loss.backward()
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


# ── Inference tests ──────────────────────────────────────────────────────

class TestClassifyRelationship:
    def test_returns_valid_dict(self):
        from contrastive_model import RelationClassifier, classify_relationship
        f2v = MockFood2Vec(dim=32)
        model = RelationClassifier(embedding_dim=32)

        result = classify_relationship(model, f2v, "butter", "margarine")
        assert "relationship" in result
        assert result["relationship"] in ["substitute", "pairs_with", "unrelated"]
        assert 0 <= result["confidence"] <= 1
        assert len(result["scores"]) == 3

    def test_scores_sum_to_one(self):
        from contrastive_model import RelationClassifier, classify_relationship
        f2v = MockFood2Vec(dim=32)
        model = RelationClassifier(embedding_dim=32)

        result = classify_relationship(model, f2v, "garlic", "butter")
        total = sum(result["scores"].values())
        assert abs(total - 1.0) < 0.01

    def test_unknown_ingredient(self):
        from contrastive_model import RelationClassifier, classify_relationship
        f2v = MockFood2Vec(dim=32)
        model = RelationClassifier(embedding_dim=32)

        result = classify_relationship(model, f2v, "unknown", "butter")
        assert result["relationship"] == "unknown"
        assert result["confidence"] == 0.0


# ── Training integration test ────────────────────────────────────────────

class TestTrainClassifier:
    def test_trains_on_synthetic_data(self):
        from contrastive_model import train_classifier
        f2v = MockFood2Vec(dim=32)

        # Create synthetic pairs
        pairs = (
            [("butter", "margarine", "substitute")] * 20 +
            [("garlic", "butter", "pairs_with")] * 20 +
            [("chocolate", "salmon", "unrelated")] * 20
        )

        result = train_classifier(
            pairs, f2v,
            embedding_dim=32, hidden_dim=32,
            epochs=20, batch_size=16,
        )

        assert result["model"] is not None
        assert len(result["history"]["loss"]) > 0
        # Should achieve some learning (better than random 33%)
        assert result["history"]["test_accuracy"][-1] > 0.3
