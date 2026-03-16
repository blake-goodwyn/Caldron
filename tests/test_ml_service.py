"""Tests for the production ML service layer."""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cauldron-app'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'research', 'phase7'))


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton between tests."""
    from ml_service import CulinaryMLService
    CulinaryMLService.reset()
    yield
    CulinaryMLService.reset()


class TestCulinaryMLServiceInit:
    def test_singleton_pattern(self):
        from ml_service import CulinaryMLService
        a = CulinaryMLService()
        b = CulinaryMLService()
        assert a is b

    def test_reset_creates_new_instance(self):
        from ml_service import CulinaryMLService
        a = CulinaryMLService()
        CulinaryMLService.reset()
        b = CulinaryMLService()
        assert a is not b

    def test_instantiation_is_fast(self):
        import time
        from ml_service import CulinaryMLService
        start = time.perf_counter()
        CulinaryMLService()
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1  # Should be <50ms, allow 100ms for safety

    def test_available_false_when_disabled(self):
        from ml_service import CulinaryMLService
        service = CulinaryMLService()
        service._enabled = False
        assert not service.available

    def test_available_false_when_no_models(self, tmp_path):
        from ml_service import CulinaryMLService
        CulinaryMLService.reset()
        service = CulinaryMLService(models_dir=str(tmp_path))
        assert not service.available


class TestSuggestSubstitutions:
    def test_returns_list_of_dicts(self):
        from ml_service import CulinaryMLService
        service = CulinaryMLService()

        mock_model = MagicMock()
        mock_model.most_similar.return_value = [
            ("margarine", 0.93), ("oleo", 0.77), ("shortening", 0.55)
        ]
        service._food2vec = mock_model
        service._canonical_map = MagicMock()

        with patch.object(service, '_normalize', return_value='butter'):
            results = service.suggest_substitutions("butter", n=3)

        assert len(results) == 3
        assert results[0]["name"] == "margarine"
        assert results[0]["score"] == 0.93
        assert results[0]["source"] == "food2vec"

    def test_returns_empty_when_disabled(self):
        from ml_service import CulinaryMLService
        service = CulinaryMLService()
        service._enabled = False
        assert service.suggest_substitutions("butter") == []

    def test_returns_empty_when_no_model(self):
        from ml_service import CulinaryMLService
        service = CulinaryMLService()
        service._food2vec = None
        # Force _load_food2vec to return None
        with patch.object(service, '_load_food2vec', return_value=None):
            assert service.suggest_substitutions("butter") == []

    def test_returns_empty_for_unknown_ingredient(self):
        from ml_service import CulinaryMLService
        service = CulinaryMLService()
        mock_model = MagicMock()
        mock_model.most_similar.return_value = []
        service._food2vec = mock_model

        with patch.object(service, '_normalize', return_value=''):
            results = service.suggest_substitutions("xyzzy")
        assert results == []


class TestCompleteRecipe:
    def test_returns_suggestions(self):
        from ml_service import CulinaryMLService
        service = CulinaryMLService()

        mock_cf = MagicMock()
        mock_cf.suggest_ingredients.return_value = [
            ("flour", 0.96), ("vanilla", 0.89), ("eggs", 0.84)
        ]
        service._cf = mock_cf

        with patch.object(service, '_normalize', side_effect=lambda x: x.lower()):
            results = service.complete_recipe(["chocolate", "butter", "sugar"], n=3)

        assert len(results) == 3
        assert results[0]["name"] == "flour"
        assert results[0]["source"] == "collaborative_filtering"

    def test_returns_empty_when_disabled(self):
        from ml_service import CulinaryMLService
        service = CulinaryMLService()
        service._enabled = False
        assert service.complete_recipe(["garlic"]) == []

    def test_handles_empty_input(self):
        from ml_service import CulinaryMLService
        service = CulinaryMLService()
        mock_cf = MagicMock()
        service._cf = mock_cf

        with patch.object(service, '_normalize', return_value=''):
            results = service.complete_recipe([])
        assert results == []


class TestScoreAffinity:
    def test_returns_score_dict(self):
        from ml_service import CulinaryMLService
        service = CulinaryMLService()

        mock_model = MagicMock()
        mock_model.similarity.return_value = 0.73
        service._food2vec = mock_model

        with patch.object(service, '_normalize', side_effect=lambda x: x.lower()):
            result = service.score_affinity("garlic", "butter")

        assert result["score"] == 0.73
        assert result["source"] == "food2vec"

    def test_returns_zero_when_disabled(self):
        from ml_service import CulinaryMLService
        service = CulinaryMLService()
        service._enabled = False
        result = service.score_affinity("garlic", "butter")
        assert result["score"] == 0.0
        assert result["source"] == "unavailable"

    def test_returns_zero_for_unknown(self):
        from ml_service import CulinaryMLService
        service = CulinaryMLService()
        mock_model = MagicMock()
        service._food2vec = mock_model

        with patch.object(service, '_normalize', return_value=''):
            result = service.score_affinity("garlic", "xyzzy")
        assert result["source"] == "unknown_ingredient"


class TestIntegrationWithRealModels:
    """Integration tests that run only when model files exist."""

    @pytest.fixture
    def models_dir(self):
        d = Path(__file__).parent.parent / "research" / "phase7" / "data"
        if not (d / "food2vec.model").exists():
            pytest.skip("Model files not available")
        return d

    def test_real_substitution(self, models_dir):
        from ml_service import CulinaryMLService
        service = CulinaryMLService(models_dir=str(models_dir))
        results = service.suggest_substitutions("butter", n=3)
        assert len(results) > 0
        names = [r["name"] for r in results]
        assert "margarine" in names

    @pytest.mark.slow
    def test_real_completion(self, models_dir):
        from ml_service import CulinaryMLService
        service = CulinaryMLService(models_dir=str(models_dir))
        results = service.complete_recipe(["chocolate", "butter", "sugar"], n=5)
        assert len(results) > 0
        names = [r["name"] for r in results]
        # Should suggest baking staples
        assert any(n in names for n in ["flour", "vanilla", "eggs", "salt", "milk"])

    def test_real_affinity(self, models_dir):
        from ml_service import CulinaryMLService
        service = CulinaryMLService(models_dir=str(models_dir))
        result = service.score_affinity("garlic", "butter")
        assert result["score"] > 0
