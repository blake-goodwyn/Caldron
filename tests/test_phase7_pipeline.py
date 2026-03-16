"""Tests for Phase 7 culinary ML data pipeline and models."""

import sys
import os
import pytest
import numpy as np

# Add research/phase7 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'research', 'phase7'))


# ── Ingredient normalization ─────────────────────────────────────────────

class TestNormalizeIngredient:
    def test_strips_quantities(self):
        from data_pipeline import normalize_ingredient
        assert normalize_ingredient("2 cups flour") == "flour"

    def test_strips_units(self):
        from data_pipeline import normalize_ingredient
        assert normalize_ingredient("1 tablespoon olive oil") == "olive oil"

    def test_strips_fractions(self):
        from data_pipeline import normalize_ingredient
        assert normalize_ingredient("½ cup sugar") == "sugar"

    def test_strips_ascii_fractions(self):
        from data_pipeline import normalize_ingredient
        assert normalize_ingredient("1/2 cup butter") == "butter"

    def test_strips_prep_instructions(self):
        from data_pipeline import normalize_ingredient
        result = normalize_ingredient("3 cloves garlic, minced")
        assert "garlic" in result
        assert "minced" not in result
        assert "cloves" not in result

    def test_strips_parenthetical(self):
        from data_pipeline import normalize_ingredient
        result = normalize_ingredient("chicken breast (boneless, skinless)")
        assert "boneless" not in result
        assert "chicken" in result

    def test_preserves_compound_names(self):
        from data_pipeline import normalize_ingredient
        result = normalize_ingredient("all-purpose flour")
        assert "all-purpose" in result

    def test_lowercases(self):
        from data_pipeline import normalize_ingredient
        assert normalize_ingredient("SALT") == "salt"

    def test_empty_after_strip(self):
        from data_pipeline import normalize_ingredient
        result = normalize_ingredient("1/2 cup")
        assert result == ""


# ── RecipeNLG parsing ────────────────────────────────────────────────────

class TestParseRecipeNLG:
    def test_parses_valid_row(self):
        from data_pipeline import parse_recipenlg_row
        row = {
            "title": "Simple Pasta",
            "NER": '["pasta", "garlic", "olive oil", "salt"]',
            "ingredients": '["1 lb pasta", "3 cloves garlic"]',
            "source": "test",
        }
        result = parse_recipenlg_row(row)
        assert result is not None
        assert result["title"] == "Simple Pasta"
        assert len(result["ingredients"]) >= 2

    def test_skips_single_ingredient(self):
        from data_pipeline import parse_recipenlg_row
        row = {"title": "Water", "NER": '["water"]', "ingredients": '["water"]'}
        assert parse_recipenlg_row(row) is None

    def test_handles_malformed_ner(self):
        from data_pipeline import parse_recipenlg_row
        row = {"title": "Bad", "NER": "not a list", "ingredients": "also bad"}
        assert parse_recipenlg_row(row) is None

    def test_deduplicates_ingredients(self):
        from data_pipeline import parse_recipenlg_row
        row = {
            "title": "Test",
            "NER": '["salt", "pepper", "salt", "garlic"]',
            "ingredients": "[]",
        }
        result = parse_recipenlg_row(row)
        assert result is not None
        assert len(result["ingredients"]) == len(set(result["ingredients"]))


# ── Vocabulary ───────────────────────────────────────────────────────────

class TestIngredientVocab:
    @pytest.fixture
    def sample_recipes(self):
        return [
            {"ingredients": ["garlic", "butter", "salt"]},
            {"ingredients": ["garlic", "olive oil", "pepper"]},
            {"ingredients": ["butter", "flour", "sugar"]},
            {"ingredients": ["garlic", "soy sauce", "ginger"]},
            {"ingredients": ["butter", "garlic", "lemon"]},
        ]

    def test_fit_builds_vocabulary(self, sample_recipes):
        from data_pipeline import IngredientVocab
        vocab = IngredientVocab(min_count=2)
        vocab.fit(sample_recipes)
        assert vocab.size > 0
        assert "garlic" in vocab.word2idx  # appears 4 times

    def test_min_count_filters(self, sample_recipes):
        from data_pipeline import IngredientVocab
        vocab = IngredientVocab(min_count=3)
        vocab.fit(sample_recipes)
        assert "garlic" in vocab.word2idx   # 4 occurrences
        assert "butter" in vocab.word2idx   # 3 occurrences
        assert "flour" not in vocab.word2idx  # 1 occurrence

    def test_encode_decode_roundtrip(self, sample_recipes):
        from data_pipeline import IngredientVocab
        vocab = IngredientVocab(min_count=1)
        vocab.fit(sample_recipes)
        idx = vocab.encode("garlic")
        assert idx is not None
        assert vocab.decode(idx) == "garlic"

    def test_encode_unknown_returns_none(self, sample_recipes):
        from data_pipeline import IngredientVocab
        vocab = IngredientVocab(min_count=1)
        vocab.fit(sample_recipes)
        assert vocab.encode("unicorn tears") is None

    def test_save_load_roundtrip(self, sample_recipes, tmp_path):
        from data_pipeline import IngredientVocab
        vocab = IngredientVocab(min_count=1)
        vocab.fit(sample_recipes)

        path = tmp_path / "vocab.json"
        vocab.save(path)

        loaded = IngredientVocab.load(path)
        assert loaded.size == vocab.size
        assert loaded.encode("garlic") == vocab.encode("garlic")


# ── Co-occurrence matrix ─────────────────────────────────────────────────

class TestCooccurrenceMatrix:
    def test_builds_symmetric_matrix(self):
        from data_pipeline import IngredientVocab, build_cooccurrence_matrix
        recipes = [
            {"ingredients": ["a", "b", "c"]},
            {"ingredients": ["a", "b"]},
        ]
        vocab = IngredientVocab(min_count=1).fit(recipes)
        matrix = build_cooccurrence_matrix(recipes, vocab)

        idx_a = vocab.encode("a")
        idx_b = vocab.encode("b")
        # a and b co-occur in both recipes
        assert matrix[idx_a, idx_b] == matrix[idx_b, idx_a]
        assert matrix[idx_a, idx_b] > 0

    def test_diagonal_is_zero(self):
        from data_pipeline import IngredientVocab, build_cooccurrence_matrix
        recipes = [{"ingredients": ["a", "b", "c"]}]
        vocab = IngredientVocab(min_count=1).fit(recipes)
        matrix = build_cooccurrence_matrix(recipes, vocab)

        for i in range(vocab.size):
            assert matrix[i, i] == 0


# ── Recipe-ingredient matrix ────────────────────────────────────────────

class TestRecipeIngredientMatrix:
    def test_correct_shape(self):
        from data_pipeline import IngredientVocab, build_recipe_ingredient_matrix
        recipes = [
            {"ingredients": ["a", "b"]},
            {"ingredients": ["b", "c"]},
            {"ingredients": ["a", "c"]},
        ]
        vocab = IngredientVocab(min_count=1).fit(recipes)
        matrix = build_recipe_ingredient_matrix(recipes, vocab)
        assert matrix.shape == (3, vocab.size)

    def test_binary_values(self):
        from data_pipeline import IngredientVocab, build_recipe_ingredient_matrix
        recipes = [
            {"ingredients": ["a", "b", "c"]},
            {"ingredients": ["a", "b"]},
        ]
        vocab = IngredientVocab(min_count=1).fit(recipes)
        matrix = build_recipe_ingredient_matrix(recipes, vocab)
        assert matrix.max() == 1.0
        # Sparse matrix only stores non-zero; check dense has zeros
        dense = matrix.toarray()
        assert dense.min() == 0.0


# ── Compound overlap ────────────────────────────────────────────────────

class TestCompoundOverlap:
    def test_identical_compounds(self):
        from data_pipeline import compound_overlap_score
        flavordb = {"a": ["c1", "c2", "c3"], "b": ["c1", "c2", "c3"]}
        assert compound_overlap_score("a", "b", flavordb) == 1.0

    def test_no_overlap(self):
        from data_pipeline import compound_overlap_score
        flavordb = {"a": ["c1", "c2"], "b": ["c3", "c4"]}
        assert compound_overlap_score("a", "b", flavordb) == 0.0

    def test_partial_overlap(self):
        from data_pipeline import compound_overlap_score
        flavordb = {"a": ["c1", "c2", "c3"], "b": ["c2", "c3", "c4"]}
        score = compound_overlap_score("a", "b", flavordb)
        assert 0 < score < 1
        # Jaccard: |{c2,c3}| / |{c1,c2,c3,c4}| = 2/4 = 0.5
        assert abs(score - 0.5) < 1e-6

    def test_missing_ingredient(self):
        from data_pipeline import compound_overlap_score
        flavordb = {"a": ["c1"]}
        assert compound_overlap_score("a", "unknown", flavordb) == 0.0


# ── Food2Vec ─────────────────────────────────────────────────────────────

class TestFood2Vec:
    @pytest.fixture
    def trained_model(self):
        from food2vec import Food2Vec
        # Create enough recipes for meaningful training
        recipes = [
            {"ingredients": ["garlic", "butter", "parsley", "salt"]},
            {"ingredients": ["garlic", "olive oil", "basil", "tomato"]},
            {"ingredients": ["butter", "flour", "sugar", "eggs"]},
            {"ingredients": ["soy sauce", "ginger", "garlic", "sesame oil"]},
            {"ingredients": ["garlic", "butter", "lemon", "capers"]},
            {"ingredients": ["olive oil", "garlic", "chili flakes", "pasta"]},
            {"ingredients": ["butter", "cream", "parmesan", "pasta"]},
            {"ingredients": ["soy sauce", "rice vinegar", "ginger", "garlic"]},
            {"ingredients": ["tomato", "basil", "mozzarella", "olive oil"]},
            {"ingredients": ["butter", "garlic", "white wine", "shallot"]},
        ] * 20  # Repeat to meet min_count

        model = Food2Vec(vector_size=32, window=5, min_count=5, epochs=10)
        model.train(recipes)
        return model

    def test_train_produces_vectors(self, trained_model):
        vec = trained_model.get_vector("garlic")
        assert vec is not None
        assert len(vec) == 32

    def test_vocabulary_populated(self, trained_model):
        assert len(trained_model.vocabulary) > 0
        assert "garlic" in trained_model.vocabulary

    def test_most_similar_returns_results(self, trained_model):
        results = trained_model.most_similar("garlic", topn=5)
        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_similarity_range(self, trained_model):
        sim = trained_model.similarity("garlic", "butter")
        assert -1.0 <= sim <= 1.0

    def test_unknown_ingredient_returns_empty(self, trained_model):
        assert trained_model.most_similar("unicorn tears") == []

    def test_unknown_similarity_returns_zero(self, trained_model):
        assert trained_model.similarity("garlic", "unicorn tears") == 0.0

    def test_save_load_roundtrip(self, trained_model, tmp_path):
        from food2vec import Food2Vec
        path = tmp_path / "test_model.model"
        trained_model.save(path)

        loaded = Food2Vec.load(path)
        # Vectors should be identical
        original_vec = trained_model.get_vector("garlic")
        loaded_vec = loaded.get_vector("garlic")
        np.testing.assert_array_almost_equal(original_vec, loaded_vec)

    def test_untrained_model_raises(self):
        from food2vec import Food2Vec
        model = Food2Vec()
        with pytest.raises(RuntimeError):
            model.most_similar("garlic")


# ── Collaborative Filtering ─────────────────────────────────────────────

class TestIngredientCF:
    @pytest.fixture
    def cf_setup(self):
        from data_pipeline import IngredientVocab, build_recipe_ingredient_matrix
        from affinity_models import IngredientCF

        recipes = [
            {"ingredients": ["garlic", "butter", "parsley"]},
            {"ingredients": ["garlic", "olive oil", "basil"]},
            {"ingredients": ["butter", "flour", "sugar"]},
            {"ingredients": ["garlic", "soy sauce", "ginger"]},
            {"ingredients": ["butter", "garlic", "lemon"]},
        ] * 10

        vocab = IngredientVocab(min_count=2).fit(recipes)
        matrix = build_recipe_ingredient_matrix(recipes, vocab)
        cf = IngredientCF(n_neighbors=5).fit(matrix, vocab)
        return cf, vocab

    def test_similar_ingredients_returns_results(self, cf_setup):
        cf, vocab = cf_setup
        results = cf.similar_ingredients("garlic", topn=3)
        assert len(results) > 0
        assert all(isinstance(r, tuple) for r in results)

    def test_similar_ingredients_excludes_self(self, cf_setup):
        cf, vocab = cf_setup
        results = cf.similar_ingredients("garlic", topn=10)
        names = [r[0] for r in results]
        assert "garlic" not in names

    def test_unknown_ingredient(self, cf_setup):
        cf, vocab = cf_setup
        assert cf.similar_ingredients("dragon fruit") == []

    def test_suggest_ingredients(self, cf_setup):
        cf, vocab = cf_setup
        suggestions = cf.suggest_ingredients(["garlic", "butter"], topn=5)
        assert len(suggestions) > 0
        names = [s[0] for s in suggestions]
        assert "garlic" not in names
        assert "butter" not in names


# ── Technique extraction ─────────────────────────────────────────────────

class TestTechniqueExtraction:
    def test_extracts_basic_techniques(self):
        from affinity_models import extract_techniques_from_instructions
        text = "Sauté the garlic in olive oil, then roast the chicken."
        techniques = extract_techniques_from_instructions(text)
        assert "saute" in techniques
        assert "roast" in techniques

    def test_handles_conjugated_verbs(self):
        from affinity_models import extract_techniques_from_instructions
        text = "The bread was baked until golden. Simmer the sauce."
        techniques = extract_techniques_from_instructions(text)
        assert "bake" in techniques
        assert "simmer" in techniques

    def test_empty_instructions(self):
        from affinity_models import extract_techniques_from_instructions
        assert extract_techniques_from_instructions("") == []

    def test_no_techniques_found(self):
        from affinity_models import extract_techniques_from_instructions
        text = "Mix ingredients together and serve cold."
        techniques = extract_techniques_from_instructions(text)
        # "mix" is not in our technique list
        assert "mix" not in techniques


# ── Combined Affinity ────────────────────────────────────────────────────

class TestCombinedAffinity:
    def test_combines_scores(self):
        from affinity_models import CombinedAffinity

        # Mock food2vec
        class MockF2V:
            def similarity(self, a, b): return 0.8
            def most_similar(self, ing, topn=10):
                return [("butter", 0.8), ("salt", 0.6)]

        class MockCF:
            def similar_ingredients(self, ing, topn=10):
                return [("butter", 0.7), ("pepper", 0.5)]

        combined = CombinedAffinity(MockF2V(), MockCF(), alpha=0.5)
        score = combined.affinity("garlic", "butter")
        expected = 0.5 * 0.8 + 0.5 * 0.7
        assert abs(score - expected) < 1e-6

    def test_top_affinities_unions_candidates(self):
        from affinity_models import CombinedAffinity

        class MockF2V:
            def similarity(self, a, b): return 0.5
            def most_similar(self, ing, topn=10):
                return [("a", 0.9), ("b", 0.7)]

        class MockCF:
            def similar_ingredients(self, ing, topn=10):
                return [("b", 0.8), ("c", 0.6)]

        combined = CombinedAffinity(MockF2V(), MockCF(), alpha=0.5)
        results = combined.top_affinities("garlic", topn=5)
        names = [r[0] for r in results]
        assert "a" in names
        assert "b" in names
        assert "c" in names
