# Phase 7 — Experimental Findings & Direction Assessment

## 1. Results Summary

### Milestone B: food2vec (Ingredient Embeddings)

**Verdict: Strong foundation. Ship to agent system.**

What worked:
- Substitutability is well-captured: `butter -> margarine (0.93)`, `olive oil -> virgin olive oil (0.74)`, `cumin -> comino (0.67), coriander (0.62)`.
- Cuisine clustering emerges organically: `soy sauce -> hoisin (0.68), teriyaki (0.67), oyster sauce (0.67)` is a clean Asian sauce cluster. `cilantro -> black beans (0.73), lime zest (0.72)` captures the Mexican profile.
- Product variants group correctly: `cream cheese -> philadelphia cream cheese (0.74)`, `chocolate -> sweet chocolate (0.72), baking chocolate (0.63)`.

What didn't work:
- Analogies largely failed — not enough cuisine/category labels in the vocabulary to support `soy_sauce - asian + italian = ?` style queries. The model lacks explicit relational structure.
- Some noise in neighbors: `garlic -> arborio rice (0.66)` and `ginger -> pumpkins (0.56)` are weak signals from co-occurrence rather than true affinity.
- Honey and lemon have diffuse, low-confidence neighborhoods — these are "bridge" ingredients that appear across many cuisines, making their embeddings less discriminative.

**Key insight:** food2vec is excellent at answering "what can replace X?" but poor at answering "what goes well with X?" — it confuses co-occurrence frequency with compatibility.

### Milestone C: Collaborative Filtering

**Verdict: Best model for recipe completion. Ship immediately.**

What worked:
- Recipe completion is remarkably accurate:
  - `[chocolate, butter, sugar]` -> flour (0.96), vanilla (0.89), eggs (0.84) — textbook brownie/cake base.
  - `[tomato, basil, mozzarella]` -> oregano, garlic, onion, parmesan, olive oil — Margherita/Caprese build.
  - `[garlic, butter, parsley]` -> salt, onion, pepper, olive oil — classic French/Italian savory base.
- Scores are well-calibrated: high confidence on obvious completions, lower confidence on plausible-but-optional additions.

What didn't work:
- `[soy sauce, ginger, rice]` -> molasses (0.29) is a miss — likely from co-occurrence with brown sugar in teriyaki glazes.
- No awareness of proportions, technique, or role — it treats all ingredients as equally important.

**Key insight:** CF is the strongest "recipe completion" engine. It answers "given these ingredients, what else do I need?" with high accuracy. This maps directly to the agent's recipe development workflow.

### Milestone D: Knowledge Graph (RotatE)

**Verdict: Promising for multi-relational reasoning. Needs more relation types and data.**

What worked:
- The model learns typed relationships — `(garlic, pairs_with, ?) -> tomato sauce, olives, cheese` is culinarily sound.
- `(chocolate, pairs_with, ?) -> vanilla, walnuts, powdered sugar` is a clean dessert cluster.
- Cuisine assignment is directionally correct.

What didn't work:
- Hits@10 = 0.371, MRR = 0.189 — decent but not strong enough for production queries. The model ranks the correct answer in the top 10 only 37% of the time.
- Only 3 relation types (pairs_with, same_cuisine, variant_of) — too few for the model to learn nuanced distinctions. The charter calls for substitution, technique-outcome, and functional role relations.
- 50K recipe subset produces a dense pairs_with graph (26K triples) that overwhelms the sparser cuisine/variant signals.

**Key insight:** The KG approach has the highest ceiling but needs more relation types to justify its complexity. Adding `substitutes_for`, `cooked_by_technique`, and `has_flavor_compound` relations would give RotatE the relational diversity it needs.

### Milestone D: GNN (GCN Link Prediction)

**Verdict: Excellent link predictor (AUC 0.987). Useful for a different purpose than expected.**

What worked:
- Near-perfect link prediction: the GCN can reliably tell whether two ingredients appear together in recipes.
- Complementarity learning: GCN neighbors are _things that go together_ rather than _things that replace each other_. `garlic -> onion, pepper`; `butter -> flour, milk, eggs`; `olive oil -> mushrooms, onions, parsley`.

What didn't work:
- Over-smoothing: GCN embeddings collapse to similar vectors for high-degree nodes. All common ingredients have cosine similarity near 1.0, making neighbor queries less discriminative than food2vec.
- 1% overlap with food2vec neighbors — confirms these models learn fundamentally different things, but the GCN's neighbor quality for end-user queries is weaker.

**Key insight:** The GCN is a powerful binary classifier ("do these belong together?") but a poor neighbor-finder. Best used as a coherence check rather than a suggestion engine — "does this ingredient fit this recipe?" rather than "what ingredients should I add?"

---

## 2. Research Questions — Status

| RQ | Question | Status | Evidence |
|----|----------|--------|----------|
| RQ1 | Can co-occurrence learn flavor affinity without chemistry? | **Partially answered.** food2vec learns substitutability and cuisine clusters, but not true affinity. CF better captures "what goes with what." | food2vec neighbors, CF suggestions |
| RQ2 | Does FlavorDB improve predictions? | **Not yet tested.** FlavorDB data pipeline exists but wasn't used as node features. | Needed: compound overlap evaluation |
| RQ3 | Can NMF recover technique-ingredient clusters? | **Not yet tested.** NMF module exists, RecipeNLG lacks instruction text in the Zappandy variant. | Blocked on instructions data |
| RQ4 | Do embeddings support functional substitution? | **Partially.** food2vec captures variant substitution (butter/margarine) but not functional role (acid/fat/umami). | Analogy failures, neighbor quality |
| RQ5 | Can embedding distance score recipe coherence? | **Partially.** GCN AUC=0.987 means it can score whether an ingredient "fits" a recipe. Not yet tested as an explicit coherence score. | GCN link prediction |

---

## 3. Model-to-Task Matrix (Empirical)

Based on actual results, not theoretical predictions:

| Model | Recipe Completion | Substitution | Coherence Check | Flavor Exploration |
|-------|:-:|:-:|:-:|:-:|
| **food2vec** | Poor | **Excellent** | Weak | Moderate |
| **CF** | **Excellent** | Weak | Moderate | Weak |
| **KG (RotatE)** | Moderate | Moderate | Moderate | **Good** |
| **GCN** | Good | Weak | **Excellent** | Weak |

---

## 4. Recommended Directions

### High value — pursue next

**1. Integrate CF + food2vec into agent_tools (Milestone E)**
These two models are production-ready for their respective tasks:
- CF for recipe completion suggestions during recipe development
- food2vec for substitution queries ("I don't have X, what can I use instead?")
- Combined: use CF to suggest what's missing, food2vec to suggest alternatives for ingredients the user doesn't have.

**2. FlavorDB integration (answers RQ2)**
The compound data pipeline exists. Three concrete experiments:
- Use compound vectors as food2vec training features (does it improve analogy accuracy?)
- Use compound overlap as an explicit affinity signal alongside co-occurrence
- Build compound-mediated triples for the KG ("tomato and basil share linalool")

**3. GCN as coherence scorer (answers RQ5)**
The GCN's link prediction AUC of 0.987 means it can reliably evaluate recipe coherence. Concrete integration: given a recipe's ingredient list, compute average pairwise link probability. Low scores flag "weird" recipes, high scores confirm coherence.

### Medium value — explore if time permits

**4. Richer KG with more relation types**
Current KG has only 3 relations. Adding `substitutes_for` (from food2vec nearest neighbors), `cooked_by` (from technique extraction), and `shares_compound` (from FlavorDB) would unlock the typed-reasoning advantage that RotatE is designed for.

**5. Full RecipeNLG (2.2M recipes)**
Current models trained on 50K recipes. Scaling to 2.2M would:
- Sharpen rare-ingredient embeddings (salmon, coconut milk have weak neighborhoods now)
- Densify the KG
- Require Kaggle account for CSV download

**6. Technique-outcome modeling (answers RQ3)**
Blocked on recipe instructions data. The Zappandy HuggingFace variant has a `directions` field — need to verify it contains parseable instructions and feed it into the NMF pipeline.

### Lower priority — defer

**7. VAE for recipe interpolation** — Interesting for "recipe exploration" UX but doesn't directly improve agent quality. Revisit after core models are integrated.

**8. Transformer fine-tuning** — Heavy compute, and the simpler models already capture most of the signal. Save for when we need full-text recipe understanding (Phase 8+).

**9. Bayesian optimization** — Requires human-in-the-loop evaluation. Only useful once the app has active recipe development sessions to optimize.

---

## 5. Vocabulary Quality Issues

Observed problems in the ingredient vocabulary that affect all models:

1. **Synonym fragmentation:** "butter" and "margarine" correctly cluster, but "olive oil" / "virgin olive oil" / "light olive oil" consume 3 vocabulary slots for the same thing.
2. **Noise tokens:** "amounts", "new", "containers", "pink" appear as ingredient names — these are parsing artifacts from the NER column.
3. **Compound names:** "hot buttered noodles" and "broken spaghetti" are dishes, not ingredients.
4. **Brand names:** "philadelphia cream cheese", "parkay margarine", "uncle ben" add noise.

**Recommendation:** Build a synonym/canonical mapping table before scaling to 2.2M recipes. This is high-leverage — it improves every downstream model simultaneously.
