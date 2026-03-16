# Phase 7 — Culinary ML Research

## 1. Objective

Build ML models and embeddings that feed back into Caldron's agent system — making Remy, HealthNut, and Critic meaningfully smarter than prompt-engineered wrappers around a general-purpose LLM.

## 2. Research Questions

| ID | Question | Target Task |
|----|----------|-------------|
| RQ1 | Can ingredient co-occurrence in recipe corpora learn meaningful flavor affinity without chemical data? | Flavor affinity |
| RQ2 | Does adding flavor compound features (FlavorDB) improve affinity predictions over co-occurrence alone? | Flavor affinity |
| RQ3 | Can matrix factorization recover interpretable technique–ingredient clusters? | Technique-outcome |
| RQ4 | Do learned embeddings support functional substitution (acid↔acid, fat↔fat) beyond category matching? | Substitution |
| RQ5 | Can reconstruction error or embedding distance serve as a recipe coherence score? | Coherence scoring |

## 3. Public Datasets

| Dataset | Contents | Size | Access |
|---------|----------|------|--------|
| **RecipeNLG** | 2.2M recipes (title, ingredients, instructions) | ~2 GB | [recipenlg.cs.put.poznan.pl](https://recipenlg.cs.put.poznan.pl/) |
| **FlavorDB** | Flavor compounds for ~1,000 ingredients | ~5 MB | [cosylab.iiitd.edu.in/flavordb](https://cosylab.iiitd.edu.in/flavordb/) |
| **FooDB** | Detailed food composition (compounds, nutrients) | ~500 MB | [foodb.ca](https://foodb.ca/) |
| **USDA FoodData Central** | Nutritional composition, API available | API | [fdc.nal.usda.gov](https://fdc.nal.usda.gov/) |
| **FoodKG** | Structured food knowledge graph | ~200 MB | USC ISI |

**Starting dataset:** RecipeNLG (ingredient lists) + FlavorDB (compound features).

## 4. Technique Survey

### Tier 1 — Foundations (scikit-learn, no GPU)

These produce results in hours, not days, and establish baselines.

#### food2vec (Ingredient Embeddings)
- **What:** Train Word2Vec on recipe ingredient lists. Each recipe = "sentence", each ingredient = "word."
- **Why:** Ingredients that co-occur get similar vectors. Vector arithmetic works: `soy_sauce - asian + french ≈ worcestershire`.
- **Data:** 50K–100K recipes, normalized ingredient names.
- **Complexity:** Low. gensim Word2Vec, <50 lines, trains in minutes on CPU.
- **Serves:** Flavor affinity (primary), Substitution (secondary).
- **Key ref:** Altosaar (2015) food2vec; Park et al. (2019) KitcheNette.

#### Collaborative Filtering
- **What:** Treat recipe×ingredient matrix like user×item. Predict which ingredients "belong" given a partial set.
- **Why:** Captures culinary co-occurrence without chemistry data. Strong baseline.
- **Data:** Recipe-ingredient binary matrix, 50K+ recipes.
- **Complexity:** Low. implicit ALS or scikit-learn NearestNeighbors on TF-IDF.
- **Serves:** Flavor affinity (primary), Coherence scoring (secondary).
- **Key ref:** Freyne & Berkovsky (2010) Intelligent Food Planning.

#### Matrix Factorization (NMF)
- **What:** Decompose ingredient×technique matrix into latent factors.
- **Why:** Reveals interpretable clusters: "things that get roasted" vs. "things that get poached."
- **Data:** Co-occurrence counts from recipe instructions, 50K+ recipes.
- **Complexity:** Low. scikit-learn NMF/TruncatedSVD, ~20 lines.
- **Serves:** Technique-outcome (primary), Flavor affinity (secondary).
- **Key ref:** Ahn et al. (2011) Flavor network paper.

### Tier 2 — Graph & Relational (PyTorch, single GPU optional)

#### Knowledge Graph Embeddings (PyKEEN)
- **What:** Embed (ingredient, relation, ingredient) triples — e.g., (tomato, pairs_with, basil), (butter, substituted_by, coconut_oil).
- **Why:** Models typed relationships, not just similarity. Distinguishes "pairs with" from "substitutes for."
- **Data:** Food KG with 10K–50K triples. Bootstrap from recipes + FlavorDB.
- **Complexity:** Medium-low. PyKEEN makes training RotatE a ~30-line script.
- **Serves:** Flavor affinity + Substitution (both primary).
- **Key ref:** Haussmann et al. (2019) FoodKG.

#### Graph Neural Networks (PyTorch Geometric)
- **What:** Message-passing on ingredient co-occurrence graph. Learn representations encoding multi-hop relationships.
- **Why:** A→B and B→C suggests A↔C in certain contexts. Naturally fits food's graph structure.
- **Data:** Co-occurrence graph from 50K+ recipes, FlavorDB node features.
- **Complexity:** Medium. ~100–150 lines PyTorch. CPU feasible for <100K nodes.
- **Serves:** Flavor affinity (primary), Substitution + Coherence (secondary).
- **Key ref:** Park et al. (2021) FlavorGraph.

#### Contrastive Learning
- **What:** Pull co-occurring ingredient pairs closer, push random pairs apart. InfoNCE loss.
- **Why:** Can incorporate multimodal features (text + compounds + nutrition).
- **Data:** Same as food2vec, optionally augmented with FlavorDB vectors.
- **Complexity:** Medium. ~100 lines PyTorch.
- **Serves:** Flavor affinity (primary), Substitution (secondary).

### Tier 3 — Generative & Sequential (GPU recommended)

#### Variational Autoencoders
- Encode recipes as points in continuous latent space. Interpolate between recipes.
- **Serves:** Trajectory learning, Substitution.

#### Transformers (RecipeBERT / fine-tuned DistilBERT)
- Full recipe understanding. Coherence classification: is this recipe internally consistent?
- **Serves:** Coherence scoring, Trajectory learning.

#### Bayesian Optimization
- Suggest next experiment in recipe parameter tuning (temp, time, ratios).
- **Serves:** Trajectory learning (human-in-the-loop).

## 5. Implementation Plan

### Milestone A — Data Pipeline (1 session)
- [ ] Download and parse RecipeNLG subset (100K recipes)
- [ ] Normalize ingredient names (lemmatize, strip quantities/units)
- [ ] Build ingredient vocabulary and co-occurrence matrix
- [ ] Download FlavorDB compound data
- [ ] Create unified data loader module

### Milestone B — food2vec Baseline (1 session)
- [ ] Train Word2Vec on ingredient lists
- [ ] Evaluate: nearest neighbors for 20 common ingredients
- [ ] Evaluate: analogy tasks (cuisine transfer, role substitution)
- [ ] Visualize embedding space (t-SNE/UMAP)
- [ ] Benchmark: flavor affinity prediction vs. FlavorDB ground truth

### Milestone C — Collaborative Filtering + NMF (1 session)
- [ ] Train ALS on recipe×ingredient matrix
- [ ] Train NMF on ingredient×technique matrix
- [ ] Compare CF ingredient suggestions against food2vec neighbors
- [ ] Inspect NMF components for interpretability
- [ ] Build combined affinity score: food2vec similarity + CF score

### Milestone D — Knowledge Graph + GNN (2 sessions)
- [ ] Construct food knowledge graph (ingredients, techniques, cuisines, compounds)
- [ ] Train RotatE embeddings with PyKEEN
- [ ] Build ingredient co-occurrence graph with FlavorDB node features
- [ ] Train 2-layer GCN for link prediction
- [ ] Evaluate: does the GNN beat food2vec on affinity prediction?

### Milestone E — Integration Prototype (1 session)
- [ ] Package best-performing model as a Python module
- [ ] Define API: `get_affinity(a, b) → float`, `get_substitutions(ingredient, context) → list`
- [ ] Wire into Caldron agent_tools as new tool functions
- [ ] Write tests for the ML module interface
- [ ] Demo: agent uses learned embeddings instead of hardcoded rules

## 6. Evaluation Criteria

| Metric | Method | Target |
|--------|--------|--------|
| Affinity accuracy | Precision@10 against FlavorDB compound overlap | >0.6 |
| Substitution quality | Human eval: 20 substitution queries rated 1–5 | avg >3.5 |
| Embedding quality | Ingredient analogy accuracy | >0.4 |
| Coherence detection | Binary classification on synthetic corrupted recipes | AUC >0.75 |
| Technique clustering | NMF component interpretability (manual inspection) | Recognizable clusters |

## 7. Key References

1. Ahn, Y.-Y. et al. "Flavor network and the principles of food pairing." *Scientific Reports* (2011).
2. Altosaar, J. "food2vec — Augmented cooking with machine intelligence." (2015).
3. Park, D. et al. "KitcheNette: Predicting and Ranking Food Ingredient Pairings using Siamese Neural Networks." *IJCAI* (2019).
4. Park, D. et al. "FlavorGraph: A Large-Scale Food-Chemical Graph." *Scientific Reports* (2021).
5. Haussmann, S. et al. "FoodKG: A Semantics-Driven Knowledge Graph for Food Recommendation." (2019).
6. Salvador, A. et al. "Learning Cross-Modal Embeddings for Cooking Recipes and Food Images." *CVPR* (2017).
7. Majumder, B. et al. "Generating Personalized Recipes from Historical User Preferences." *EMNLP* (2019).
