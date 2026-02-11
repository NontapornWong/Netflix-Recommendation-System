# Netflix Recommendation System

Content-based filtering system for 32,000 Netflix titles (16K Movies + 16K TV Shows, 2010–2025) using TF-IDF and Sentence-BERT embeddings. Built with Python, Streamlit, and the TMDB API.

## Project Structure

```
├── app.py                  # Streamlit web app (search, filter, posters)
├── model.py                # DataProcessor, FeatureEngineer, ContentRecommender, Evaluator
├── EDA.ipynb               # Exploratory data analysis
├── results.ipynb           # Model training, evaluation, hypothesis testing
├── presentation.py         # Generates presentation.pdf (8 slides)
├── model_artifacts/        # Exported model files (parquet, npz, pkl, npy)
└── .env                    # TMDB API credentials
```

## How to Run

```bash
pip install streamlit pandas numpy scipy scikit-learn sentence-transformers requests python-dotenv
streamlit run app.py
```

---

## Exploratory Data Analysis

### Dataset Overview

- **Movies**: 16,000 rows, 18 columns (includes `budget`, `revenue`)
- **TV Shows**: 16,000 rows, 16 columns
- **Temporal balance**: Exactly 1,000 titles per year (2010–2025) in both datasets
- **ID overlap**: 397 `show_id` values appear in both datasets but refer to different content — resolved with composite key `(show_id, type)`

### Data Quality

| Feature     | Movies Missing | TV Missing |
|-------------|---------------|------------|
| director    | 0.8%          | 68.5%      |
| description | 0.8%          | 20.0%      |
| genres      | 0.7%          | 6.1%       |
| cast        | 1.3%          | 7.2%       |

- `duration` is 100% null for movies; string format ("X Seasons") for TV
- `date_added` stored as string in both datasets
- TV shows with `vote_count=0` carry non-zero `vote_average` (values like 3, 8, 10) — unreliable without confidence weighting

### Genre Taxonomy Mismatch

Only 8 of ~28 genres are shared directly between datasets. Key mappings applied:

- "Sci-Fi & Fantasy" → "Science Fiction, Fantasy"
- "Action & Adventure" → "Action, Adventure"
- "War & Politics" → "War"
- "Kids" → "Family", "Soap" → "Drama"

### Engagement Metrics

- **vote_average**: Movies centered around 6.0–7.0; TV is bimodal (properly rated shows at 5–8, large cluster at 0)
- **vote_count**: Heavily right-skewed in both; movies median 138, TV median 4
- **popularity**: TV has higher median popularity (36.2) than movies (10.9) despite fewer votes, suggesting the metric incorporates factors beyond voting

### Cross-Dataset Bridges

- **Cast**: ~9,586 shared actors (16% of ~60K unique) — strongest cross-dataset signal
- **Directors**: ~502 shared (3.5%) — supplementary signal only, unusable for TV (68.5% null)
- **Languages**: 62 shared out of 74 movie / 71 TV languages; English dominates (~60% movies, ~28% TV)

### Feature Roles

- **Similarity features** (priority order): genres, description, cast, country, director (movies only)
- **Ranking signals**: weighted composite of vote_average, vote_count, and popularity — used to order equally similar items by quality

---

## System Architecture

### Pipeline

1. **Data Processing** — Merge movies + TV, harmonize genres, handle missing values, create composite UID
2. **Feature Engineering** — Build TF-IDF "tag soup" (weighted: 3x genre, 2x cast, 1x director, 1x description) + separate per-feature TF-IDF matrices
3. **SBERT Embeddings** — `all-MiniLM-L6-v2` encodes tag soup into 384-dim dense vectors
4. **Similarity** — Cosine similarity across TF-IDF, SBERT, or hybrid matrices
5. **Ranking** — IMDB weighted rating combined with similarity: `final_score = 0.8 * similarity + 0.2 * normalized_rating`
6. **App** — Streamlit UI with search, genre filtering, TMDB poster display

### 6 Approaches Compared

| Approach | Method |
|----------|--------|
| Combined (TF-IDF Soup) | Single TF-IDF matrix on weighted tag soup |
| Separate (TF-IDF Weighted) | Per-feature TF-IDF matrices with manual weights |
| Embedding (SBERT Tag) | SBERT on tag soup |
| Embedding (SBERT Desc) | SBERT on description only |
| Hybrid (TF-IDF + SBERT Tag) | Normalized blend of TF-IDF + SBERT Tag (50/50) |
| Hybrid (TF-IDF + SBERT Desc) | Normalized blend of TF-IDF + SBERT Desc (50/50) |

---

## Hypothesis Testing

### H1: Multi-Feature Similarity Outperforms Genre-Only

**Rationale**: With only 19 genre labels across 16K+ movies, genre-only similarity assigns identical scores to thousands of items. Multi-feature should produce more granular differentiation.

- **Test**: Wilcoxon signed-rank on per-item top-10 score variance (genre-only vs multi-feature), n=500
- **Result**: W = 12,458, **p = 9.42e-51**
- **Interpretation**: Multi-feature variance is 6.8x higher (0.0009 vs 0.0001). **Reject H0.** Combining genres, cast, director, and description produces significantly more discriminative similarity scores than genre alone. This validates the multi-feature design.

### H2: Feature Independence

**Rationale**: If features are too correlated, combining them adds no value. If completely independent, combining may introduce noise. Moderate correlation (0.2–0.5) justifies weighted combination.

- **Test**: Spearman rank correlation on pairwise feature similarities, n=1000 pairs
- **Result**: genres–cast ρ = 0.025, genres–description ρ = 0.093, cast–description ρ = -0.024
- **Interpretation**: All feature pairs have ρ < 0.2 — effectively independent. Each feature captures a distinct dimension of similarity. **Weighted combination is justified** since features are complementary, not redundant. Note: director correlations returned NaN due to extreme sparsity (68.5% null in TV), confirming director is too sparse to serve as a reliable standalone feature.

### H3: Weighted Rating > Raw vote_average

**Rationale**: Raw `vote_average` is unreliable — TV items with 0 votes carry ratings of 10.0. The IMDB weighted formula `WR = v/(v+m) * R + m/(v+m) * C` pulls low-confidence ratings toward the global mean.

- **Test**: Wilcoxon on average vote_count in top-5 recommendations under each ranking method, n=500
- **Result**: Kendall τ = 0.064, Wilcoxon **p = 1.04e-79**
- **Interpretation**: The two rankings are nearly uncorrelated (τ ≈ 0.06), meaning weighted rating produces fundamentally different orderings. Weighted rating surfaces items with avg 1,237 votes in top-5 vs 850 for raw — a 45% increase in vote confidence. **Reject H0.** Weighted rating reliably promotes better-evidenced content.

### H4: Method Distinctness (TF-IDF vs SBERT)

**Rationale**: If TF-IDF and SBERT produce the same recommendations, there is no value in having both.

- **Test**: Jaccard similarity of top-10 recommendation sets, n=500
- **Result**: Mean Jaccard = **0.034**
- **Interpretation**: Only 3.4% overlap — SBERT finds almost entirely different items than TF-IDF. **High distinctness.** The two methods capture fundamentally different notions of similarity (lexical vs semantic). This has a critical implication for hybrid fusion: averaging two near-disjoint sets produces a compromised ranking rather than a complementary one.

### H5: Score Distribution (The Loudness Problem)

**Rationale**: If one method's scores are systematically higher, a naive average will be dominated by the "louder" method.

- **Test**: Kolmogorov-Smirnov test on top-50 score distributions, n=500
- **Result**: KS = 0.70, **p ≈ 0**, TF-IDF mean = 0.42, SBERT mean = 0.63
- **Interpretation**: SBERT scores are ~50% higher than TF-IDF on average. The distributions are fundamentally different (KS = 0.70). **Normalization is necessary before hybrid blending**, otherwise SBERT dominates the combined score and TF-IDF contributes almost nothing.

### H6: Hybrid Quality Lift

**Rationale**: Does blending TF-IDF + SBERT actually improve recommendation quality (measured by avg weighted_rating of top-10)?

- **Test**: Wilcoxon on avg weighted_rating of top-10 (TF-IDF-only vs Hybrid), n=500
- **Result**: TF-IDF avg = 5.82, Hybrid avg = 5.83, **p = 0.043**
- **Interpretation**: Statistically significant but **practically negligible** (+0.01 rating improvement). The naive 50/50 hybrid barely outperforms TF-IDF alone, far from matching pure SBERT's performance. This confirms the hybrid strategy needs better weight tuning.

---

## Results

### Approach Comparison (200-title benchmark)

| Rank | Approach | Mean Final Score | Mean Weighted Rating |
|------|----------|-----------------|---------------------|
| 1 | **Embedding (SBERT Tag)** | **0.6152** | 5.892 |
| 2 | Embedding (SBERT Desc) | 0.5319 | 5.896 |
| 3 | Hybrid (TF-IDF + SBERT Tag) | 0.5137 | 5.827 |
| 4 | Combined (TF-IDF Soup) | 0.4548 | 5.820 |
| 5 | Hybrid (TF-IDF + SBERT Desc) | 0.4489 | 5.870 |
| 6 | Separate (TF-IDF Weighted) | 0.4107 | 5.924 |

### Why SBERT Tag Won (Not Hybrid)

The expectation was that hybrid would outperform individual methods. It did not, for four reasons:

1. **The Loudness Problem (H5)**: SBERT scores are ~50% higher than TF-IDF. Even after normalization, the 50/50 blend dilutes SBERT's strong signal with TF-IDF's weaker one.
2. **Signal Dilution (H4)**: With only 3.4% top-10 overlap, TF-IDF and SBERT find almost entirely different items. Averaging two divergent rankings produces a compromise worse than the stronger individual.
3. **Tag Soup is Already Rich**: The weighted tag (3x genre + 2x cast + director + description) gives SBERT a dense, semantically complete input. SBERT captures latent relationships (tone, theme, narrative style) that TF-IDF's bag-of-words approach cannot.
4. **Naive Hybrid Weights**: A fixed 50/50 split treats both methods as equally valuable. The data shows SBERT is clearly stronger — learned weights (e.g., 80/20 SBERT-to-TF-IDF) would likely improve hybrid performance.

---

## Areas of Improvement

- **Tune Hybrid Weights**: Replace fixed 50/50 with grid search or learned weights. An 80/20 SBERT-to-TF-IDF ratio would likely outperform pure SBERT.
- **Handle Data Sparsity in TV**: 68.5% missing directors and 20% missing descriptions weaken TF-IDF features. Imputation or feature fallback strategies would help.
- **Upgrade Embedding Model**: `all-MiniLM-L6-v2` (384-dim) is lightweight. Larger models like `all-mpnet-base-v2` (768-dim) could capture richer semantic relationships.
- **Add Collaborative Filtering**: Current system is purely content-based. User interaction data (watch history, ratings) would enable a content + collaborative hybrid.
- **Evaluation with Human Judgement**: Current metric (`final_score`) is automated. A/B testing or user studies would provide ground-truth validation of recommendation relevance.
