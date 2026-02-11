import logging
import pandas as pd
import numpy as np
import re
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def _top_k_similar(sim_scores, idx, top_k):
    """Mask self-similarity and return top-k indices with scores."""
    sim_scores = sim_scores.copy()
    sim_scores[idx] = -1
    top_indices = sim_scores.argsort()[::-1][:top_k]
    return top_indices, sim_scores[top_indices]


def _normalize(scores):
    """Min-max normalize an array of scores to [0, 1]."""
    min_val, max_val = scores.min(), scores.max()
    if max_val - min_val == 0:
        return np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val)

class DataProcessor:
    SELECTED_COLS = [
        'show_id', 'type', 'title', 'director', 'cast', 'country',
        'date_added', 'release_year', 'rating', 'genres', 'language',
        'description', 'popularity', 'vote_count', 'vote_average'
    ]

    GENRE_MAP = {
        'Sci-Fi & Fantasy': 'Science Fiction, Fantasy',
        'Action & Adventure': 'Action, Adventure',
        'War & Politics': 'War',
        'Kids': 'Family',
        'Soap': 'Drama',
    }

    TEXT_FILL_COLS = ['genres', 'cast', 'director', 'description', 'country']

    def __init__(self, movies_path, tv_path):
        self.movie_raw = pd.read_csv(movies_path)
        self.tv_raw = pd.read_csv(tv_path)
        self._validate_columns(self.movie_raw, movies_path)
        self._validate_columns(self.tv_raw, tv_path)
        self.df = None

    def _validate_columns(self, df, source_path):
        missing = set(self.SELECTED_COLS) - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing columns in '{source_path}': {sorted(missing)}"
            )

    def _create_unified_dataset(self):
        df_movie = self.movie_raw[self.SELECTED_COLS]
        df_tv = self.tv_raw[self.SELECTED_COLS]
        self.df = pd.concat([df_movie, df_tv], axis=0, ignore_index=True)
        self.df['uid'] = self.df['show_id'].astype(str) + '_' + self.df['type']
        return self

    def _harmonize_genres(self):
        def map_genres(genre_str):
            if pd.isna(genre_str):
                return genre_str
            genres = [g.strip() for g in genre_str.split(',')]
            mapped = []
            for g in genres:
                if g in self.GENRE_MAP:
                    mapped.extend([x.strip() for x in self.GENRE_MAP[g].split(',')])
                else:
                    mapped.append(g)
            return ', '.join(dict.fromkeys(mapped))

        self.df['genres'] = self.df['genres'].apply(map_genres)
        return self

    def _handle_missing(self):
        for col in self.TEXT_FILL_COLS:
            self.df[col] = self.df[col].fillna('')
        return self

    def process(self):
        self._create_unified_dataset()
        self._harmonize_genres()
        self._handle_missing()
        return self.df


class FeatureEngineer:
    TFIDF_CONFIG = {
        'stop_words': 'english',
        'max_features': 10000,
        'ngram_range': (1, 2),
        'min_df': 2,
        'max_df': 0.95,
    }

    FEATURE_WEIGHTS = {
        'genres': 3,
        'cast': 2,
        'director': 1,
        'description': 1,
    }

    def __init__(self, df):
        self.df = df.copy().reset_index(drop=True)
        self.tfidf = None
        self.tfidf_matrix = None
        self.feature_matrices = {}
        self.feature_vectorizers = {}

    @staticmethod
    def _clean_text(text):
        return re.sub(r'[^a-z0-9\s]', ' ', str(text).lower())

    @staticmethod
    def _names_to_tokens(name_str):
        return ' '.join(name.strip().replace(' ', '_')
                        for name in str(name_str).split(',') if name.strip())

    def _build_soup(self, row):
        genre = self._clean_text(row['genres'])
        cast = self._names_to_tokens(row['cast'])
        director = self._names_to_tokens(row['director'])
        description = self._clean_text(row['description'])

        parts = []
        for text, weight in zip(
            [genre, cast, director, description],
            [self.FEATURE_WEIGHTS['genres'], self.FEATURE_WEIGHTS['cast'],
             self.FEATURE_WEIGHTS['director'], self.FEATURE_WEIGHTS['description']]
        ):
            parts.extend([text] * weight)
        return ' '.join(parts)

    def build_separate_matrices(self):
        feature_configs = {
            'genres':      (self._clean_text, {'stop_words': 'english'}),
            'description': (self._clean_text, self.TFIDF_CONFIG),
            'cast':        (self._names_to_tokens, {}),
            'director':    (self._names_to_tokens, {}),
        }

        for feature, (preprocessor, tfidf_params) in feature_configs.items():
            vectorizer = TfidfVectorizer(**tfidf_params)
            self.feature_matrices[feature] = vectorizer.fit_transform(
                self.df[feature].apply(preprocessor)
            )
            self.feature_vectorizers[feature] = vectorizer

        return self

    def create_tag(self):
        self.df['tag'] = self.df.apply(self._build_soup, axis=1)
        return self

    def build_tfidf_matrix(self):
        self.tfidf = TfidfVectorizer(**self.TFIDF_CONFIG)
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['tag'])
        return self

    def get_outputs(self):
        return self.df, self.tfidf_matrix, self.tfidf

class EmbeddingFeatureEngineer:
    def __init__(self, df, model_name='all-MiniLM-L6-v2'):
        self.df = df
        self.model = SentenceTransformer(model_name)
        self.embeddings = None

    def build_embeddings(self, column='tag'):
        texts = self.df[column].tolist()
        self.embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=64
        )
        logger.info("Embedding shape: %s", self.embeddings.shape)
        return self

    def get_outputs(self):
        return self.df, self.embeddings
    
class ContentRecommender:
    def __init__(self, df, tfidf_matrix=None, feature_matrices=None, 
                 feature_weights=None, embeddings=None):
        self.df = df.copy().reset_index(drop=True)
        self.tfidf_matrix = tfidf_matrix
        self.feature_matrices = feature_matrices
        self.feature_weights = feature_weights or {
            'genres': 0.4, 'cast': 0.3,
            'description': 0.2, 'director': 0.1
        }
        self.embeddings = embeddings

    def compute_weighted_rating(self, percentile=0.85):
        v = self.df['vote_count']
        R = self.df['vote_average']
        C = R.mean()
        m = v.quantile(percentile)

        self.df['weighted_rating'] = (v / (v + m)) * R + (m / (v + m)) * C
        self.df['weighted_rating'] = self.df['weighted_rating'].round(2)
        return self

    def _get_similar_combined(self, idx, top_k=10):
        sim_row = cosine_similarity(
            self.tfidf_matrix[idx:idx+1], self.tfidf_matrix
        ).flatten()
        return _top_k_similar(sim_row, idx, top_k)

    def _get_similar_separate(self, idx, top_k=10):
        score = np.zeros(len(self.df))
        for feature, weight in self.feature_weights.items():
            matrix = self.feature_matrices[feature]
            sim_row = cosine_similarity(matrix[idx:idx+1], matrix).flatten()
            score += sim_row * weight
        return _top_k_similar(score, idx, top_k)

    def _get_similar_embedding(self, idx, top_k=10):
        sim_row = cosine_similarity(
            self.embeddings[idx:idx+1], self.embeddings
        ).flatten()
        return _top_k_similar(sim_row, idx, top_k)
    
    def _get_similar_hybrid(self, idx, top_k=10, weights=(0.5, 0.5)):
        sim_tfidf = cosine_similarity(
            self.tfidf_matrix[idx:idx+1], self.tfidf_matrix
        ).flatten()
        sim_emb = cosine_similarity(
            self.embeddings[idx:idx+1], self.embeddings
        ).flatten()
        sim_tfidf = _normalize(sim_tfidf)
        sim_emb = _normalize(sim_emb)
        w_tfidf, w_emb = weights
        hybrid_score = (sim_tfidf * w_tfidf) + (sim_emb * w_emb)
        return _top_k_similar(hybrid_score, idx, top_k)
    
    def _compute_final_score(self, sim_scores, indices, alpha=0.8):
        ratings = self.df.loc[indices, 'weighted_rating'].values
        rating_min = self.df['weighted_rating'].min()
        rating_max = self.df['weighted_rating'].max()
        norm_ratings = (ratings - rating_min) / (rating_max - rating_min)
        return alpha * sim_scores + (1 - alpha) * norm_ratings

    def recommend(self, title, top_k=10, approach='combined', content_type=None, 
                  year=None, alpha=0.8, hybrid_weights=(0.5, 0.5)):
        matches = self.df[self.df['title'].str.lower() == title.lower()]
        if matches.empty:
            logger.warning("'%s' not found.", title)
            return None

        if content_type:
            matches = matches[matches['type'] == content_type]

        if year:
            matches = matches[matches['release_year'] == year]

        if matches.empty:
            logger.warning("'%s' not found with the given filters.", title)
            return None

        if len(matches) > 1:
            matches = matches.sort_values('popularity', ascending=False)
            logger.info(
                "Multiple matches for '%s'. Using most popular match. "
                "Specify content_type or year to disambiguate.\n%s",
                title, matches[['title', 'type', 'genres', 'release_year', 'popularity']].to_string()
            )

        idx = matches.index[0]

        approach_map = {
            'combined':  ([self.tfidf_matrix], self._get_similar_combined, {}),
            'separate':  ([self.feature_matrices], self._get_similar_separate, {}),
            'embedding': ([self.embeddings], self._get_similar_embedding, {}),
            'hybrid':    ([self.tfidf_matrix, self.embeddings], self._get_similar_hybrid, {'weights': hybrid_weights})
        }

        if approach not in approach_map:
            raise ValueError(f"Unknown approach: '{approach}'. Use: {list(approach_map.keys())}")
        
        required, method, kwargs = approach_map[approach]
        if any(r is None for r in required):
            raise ValueError(f"'{approach}' requires data that hasn't been provided.")
        
        indices, scores = method(idx, top_k, **kwargs)

        result_cols = ['title', 'type', 'genres', 'vote_average']
        if 'weighted_rating' in self.df.columns:
            result_cols.append('weighted_rating')

        results = self.df.loc[indices, result_cols].copy()
        results['similarity'] = scores
        if 'weighted_rating' in self.df.columns:
            results['final_score'] = self._compute_final_score(scores, indices, alpha)
            results = results.sort_values('final_score', ascending=False)
        return results


class Evaluator:
    def __init__(self, df, tfidf_matrix, feature_matrices, feature_weights,
                 embeddings, hybrid_weights=(0.5,0.5), seed=42):
        self.df = df
        self.tfidf_matrix = tfidf_matrix
        self.feature_matrices = feature_matrices
        self.feature_weights = feature_weights
        self.embeddings = embeddings
        self.hybrid_weights = hybrid_weights
        self.seed = seed

    def _get_sample_indices(self, sample_size, rng=None):
        if rng is None:
            rng = np.random.RandomState(self.seed)
        return rng.choice(len(self.df), sample_size, replace=False)

    def _compute_multi_similarity(self, idx):
        score = np.zeros(len(self.df))
        for feature, weight in self.feature_weights.items():
            matrix = self.feature_matrices[feature]
            sim = cosine_similarity(matrix[idx:idx+1], matrix).flatten()
            score += sim * weight
        score[idx] = -1
        return score


    def test_h1_discriminative_power(self, sample_size=500, top_k=10):
        """
        H0: No significant difference in score variance between
            genre-only and multi-feature approaches
        H1: Multi-feature produces higher score variance (more discriminative)
        """
        rng = np.random.RandomState(self.seed)
        sample_idx = rng.choice(len(self.df), sample_size, replace=False)
        genre_matrix = self.feature_matrices['genres']

        # Batch genre similarity
        genre_sims = cosine_similarity(genre_matrix[sample_idx], genre_matrix)
        for i, idx in enumerate(sample_idx):
            genre_sims[i, idx] = -1

        genre_variances = []
        for i in range(len(sample_idx)):
            top_genre = np.sort(genre_sims[i])[::-1][:top_k]
            genre_variances.append(np.var(top_genre))

        # Multi-feature still needs per-item due to weighted combination
        multi_variances = []
        for idx in sample_idx:
            multi_sim = self._compute_multi_similarity(idx)
            top_multi = np.sort(multi_sim)[::-1][:top_k]
            multi_variances.append(np.var(top_multi))

        stat, p_value = stats.wilcoxon(multi_variances, genre_variances, alternative='greater')

        logger.info("Hypothesis 1: Discriminative Power")
        logger.info("Mean genre-only variance:    %.6f", np.mean(genre_variances))
        logger.info("Mean multi-feature variance: %.6f", np.mean(multi_variances))
        logger.info("Wilcoxon statistic: %.2f", stat)
        logger.info("P-value: %.2e", p_value)
        logger.info("Significant at α=0.05 (one-tailed, greater): %s", p_value < 0.05)

        return {
            'genre_variances': genre_variances,
            'multi_variances': multi_variances,
            'statistic': stat,
            'p_value': p_value
        }

    def test_h2_feature_independence(self, sample_size=1000):
        """
        H0: Feature similarity rankings are not correlated (independent)
        H1: Features are partially correlated (0.2-0.5) — overlapping but distinct
        """
        rng = np.random.RandomState(self.seed)
        n = len(self.df)
        pairs_i = rng.choice(n, sample_size, replace=False)
        pairs_j = rng.choice(n, sample_size, replace=False)

        original_size = sample_size
        mask = pairs_i != pairs_j
        pairs_i, pairs_j = pairs_i[mask], pairs_j[mask]
        if len(pairs_i) < original_size:
            logger.info("Filtered %d self-pairs, %d pairs remaining",
                        original_size - len(pairs_i), len(pairs_i))

        features = list(self.feature_weights.keys())
        feature_sims = {}
        for feature in features:
            matrix = self.feature_matrices[feature]
            sims = [
                cosine_similarity(matrix[i:i+1], matrix[j:j+1]).flatten()[0]
                for i, j in zip(pairs_i, pairs_j)
            ]
            feature_sims[feature] = np.array(sims)

        logger.info("Hypothesis 2: Feature Independence")
        logger.info("Sample pairs: %d", len(pairs_i))
        logger.info("%-30s %10s %12s %s", "Pair", "Spearman ρ", "P-value", "Interpretation")
        logger.info("-" * 75)

        results = {}
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                f1, f2 = features[i], features[j]

                if np.std(feature_sims[f1]) == 0 or np.std(feature_sims[f2]) == 0:
                    rho, p_val = np.nan, np.nan
                    interp = "Undefined (constant input - feature too sparse)"
                else:
                    rho, p_val = stats.spearmanr(feature_sims[f1], feature_sims[f2])
                    if abs(rho) < 0.2:
                        interp = "Independent"
                    elif abs(rho) < 0.5:
                        interp = "Partially correlated (supports H2)"
                    elif abs(rho) < 0.7:
                        interp = "Moderately correlated"
                    else:
                        interp = "Highly correlated (redundant)"

                logger.info("%-30s %10s %12s %s", f1 + ' vs ' + f2,
                            f"{rho:.4f}" if not np.isnan(rho) else "nan",
                            f"{p_val:.2e}" if not np.isnan(p_val) else "nan",
                            interp)
                results[f"{f1}_vs_{f2}"] = {'rho': rho, 'p_value': p_val, 'interpretation': interp}

        valid_p_values = [v['p_value'] for v in results.values() if not np.isnan(v['p_value'])]
        n_tests = len(valid_p_values)
        if n_tests > 0:
            corrected_alpha = 0.05 / n_tests
            logger.info("Bonferroni-corrected alpha (k=%d): %.4f", n_tests, corrected_alpha)

        return results

    def test_h3_ranking_quality(self, sample_size=500, top_k=20):
        """
        H0: Rankings by raw vote_average and weighted rating are not 
        significantly different
        H1: Weighted rating produces significantly different rankings 
        favoring higher vote confidence
        """
        if 'weighted_rating' not in self.df.columns:
            raise ValueError(
                "Column 'weighted_rating' not found. "
                "Call ContentRecommender.compute_weighted_rating() first."
            )

        rng = np.random.RandomState(self.seed)
        sample_idx = self._get_sample_indices(sample_size, rng)

        tau_scores = []
        raw_avg_votes = []
        weighted_avg_votes = []

        for idx in sample_idx:
            multi_sim = self._compute_multi_similarity(idx)
            top_indices, _ = _top_k_similar(multi_sim, idx, top_k)

            raw_ranking = self.df.loc[top_indices, 'vote_average'].values.argsort()[::-1]
            weighted_ranking = self.df.loc[top_indices, 'weighted_rating'].values.argsort()[::-1]

            tau, _ = stats.kendalltau(raw_ranking, weighted_ranking)
            if not np.isnan(tau):
                tau_scores.append(tau)

            raw_top5 = top_indices[raw_ranking[:5]]
            weighted_top5 = top_indices[weighted_ranking[:5]]
            raw_avg_votes.append(self.df.loc[raw_top5, 'vote_count'].mean())
            weighted_avg_votes.append(self.df.loc[weighted_top5, 'vote_count'].mean())

        stat, p_value = stats.wilcoxon(weighted_avg_votes, raw_avg_votes, alternative='greater')

        logger.info("Hypothesis 3: Ranking Quality")
        logger.info("Kendall's Tau (raw vs weighted ranking):")
        logger.info("  Mean tau: %.4f", np.mean(tau_scores))
        logger.info("  (1.0 = identical, 0.0 = no correlation)")
        logger.info("Avg vote_count in top-5 recommendations:")
        logger.info("  Raw vote_average ranking:  %.1f", np.mean(raw_avg_votes))
        logger.info("  Weighted rating ranking:   %.1f", np.mean(weighted_avg_votes))
        logger.info("Wilcoxon test (vote confidence):")
        logger.info("  Statistic: %.2f", stat)
        logger.info("  P-value: %.2e", p_value)
        logger.info("  Significant at α=0.05 (one-tailed, greater): %s", p_value < 0.05)

        return {
            'tau_scores': tau_scores,
            'raw_avg_votes': raw_avg_votes,
            'weighted_avg_votes': weighted_avg_votes,
            'statistic': stat,
            'p_value': p_value
        }
    
    def test_h4_method_distinctness(self, sample_size=500, top_k=10):
        """
        H4: Measures the overlap between TF-IDF and Embedding recommendations.
        Metric: Jaccard Similarity (Intersection over Union).
        Low Jaccard = The models are finding DIFFERENT movies (High Novelty).
        """
        if self.embeddings is None or self.tfidf_matrix is None:
            raise ValueError("H4 requires both TF-IDF matrix and Embeddings.")

        rng = np.random.RandomState(self.seed)
        sample_idx = rng.choice(len(self.df), sample_size, replace=False)
        jaccard_scores = []

        for idx in sample_idx:
            tfidf_sim = cosine_similarity(self.tfidf_matrix[idx:idx+1], self.tfidf_matrix).flatten()
            tfidf_top_idx, _ = _top_k_similar(tfidf_sim, idx, top_k)
            tfidf_top = set(tfidf_top_idx)

            emb_sim = cosine_similarity(self.embeddings[idx:idx+1], self.embeddings).flatten()
            emb_top_idx, _ = _top_k_similar(emb_sim, idx, top_k)
            emb_top = set(emb_top_idx)

            intersection = len(tfidf_top.intersection(emb_top))
            union = len(tfidf_top.union(emb_top))
            jaccard_scores.append(intersection / union)

        mean_jaccard = np.mean(jaccard_scores)
        
        logger.info("Hypothesis 4: Method Distinctness (Novelty)")
        logger.info("Mean Jaccard Similarity (TF-IDF vs SBERT): %.4f", mean_jaccard)
        if mean_jaccard < 0.2:
            logger.info("Interpretation: High Distinctness. SBERT is finding completely new items.")
        elif mean_jaccard > 0.8:
            logger.info("Interpretation: High Redundancy. SBERT adds little value over TF-IDF.")
        else:
            logger.info("Interpretation: Moderate Overlap. Complementary methods.")
            
        return jaccard_scores

    def test_h5_score_distribution(self, sample_size=500):
        """
        H5: Tests if SBERT scores significantly differ in distribution from TF-IDF.
        Test: Kolmogorov-Smirnov (KS) Test.
        """
        if self.embeddings is None or self.tfidf_matrix is None:
            raise ValueError("H5 requires both TF-IDF matrix and Embeddings.")

        rng = np.random.RandomState(self.seed)
        sample_idx = rng.choice(len(self.df), sample_size, replace=False)
        
        tfidf_dist = []
        emb_dist = []

        for idx in sample_idx:
            t_sim = cosine_similarity(self.tfidf_matrix[idx:idx+1], self.tfidf_matrix).flatten()
            e_sim = cosine_similarity(self.embeddings[idx:idx+1], self.embeddings).flatten()
            
            t_sim[idx] = -1
            e_sim[idx] = -1
            tfidf_dist.extend(np.sort(t_sim)[::-1][:50])
            emb_dist.extend(np.sort(e_sim)[::-1][:50])

        stat, p_value = stats.ks_2samp(tfidf_dist, emb_dist)

        logger.info("Hypothesis 5: Score Distribution (The 'Loudness' Problem)")
        logger.info("Mean Top-50 TF-IDF Score:   %.4f", np.mean(tfidf_dist))
        logger.info("Mean Top-50 Embedding Score: %.4f", np.mean(emb_dist))
        logger.info("KS Statistic: %.4f", stat)
        logger.info("P-value: %.2e", p_value)
        logger.info("Significant Difference: %s", p_value < 0.05)

        if np.mean(emb_dist) > np.mean(tfidf_dist) and p_value < 0.05:
            logger.info("Result: Embeddings are significantly 'louder'. Normalization is recommended for Hybrid.")

        return {'statistic': stat, 'p_value': p_value}

    def test_h6_hybrid_quality_lift(self, sample_size=500, top_k=10):
        """
        H6: Tests if Hybrid recommendations have higher average user ratings 
        than the single-model approaches.
        Test: Wilcoxon Signed-Rank Test.
        """
        if 'weighted_rating' not in self.df.columns:
            raise ValueError("Column 'weighted_rating' missing. Run compute_weighted_rating() first.")

        rng = np.random.RandomState(self.seed)
        sample_idx = rng.choice(len(self.df), sample_size, replace=False)
        
        tfidf_ratings = []
        hybrid_ratings = []

        w_tfidf, w_emb = self.hybrid_weights

        for idx in sample_idx:
            t_sim = cosine_similarity(self.tfidf_matrix[idx:idx+1], self.tfidf_matrix).flatten()
            t_indices, _ = _top_k_similar(t_sim, idx, top_k)
            tfidf_ratings.append(self.df.loc[t_indices, 'weighted_rating'].mean())

            e_sim = cosine_similarity(self.embeddings[idx:idx+1], self.embeddings).flatten()

            h_sim = (_normalize(t_sim) * w_tfidf) + (_normalize(e_sim) * w_emb)
            h_indices, _ = _top_k_similar(h_sim, idx, top_k)
            hybrid_ratings.append(self.df.loc[h_indices, 'weighted_rating'].mean())

        stat, p_value = stats.wilcoxon(hybrid_ratings, tfidf_ratings)

        logger.info("Hypothesis 6: Hybrid Quality Lift")
        logger.info("Avg Rating (TF-IDF Top-%d): %.2f", top_k, np.mean(tfidf_ratings))
        logger.info("Avg Rating (Hybrid Top-%d): %.2f", top_k, np.mean(hybrid_ratings))
        logger.info("Wilcoxon Statistic: %.2f", stat)
        logger.info("P-value: %.2e", p_value)

        if p_value < 0.05 and np.mean(hybrid_ratings) > np.mean(tfidf_ratings):
            logger.info("Result: Hybrid strategy significantly improves recommendation quality.")
        else:
            logger.info("Result: No significant rating improvement observed.")

        return {'statistic': stat, 'p_value': p_value}