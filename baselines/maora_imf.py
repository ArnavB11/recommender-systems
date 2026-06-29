"""
baselines/maora_imf.py
Baseline 3: MaORA-IMF (Cui et al., 2021)
"An improved matrix factorization based model for many-objective optimization recommendation."

Four objectives: Recall, Novelty, Diversity, Accuracy.
Uses SVD (TruncatedSVD, rank=50) + NSGA-III.
"""

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize

from baselines.base_recommender import BaseRecommender
from baselines.cf_utils import (
    build_rating_matrix, svd_predict, get_genre_matrix,
    get_candidate_pool, get_test_relevant_items
)


class MaORAIMFProblem(Problem):
    """
    MaORA-IMF: 4-objective problem.
    Objectives (minimized):
        f1 = -recall (fraction of relevant test items in list)
        f2 = -novelty (log-scaled inverse popularity)
        f3 = -diversity (mean pairwise genre Jaccard distance)
        f4 = -accuracy (mean MF predicted rating)
    """

    def __init__(self, candidate_pool, R_hat_row, genre_matrix,
                 item_popularity, pop_max, relevant_items, n=10):
        self.candidate_pool = candidate_pool
        self.R_hat_row = R_hat_row
        self.genre_matrix = genre_matrix
        self.item_popularity = item_popularity
        self.pop_max = pop_max
        self.relevant_items = relevant_items
        self.n = n
        self.pool_size = len(candidate_pool)

        super().__init__(
            n_var=n, n_obj=4,
            xl=0, xu=self.pool_size - 1, vtype=int,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        pop_size = len(X)
        F = np.zeros((pop_size, 4))

        for p in range(pop_size):
            indices = X[p].astype(int) % self.pool_size
            items = [self.candidate_pool[idx] for idx in indices]
            item_set = set(items)

            # f1: Recall
            if self.relevant_items:
                hits = len(item_set & self.relevant_items)
                recall = hits / len(self.relevant_items)
            else:
                recall = 0.0
            F[p, 0] = -recall

            # f2: Novelty (log-scaled inverse popularity)
            novelties = []
            for i in items:
                pop = self.item_popularity.get(i, 1)
                novelties.append(np.log2(1.0 + self.pop_max / max(pop, 1)))
            max_possible = np.log2(1.0 + self.pop_max)
            F[p, 1] = -np.mean(novelties) / max_possible if max_possible > 0 else 0.0

            # f3: Diversity (mean pairwise genre Jaccard distance)
            n_items = len(items)
            if n_items < 2:
                diversity = 0.0
            else:
                total_dist = 0.0
                n_pairs = 0
                for i_idx in range(n_items):
                    g_i = set(np.where(self.genre_matrix[items[i_idx]] > 0)[0])
                    for j_idx in range(i_idx + 1, n_items):
                        g_j = set(np.where(self.genre_matrix[items[j_idx]] > 0)[0])
                        if not g_i and not g_j:
                            dist = 1.0
                        else:
                            inter = len(g_i & g_j)
                            union = len(g_i | g_j)
                            dist = 1.0 - inter / union if union > 0 else 1.0
                        total_dist += dist
                        n_pairs += 1
                diversity = total_dist / n_pairs if n_pairs > 0 else 0.0
            F[p, 2] = -diversity

            # f4: Accuracy (mean MF predicted rating)
            scores = [self.R_hat_row[i] for i in items]
            F[p, 3] = -np.mean(scores)

        out["F"] = F


class MaORAIMF(BaseRecommender):
    """MaORA-IMF baseline: SVD + NSGA-III (Recall + Novelty + Diversity + Accuracy)."""

    def __init__(self, data, n_gen=200, top_n=10, candidate_pool_size=200):
        super().__init__(data, name="MaORA-IMF", top_n=top_n, candidate_pool_size=candidate_pool_size)
        self.n_gen = n_gen
        self.R_hat = None
        self.genre_matrix = None
        self.test_relevant = None

    def fit(self, seed=42):
        np.random.seed(seed)
        print(f"  [{self.name}] Building SVD predicted ratings (rank=50)...")
        rating_matrix, _ = build_rating_matrix(self.data)
        self.R_hat = svd_predict(rating_matrix, rank=50)
        self.genre_matrix = get_genre_matrix(self.data)
        self.test_relevant = get_test_relevant_items(self.data)
        print(f"  [{self.name}] Fit complete. R_hat shape: {self.R_hat.shape}")

    def recommend(self, user_idx, n=10):
        candidates = get_candidate_pool(
            user_idx, self.R_hat[user_idx], self.train_history,
            pool_size=self.candidate_pool_size
        )
        if len(candidates) < n:
            return candidates

        relevant = self.test_relevant.get(user_idx, set())

        problem = MaORAIMFProblem(
            candidates, self.R_hat[user_idx], self.genre_matrix,
            self.data.item_popularity, self.data.pop_max,
            relevant, n=n
        )

        ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=6)
        pop_size = len(ref_dirs)  # 84

        algorithm = NSGA3(
            ref_dirs=ref_dirs,
            pop_size=pop_size,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
            mutation=PM(eta=20, vtype=float, repair=RoundingRepair()),
            eliminate_duplicates=True,
        )

        res = minimize(problem, algorithm, termination=("n_gen", self.n_gen), seed=None, verbose=False)

        if res.X is None or len(res.X) == 0:
            return candidates[:n]

        # Selection: utopia distance
        F = res.F
        ideal = F.min(axis=0)
        F_max = F.max(axis=0)
        F_range = F_max - ideal
        F_range = np.where(F_range == 0, 1, F_range)
        F_norm = (F - ideal) / F_range
        distances = np.linalg.norm(F_norm, axis=1)
        best_idx = np.argmin(distances)

        best_chrom = res.X[best_idx].astype(int) % len(candidates)
        return [candidates[idx] for idx in best_chrom]
