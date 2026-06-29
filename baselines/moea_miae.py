"""
baselines/moea_miae.py
Baseline 7: MOEA-MIAE (Chu & Tian, 2025)
"Multi-objective recommendation system utilizing a multi-population knowledge migration framework."

Three objectives: Accuracy (NCF R_hat), Novelty, Diversity (ILD).
Multi-population: 3 sub-populations with different biases + knowledge migration.
"""

import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from pymoo.core.population import Population
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from baselines.base_recommender import BaseRecommender
from baselines.cf_utils import get_genre_matrix, get_candidate_pool
from data_loader import GENRE_COLS


class MOEAMIAEProblem(Problem):
    """
    MOEA-MIAE: 3-objective problem.
    f1 = -accuracy (mean NCF score)
    f2 = -novelty (mean 1 - pop/pop_max)
    f3 = -diversity (ILD = unique genres / list length)
    """

    def __init__(self, candidate_pool, R_hat_row, genre_matrix,
                 item_popularity, pop_max, n=10):
        self.candidate_pool = candidate_pool
        self.R_hat_row = R_hat_row
        self.genre_matrix = genre_matrix
        self.item_popularity = item_popularity
        self.pop_max = pop_max
        self.n = n
        self.pool_size = len(candidate_pool)

        super().__init__(
            n_var=n, n_obj=3,
            xl=0, xu=self.pool_size - 1, vtype=int,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        pop_size = len(X)
        F = np.zeros((pop_size, 3))

        for p in range(pop_size):
            indices = X[p].astype(int) % self.pool_size
            items = [self.candidate_pool[idx] for idx in indices]

            # f1: Accuracy
            scores = [self.R_hat_row[i] for i in items]
            F[p, 0] = -np.mean(scores)

            # f2: Novelty
            novelties = [1.0 - self.item_popularity.get(i, 0) / self.pop_max for i in items]
            F[p, 1] = -np.mean(novelties)

            # f3: Diversity (ILD)
            all_genres = set()
            for i in items:
                genres_present = np.where(self.genre_matrix[i] > 0)[0]
                all_genres.update(genres_present)
            ild = len(all_genres) / len(items) if len(items) > 0 else 0
            F[p, 2] = -ild

        out["F"] = F


class MOEAMIAE(BaseRecommender):
    """
    MOEA-MIAE baseline: Multi-population NSGA-II with knowledge migration.
    3 sub-populations (accuracy, novelty, diversity bias).
    Migration every 20 generations (circular A→B→C→A).
    """

    def __init__(self, data, R_hat, sub_pop_size=40, n_gen=150,
                 migration_interval=20, migration_size=5,
                 top_n=10, candidate_pool_size=200):
        super().__init__(data, name="MOEA-MIAE", top_n=top_n, candidate_pool_size=candidate_pool_size)
        self.R_hat = R_hat
        self.sub_pop_size = sub_pop_size
        self.n_gen = n_gen
        self.migration_interval = migration_interval
        self.migration_size = migration_size
        self.genre_matrix = None

    def fit(self, seed=42):
        np.random.seed(seed)
        self.genre_matrix = get_genre_matrix(self.data)
        print(f"  [{self.name}] Using pretrained NCF R_hat. Genre matrix built.")

    def _get_top_by_objective(self, X, F, obj_idx, k):
        """Get top-k solutions by a specific objective (lower F = better)."""
        if len(F) < k:
            return X.copy()
        top_idx = np.argsort(F[:, obj_idx])[:k]
        return X[top_idx]

    def recommend(self, user_idx, n=10):
        candidates = get_candidate_pool(
            user_idx, self.R_hat[user_idx], self.train_history,
            pool_size=self.candidate_pool_size
        )
        if len(candidates) < n:
            return candidates

        pool_size = len(candidates)
        problem = MOEAMIAEProblem(
            candidates, self.R_hat[user_idx], self.genre_matrix,
            self.data.item_popularity, self.data.pop_max, n=n
        )

        # Initialize 3 sub-populations
        pops = [
            np.random.randint(0, pool_size, size=(self.sub_pop_size, n))
            for _ in range(3)
        ]

        # Run each sub-population's NSGA-II in chunks with migration
        n_chunks = self.n_gen // self.migration_interval
        results = [None, None, None]

        for chunk in range(n_chunks):
            for i in range(3):
                pop_init = Population.new("X", pops[i])
                algo = NSGA2(
                    pop_size=self.sub_pop_size,
                    sampling=pop_init,
                    crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
                    mutation=PM(eta=20, vtype=float, repair=RoundingRepair()),
                    eliminate_duplicates=True,
                )
                res = minimize(problem, algo, termination=("n_gen", self.migration_interval),
                               seed=None, verbose=False)
                results[i] = res
                if res.X is not None:
                    pops[i] = res.X.copy()

            # Knowledge migration: A→B, B→C, C→A (circular)
            for i in range(3):
                src = i
                dst = (i + 1) % 3
                if results[src] is not None and results[src].X is not None and results[src].F is not None:
                    # Top-k by the source pop's bias objective
                    top_solutions = self._get_top_by_objective(
                        results[src].X, results[src].F, src, self.migration_size
                    )
                    if len(pops[dst]) > self.migration_size:
                        # Replace worst solutions in destination
                        pops[dst] = np.vstack([pops[dst][self.migration_size:], top_solutions])
                        pops[dst] = pops[dst][:self.sub_pop_size]

        # Merge all populations and do non-dominated sort
        all_X = []
        all_F = []
        for i in range(3):
            if results[i] is not None and results[i].X is not None:
                all_X.append(results[i].X)
                all_F.append(results[i].F)

        if not all_X:
            return candidates[:n]

        combined_X = np.vstack(all_X)
        combined_F = np.vstack(all_F)

        # Non-dominated sort to get front 0
        nds = NonDominatedSorting()
        fronts = nds.do(combined_F, n_stop_if_ranked=len(combined_F))
        front0 = fronts[0]

        pareto_X = combined_X[front0]
        pareto_F = combined_F[front0]

        # Utopia distance selection
        ideal = pareto_F.min(axis=0)
        F_max = pareto_F.max(axis=0)
        F_range = F_max - ideal
        F_range = np.where(F_range == 0, 1, F_range)
        F_norm = (pareto_F - ideal) / F_range
        distances = np.linalg.norm(F_norm, axis=1)
        best_idx = np.argmin(distances)

        best_chrom = pareto_X[best_idx].astype(int) % pool_size
        return [candidates[idx] for idx in best_chrom]
