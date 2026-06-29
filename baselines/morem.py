"""
baselines/morem.py
Baseline 4: MOREM (Zhang et al., 2024)
"MOREM: An evolutionary multitasking optimization algorithm for multi-objective recommendations."

Two objectives: Accuracy (NCF R_hat) and Diversity (ILD).
Evolutionary multitasking: two NSGA-II populations with knowledge transfer.
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
from pymoo.core.evaluator import Evaluator

from baselines.base_recommender import BaseRecommender
from baselines.cf_utils import get_genre_matrix, get_candidate_pool
from data_loader import GENRE_COLS


class MOREMProblem(Problem):
    """
    MOREM: 2-objective problem.
    f1 = -accuracy (mean NCF score)
    f2 = -diversity (unique genres / list length = ILD)
    """

    def __init__(self, candidate_pool, R_hat_row, genre_matrix, n=10):
        self.candidate_pool = candidate_pool
        self.R_hat_row = R_hat_row
        self.genre_matrix = genre_matrix
        self.n = n
        self.pool_size = len(candidate_pool)

        super().__init__(
            n_var=n, n_obj=2,
            xl=0, xu=self.pool_size - 1, vtype=int,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        pop_size = len(X)
        F = np.zeros((pop_size, 2))

        for p in range(pop_size):
            indices = X[p].astype(int) % self.pool_size
            items = [self.candidate_pool[idx] for idx in indices]

            # f1: Accuracy
            scores = [self.R_hat_row[i] for i in items]
            F[p, 0] = -np.mean(scores)

            # f2: Diversity (ILD = unique genres / list length)
            all_genres = set()
            for i in items:
                genres_present = np.where(self.genre_matrix[i] > 0)[0]
                all_genres.update(genres_present)
            ild = len(all_genres) / len(items) if len(items) > 0 else 0
            F[p, 1] = -ild

        out["F"] = F


class MOREM(BaseRecommender):
    """
    MOREM baseline: NCF R_hat + evolutionary multitasking (two NSGA-II populations).
    Knowledge transfer every K=5 generations.
    """

    def __init__(self, data, R_hat, pop_size=50, n_gen=150, transfer_k=5,
                 top_n=10, candidate_pool_size=200):
        super().__init__(data, name="MOREM", top_n=top_n, candidate_pool_size=candidate_pool_size)
        self.R_hat = R_hat
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.transfer_k = transfer_k
        self.genre_matrix = None

    def fit(self, seed=42):
        np.random.seed(seed)
        self.genre_matrix = get_genre_matrix(self.data)
        print(f"  [{self.name}] Using pretrained NCF R_hat. Genre matrix built.")

    def recommend(self, user_idx, n=10):
        candidates = get_candidate_pool(
            user_idx, self.R_hat[user_idx], self.train_history,
            pool_size=self.candidate_pool_size
        )
        if len(candidates) < n:
            return candidates

        problem = MOREMProblem(candidates, self.R_hat[user_idx], self.genre_matrix, n=n)

        # Evolutionary multitasking: run two tasks with manual knowledge transfer
        # Task A focuses on accuracy, Task B focuses on diversity
        # Both use the same problem but we transfer solutions between them

        # Initialize populations
        X_a = np.random.randint(0, len(candidates), size=(self.pop_size, n))
        X_b = np.random.randint(0, len(candidates), size=(self.pop_size, n))

        pop_a = Population.new("X", X_a)
        pop_b = Population.new("X", X_b)

        algo_a = NSGA2(
            pop_size=self.pop_size,
            sampling=pop_a,
            crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
            mutation=PM(eta=20, vtype=float, repair=RoundingRepair()),
            eliminate_duplicates=True,
        )

        algo_b = NSGA2(
            pop_size=self.pop_size,
            sampling=pop_b,
            crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
            mutation=PM(eta=20, vtype=float, repair=RoundingRepair()),
            eliminate_duplicates=True,
        )

        # Run in chunks with knowledge transfer
        chunk_size = self.transfer_k
        n_chunks = self.n_gen // chunk_size

        # Initial run for first chunk
        res_a = minimize(problem, algo_a, termination=("n_gen", chunk_size), seed=None, verbose=False)
        res_b = minimize(problem, algo_b, termination=("n_gen", chunk_size), seed=None, verbose=False)

        for chunk in range(1, n_chunks):
            # Knowledge transfer: copy top-3 from each task to the other
            if res_a.X is not None and res_b.X is not None:
                F_a = res_a.F
                F_b = res_b.F

                # Top-3 by accuracy from A (lowest f1)
                if len(F_a) >= 3:
                    top3_a_idx = np.argsort(F_a[:, 0])[:3]
                    transfer_a = res_a.X[top3_a_idx]
                else:
                    transfer_a = res_a.X

                # Top-3 by diversity from B (lowest f2)
                if len(F_b) >= 3:
                    top3_b_idx = np.argsort(F_b[:, 1])[:3]
                    transfer_b = res_b.X[top3_b_idx]
                else:
                    transfer_b = res_b.X

                # Inject into opposite populations
                if res_b.X is not None and len(res_b.X) > 3:
                    new_X_b = np.vstack([res_b.X[3:], transfer_a])
                    pop_b_new = Population.new("X", new_X_b[:self.pop_size])
                else:
                    pop_b_new = Population.new("X", res_b.X if res_b.X is not None else X_b)

                if res_a.X is not None and len(res_a.X) > 3:
                    new_X_a = np.vstack([res_a.X[3:], transfer_b])
                    pop_a_new = Population.new("X", new_X_a[:self.pop_size])
                else:
                    pop_a_new = Population.new("X", res_a.X if res_a.X is not None else X_a)

                algo_a_new = NSGA2(
                    pop_size=self.pop_size,
                    sampling=pop_a_new,
                    crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
                    mutation=PM(eta=20, vtype=float, repair=RoundingRepair()),
                    eliminate_duplicates=True,
                )
                algo_b_new = NSGA2(
                    pop_size=self.pop_size,
                    sampling=pop_b_new,
                    crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
                    mutation=PM(eta=20, vtype=float, repair=RoundingRepair()),
                    eliminate_duplicates=True,
                )

                res_a = minimize(problem, algo_a_new, termination=("n_gen", chunk_size), seed=None, verbose=False)
                res_b = minimize(problem, algo_b_new, termination=("n_gen", chunk_size), seed=None, verbose=False)
            else:
                break

        # Combine Pareto fronts from both tasks
        all_X = []
        all_F = []
        if res_a.X is not None:
            all_X.append(res_a.X)
            all_F.append(res_a.F)
        if res_b.X is not None:
            all_X.append(res_b.X)
            all_F.append(res_b.F)

        if not all_X:
            return candidates[:n]

        combined_X = np.vstack(all_X)
        combined_F = np.vstack(all_F)

        # Utopia distance selection on combined front
        ideal = combined_F.min(axis=0)
        F_max = combined_F.max(axis=0)
        F_range = F_max - ideal
        F_range = np.where(F_range == 0, 1, F_range)
        F_norm = (combined_F - ideal) / F_range
        distances = np.linalg.norm(F_norm, axis=1)
        best_idx = np.argmin(distances)

        best_chrom = combined_X[best_idx].astype(int) % len(candidates)
        return [candidates[idx] for idx in best_chrom]
