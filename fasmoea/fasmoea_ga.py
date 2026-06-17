"""
NSGA-II setup for FAS-MOEA.

Chromosomes are unique integer indices into a per-user candidate pool. pymoo
minimizes, so all three maximization objectives are negated.
"""

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.optimize import minimize


def _rng_from_state(random_state=None, seed=None):
    if random_state is None:
        return np.random.default_rng(seed)
    if hasattr(random_state, "integers"):
        return np.random.default_rng(int(random_state.integers(0, 2**31 - 1)))
    return np.random.default_rng(int(random_state.randint(0, 2**31 - 1)))


class FASCrossover(Crossover):
    """One-point crossover that preserves unique genes where possible."""

    def __init__(self, prob=0.9):
        super().__init__(n_parents=2, n_offsprings=2, prob=prob)

    @staticmethod
    def _make_child(parent_a, parent_b, rng, n_var):
        if n_var < 2:
            return parent_a.copy()
        split = int(rng.integers(1, n_var))
        seen = set(parent_a[:split].tolist())
        fill = [gene for gene in parent_b if gene not in seen]
        child = np.concatenate([parent_a[:split], np.asarray(fill[: n_var - split])])
        if len(child) < n_var:
            fallback = [gene for gene in parent_a if gene not in set(child.tolist())]
            child = np.concatenate([child, np.asarray(fallback[: n_var - len(child)])])
        if len(child) < n_var:
            return parent_a.copy()
        return child[:n_var]

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        _, n_matings, n_var = X.shape
        rng = _rng_from_state(random_state)
        Y = np.empty((self.n_offsprings, n_matings, n_var), dtype=X.dtype)
        for k in range(n_matings):
            parent_a = X[0, k].astype(int)
            parent_b = X[1, k].astype(int)
            Y[0, k] = self._make_child(parent_a, parent_b, rng, n_var)
            Y[1, k] = self._make_child(parent_b, parent_a, rng, n_var)
        return Y


class FASMutation(Mutation):
    """
    Replace the lowest-scored selected item with the highest-novelty item outside
    the chromosome.
    """

    def __init__(self, prob=0.1, ncf_scores=None, novelty_scores=None, pool_size=None):
        super().__init__(prob=1.0)
        self.prob_ind = prob
        self.ncf_scores = ncf_scores
        self.novelty_scores = novelty_scores
        self.pool_size = pool_size

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        rng = _rng_from_state(random_state)
        Y = X.copy().astype(int)
        pool_size = self.pool_size
        if pool_size is None:
            if self.ncf_scores is not None:
                pool_size = len(self.ncf_scores)
            elif self.novelty_scores is not None:
                pool_size = len(self.novelty_scores)
            else:
                pool_size = problem.xu + 1

        for i in range(len(Y)):
            if rng.random() > self.prob_ind:
                continue
            chromosome = Y[i]
            if len(chromosome) == 0:
                continue

            if self.ncf_scores is not None:
                worst_pos = int(np.argmin(self.ncf_scores[chromosome]))
            else:
                worst_pos = int(rng.integers(0, len(chromosome)))

            in_list = set(chromosome.tolist())
            candidates_outside = [idx for idx in range(pool_size) if idx not in in_list]
            if not candidates_outside:
                continue

            if self.novelty_scores is not None:
                novelty_values = self.novelty_scores[candidates_outside]
                replacement = candidates_outside[int(np.argmax(novelty_values))]
            else:
                replacement = int(rng.choice(candidates_outside))
            Y[i, worst_pos] = replacement
        return Y


class FASProblem(Problem):
    """pymoo wrapper for one user's FAS-MOEA recommendation problem."""

    def __init__(self, user_idx, candidate_pool, objectives, N=10):
        self.user_idx = user_idx
        self.candidate_pool = candidate_pool
        self.objectives = objectives
        self.N = N
        super().__init__(
            n_var=N,
            n_obj=3,
            n_ieq_constr=0,
            xl=0,
            xu=len(candidate_pool) - 1,
            vtype=int,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.zeros((len(X), 3), dtype=float)
        pool_size = len(self.candidate_pool)
        for row_idx, chromosome in enumerate(X):
            rec_list = [
                self.candidate_pool[int(gene)]
                for gene in chromosome
                if 0 <= int(gene) < pool_size
            ]
            scores = self.objectives.compute_all(self.user_idx, rec_list)
            F[row_idx, 0] = -scores["accuracy"]
            F[row_idx, 1] = -scores["fairness"]
            F[row_idx, 2] = -scores["serendipity"]
        out["F"] = F


def run_fasmoea_for_user(
    user_idx,
    candidate_pool,
    objectives,
    R_hat_user,
    N=10,
    pop_size=50,
    n_generations=100,
    seed=42,
):
    """
    Run FAS-MOEA for one user.

    Returns a best compromise recommendation list of item_idx values and Pareto
    objective logs with positive accuracy/fairness/serendipity scores.
    """
    pool_size = len(candidate_pool)
    if pool_size == 0:
        return [], []
    if pool_size <= N:
        rec_list = [int(item_idx) for item_idx in candidate_pool]
        return rec_list, [objectives.compute_all(user_idx, rec_list)]

    novelty_scores_pool = np.asarray(
        [objectives._novelty_score(item_idx) for item_idx in candidate_pool],
        dtype=np.float32,
    )
    ncf_scores_pool = np.asarray(
        [float(R_hat_user[item_idx]) if item_idx < len(R_hat_user) else 0.0
         for item_idx in candidate_pool],
        dtype=np.float32,
    )

    actual_pop_size = max(pop_size, 2)
    rng = np.random.default_rng(seed + int(user_idx))
    initial = []

    top_n_indices = np.argsort(-ncf_scores_pool)[:N]
    initial.append(top_n_indices.tolist())

    novelty_seed = np.argsort(-novelty_scores_pool)[:N]
    if len(set(novelty_seed.tolist())) == N:
        initial.append(novelty_seed.tolist())

    while len(initial) < actual_pop_size:
        initial.append(rng.choice(pool_size, size=N, replace=False).tolist())

    pop0 = Population.new("X", np.asarray(initial[:actual_pop_size], dtype=int))
    problem = FASProblem(user_idx, candidate_pool, objectives, N=N)

    algorithm = NSGA2(
        pop_size=actual_pop_size,
        sampling=pop0,
        crossover=FASCrossover(prob=0.9),
        mutation=FASMutation(
            prob=0.1,
            ncf_scores=ncf_scores_pool,
            novelty_scores=novelty_scores_pool,
            pool_size=pool_size,
        ),
        eliminate_duplicates=True,
    )

    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", n_generations),
        seed=seed + int(user_idx),
        verbose=False,
    )

    if res.X is None or len(res.X) == 0:
        best = [int(candidate_pool[idx]) for idx in top_n_indices]
        return best, []

    pareto_F = np.atleast_2d(res.F)
    pareto_X = np.atleast_2d(res.X)

    ideal = np.min(pareto_F, axis=0)
    nadir = np.max(pareto_F, axis=0)
    spread = np.where(nadir - ideal == 0, 1.0, nadir - ideal)
    normalized = (pareto_F - ideal) / spread
    best_idx = int(np.argmin(np.linalg.norm(normalized, axis=1)))

    best_chromosome = pareto_X[best_idx].astype(int)
    best_rec_list = [
        int(candidate_pool[idx])
        for idx in best_chromosome
        if 0 <= idx < pool_size
    ]

    pareto_metrics = [
        {
            "accuracy": float(-row[0]),
            "fairness": float(-row[1]),
            "serendipity": float(-row[2]),
        }
        for row in pareto_F
    ]
    return best_rec_list, pareto_metrics
