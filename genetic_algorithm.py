import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.population import Population

class OrderCrossover(Crossover):
    """
    Order Crossover (OX) for permutation chromosomes.
    """
    def __init__(self, prob=0.9):
        super().__init__(n_parents=2, n_offsprings=2, prob=prob)

    @staticmethod
    def _make_child(parent_a, parent_b, random_state):
        n_var = len(parent_a)
        if n_var < 2:
            return parent_a.copy()

        child = np.full(n_var, -1, dtype=parent_a.dtype)

        start, end = sorted(random_state.choice(n_var, size=2, replace=False))
        child[start:end + 1] = parent_a[start:end + 1]

        child_values = set(child[start:end + 1])
        fill_positions = [idx for idx in range(n_var) if child[idx] == -1]
        fill_values = [gene for gene in parent_b if gene not in child_values]

        for idx, gene in zip(fill_positions, fill_values):
            child[idx] = gene

        return child

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        _, n_matings, n_var = X.shape
        Y = np.empty((self.n_offsprings, n_matings, n_var), dtype=X.dtype)

        for k in range(n_matings):
            parent_a = X[0, k].astype(int)
            parent_b = X[1, k].astype(int)

            Y[0, k] = self._make_child(parent_a, parent_b, random_state)
            Y[1, k] = self._make_child(parent_b, parent_a, random_state)

        return Y


class SwapMutation(Mutation):
    """
    Swap mutation for permutation chromosomes.
    """
    def __init__(self, prob):
        super().__init__(prob=1.0)
        self.prob_gene = prob

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        Y = X.copy().astype(int)
        n_individuals, n_var = Y.shape

        if n_var < 2:
            return Y

        for i in range(n_individuals):
            for j in range(n_var):
                if random_state.random() < self.prob_gene:
                    swap_idx = random_state.integers(0, n_var)
                    while swap_idx == j:
                        swap_idx = random_state.integers(0, n_var)

                    Y[i, j], Y[i, swap_idx] = Y[i, swap_idx], Y[i, j]

        return Y




class RecommendationListProblem(Problem):
    """
    Custom pymoo Problem for optimizing a recommendation list of size N.
    
    Decision Variables:
    - An integer vector X of size N, where each element is an index in the candidate movie pool [0, K-1].
    
    Conflicting Objectives to Minimize (using negative values for maximization):
    1. f1 = -DOPM (Mean full DOPM score)
    2. f2 = -Serendipity (Mean full serendipity score)
    3. f3 = -Fairness (Mean HDB * exposure penalty * quality score)
    """
    def __init__(self, user_idx, candidate_pool, ncf_scores, dopm_recommender, serendipity_model, fairness_model, user_history, N=10):
        self.user_idx = user_idx
        self.candidate_pool = candidate_pool  # List of actual item_idx
        self.ncf_scores = ncf_scores          # NCF scores for this user across all items
        self.dopm = dopm_recommender
        self.serendipity = serendipity_model
        self.fairness_model = fairness_model
        self.user_history = user_history
        self.N = N
        
        # Precompute user preference vector for fast semantic relevance calculations
        self.user_pref_vector = self.serendipity.compute_user_preference_vector(user_history)
        
        super().__init__(
            n_var=N,
            n_obj=3,
            n_ieq_constr=0,
            xl=0,
            xu=len(candidate_pool) - 1,
            vtype=int
        )

    def _evaluate(self, X, out, *args, **kwargs):
        # X shape is (pop_size, N)
        pop_size = len(X)
        f1 = np.zeros(pop_size)
        f2 = np.zeros(pop_size)
        f3 = np.zeros(pop_size)
        
        for p in range(pop_size):
            # Map chromosome indices to actual movie indices in the candidate pool
            movie_pool_indices = X[p].astype(int)
            actual_movies = [self.candidate_pool[idx] for idx in movie_pool_indices]
            
            # 1. DOPM: average full DOPM score
            dopm_scores = [
                self.dopm.calculate_dopm(self.user_idx, m, self.ncf_scores[m])
                for m in actual_movies
            ]
            f1[p] = -np.mean(dopm_scores)
            
            # 2. Serendipity: average full serendipity score
            ser_scores = [self.serendipity.calculate_serendipity(self.user_history, self.user_pref_vector, m) for m in actual_movies]
            f2[p] = -np.mean(ser_scores)
            
            # 3. Fairness: average HDB, exposure penalty, and quality balance
            dopm_dict = {m: score for m, score in zip(actual_movies, dopm_scores)}
            ser_dict = {m: score for m, score in zip(actual_movies, ser_scores)}
            f3[p] = -self.fairness_model.compute_fairness(actual_movies, dopm_dict, ser_dict)
            
        # Pymoo minimizes all objectives
        out["F"] = np.column_stack([f1, f2, f3])

def run_nsga2_optimization(user_idx, candidate_pool, ncf_scores, dopm_recommender, 
                           serendipity_model, fairness_model, user_history, initial_population_variants, 
                           N=10, pop_size=50, n_generations=500):
    """
    Runs NSGA-II multi-objective optimization to find the Pareto optimal 
    recommendation lists for a single user.
    """
    pool_size = len(candidate_pool)
    if pool_size < N:
        # If pool size is too small, optimization is trivial: return the whole pool
        return candidate_pool, None
        
    problem = RecommendationListProblem(
        user_idx=user_idx,
        candidate_pool=candidate_pool,
        ncf_scores=ncf_scores,
        dopm_recommender=dopm_recommender,
        serendipity_model=serendipity_model,
        fairness_model=fairness_model,
        user_history=user_history,
        N=N
    )
    
    # 1. Build initial population from P0 variants
    initial_chromosomes = []
    
    # Helper to map actual movie indices back to pool indices
    movie_to_pool_idx = {movie: idx for idx, movie in enumerate(candidate_pool)}
    
    for variant in initial_population_variants:
        # Map movie list to pool indices
        mapped = [movie_to_pool_idx[m] for m in variant if m in movie_to_pool_idx]
        # Pad or truncate to exact list size N
        if len(mapped) < N:
            remaining = list(set(range(pool_size)) - set(mapped))
            mapped += list(np.random.choice(remaining, size=N - len(mapped), replace=False))
        else:
            mapped = mapped[:N]
        initial_chromosomes.append(mapped)
        
    # Fill remaining population randomly
    while len(initial_chromosomes) < pop_size:
        initial_chromosomes.append(list(np.random.choice(pool_size, size=N, replace=False)))
        
    initial_chromosomes = np.array(initial_chromosomes)
    
    # Convert to pymoo Population object
    pop = Population.new("X", initial_chromosomes)
    
    # 2. Setup NSGA2 Algorithm
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=pop,  # Seed with our high-quality PSNR population
        crossover=OrderCrossover(prob=0.9),
        mutation=SwapMutation(prob=1.0/N),
        eliminate_duplicates=True
    )
    
    # 3. Optimize
    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', n_generations),
        seed=42,
        verbose=True
    )
    
    # 4. Extract Pareto frontier and select a balanced recommendation list
    pareto_solutions = res.X
    pareto_fitness = res.F  # Shape: (num_solutions, 3) -> [-DOPM, -Ser, -Fairness]
    
    if pareto_solutions is None or len(pareto_solutions) == 0:
        # Fallback to the first P0 variant
        return initial_population_variants[0], None
        
    # To choose a single "best balanced" solution from the Pareto frontier,
    # we compute the compromise solution using a simple multi-criteria decision making heuristic:
    # We find the solution closest to the ideal point (minimum of each objective).
    ideal_point = np.min(pareto_fitness, axis=0)
    max_point = np.max(pareto_fitness, axis=0)
    
    # Avoid division by zero
    range_point = max_point - ideal_point
    range_point = np.where(range_point == 0, 1.0, range_point)
    
    # Normalized distance to the ideal point
    normalized_fitness = (pareto_fitness - ideal_point) / range_point
    distances = np.linalg.norm(normalized_fitness, axis=1)
    best_idx = np.argmin(distances)
    
    best_chromosome = pareto_solutions[best_idx].astype(int)
    best_movie_list = [candidate_pool[idx] for idx in best_chromosome]
    
    # Return best list and full Pareto fitness logs for analysis
    pareto_metrics = []
    for f in pareto_fitness:
        pareto_metrics.append({
            "dopm": float(-f[0]),
            "serendipity": float(-f[1]),
            "fairness": float(-f[2])
        })
    return best_movie_list, pareto_metrics


class MSRSIVTargetedMutation(Mutation):
    """
    Targeted mutation for permutation chromosomes in MSRS IV.
    Replaces the item with the lowest NCF score with the highest-novelty
    item outside the list that satisfies a minimum relevance (NCF score) threshold.
    """
    def __init__(self, prob, ncf_scores, novelty_scores, pool_size, relevance_threshold=None):
        super().__init__(prob=1.0)
        self.prob_gene = prob
        self.ncf_scores = ncf_scores
        self.novelty_scores = novelty_scores
        self.pool_size = pool_size
        self.relevance_threshold = relevance_threshold if relevance_threshold is not None else np.mean(ncf_scores)

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        Y = X.copy().astype(int)
        n_individuals, n_var = Y.shape

        for i in range(n_individuals):
            if random_state.random() < self.prob_gene:
                rec_list = Y[i].astype(int)
                rec_ncf = self.ncf_scores[rec_list]
                lowest_idx_in_rec = np.argmin(rec_ncf)

                # Identify items outside the list
                outside_mask = np.ones(self.pool_size, dtype=bool)
                outside_mask[rec_list] = False
                outside_indices = np.where(outside_mask)[0]

                # Filter outside items by the relevance constraint
                relevant_outside = outside_indices[self.ncf_scores[outside_indices] >= self.relevance_threshold]

                if len(relevant_outside) == 0:
                    relevant_outside = outside_indices

                if len(relevant_outside) > 0:
                    outside_novelties = self.novelty_scores[relevant_outside]
                    highest_novelty_idx_outside = relevant_outside[np.argmax(outside_novelties)]
                    Y[i, lowest_idx_in_rec] = highest_novelty_idx_outside

        return Y


def run_nsga2_optimization_msrs_iv(user_idx, candidate_pool, ncf_scores, dopm_recommender, 
                                  serendipity_model, fairness_model, user_history, item_popularity, max_popularity,
                                  N=10, pop_size=50, n_generations=100, seed=42):
    """
    Runs NSGA-II multi-objective optimization to find the Pareto optimal 
    recommendation lists for a single user (MSRS IV version).
    """
    pool_size = len(candidate_pool)
    if pool_size < N:
        return candidate_pool, None
        
    problem = RecommendationListProblem(
        user_idx=user_idx,
        candidate_pool=candidate_pool,
        ncf_scores=ncf_scores,
        dopm_recommender=dopm_recommender,
        serendipity_model=serendipity_model,
        fairness_model=fairness_model,
        user_history=user_history,
        N=N
    )
    
    # 1. Compute novelty scores for candidate pool
    novelty_scores_pool = np.zeros(pool_size, dtype=np.float32)
    max_pop = max_popularity if max_popularity > 1 else 1
    for idx, item in enumerate(candidate_pool):
        pop = item_popularity.get(item, 0)
        novelty_scores_pool[idx] = float(1.0 - (np.log(pop + 1) / np.log(max_pop + 1)))
        
    ncf_scores_pool = np.asarray(
        [float(ncf_scores[item]) if item < len(ncf_scores) else 0.0 for item in candidate_pool],
        dtype=np.float32
    )
    
    # 2. Build initial population from seeds
    initial_chromosomes = []
    
    # Seed 1: Top NCF-based recommendation list (Anchor for accuracy)
    top_ncf_indices = np.argsort(-ncf_scores_pool)[:N]
    initial_chromosomes.append(top_ncf_indices.tolist())
    
    # Seed 2: Top popularity-novelty recommendation list (Anchor for long-tail)
    top_novelty_indices = np.argsort(-novelty_scores_pool)[:N]
    if len(set(top_novelty_indices.tolist())) == N:
        initial_chromosomes.append(top_novelty_indices.tolist())
        
    # Seed 3: Fill remaining population randomly
    rng = np.random.default_rng(seed + int(user_idx))
    while len(initial_chromosomes) < pop_size:
        initial_chromosomes.append(list(rng.choice(pool_size, size=N, replace=False)))
        
    initial_chromosomes = np.array(initial_chromosomes, dtype=int)
    
    # Convert to pymoo Population object
    pop = Population.new("X", initial_chromosomes)
    
    # 3. Setup NSGA2 Algorithm
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=pop,
        crossover=OrderCrossover(prob=0.9),
        mutation=MSRSIVTargetedMutation(
            prob=0.1,
            ncf_scores=ncf_scores_pool,
            novelty_scores=novelty_scores_pool,
            pool_size=pool_size
        ),
        eliminate_duplicates=True
    )
    
    # 4. Optimize
    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', n_generations),
        seed=seed + int(user_idx),
        verbose=False
    )
    
    # 5. Extract Pareto frontier and select a balanced recommendation list
    pareto_solutions = res.X
    pareto_fitness = res.F
    
    if pareto_solutions is None or len(pareto_solutions) == 0:
        return [candidate_pool[idx] for idx in top_ncf_indices], None
        
    # Choose compromise solution closest to the ideal point
    pareto_F = np.atleast_2d(pareto_fitness)
    pareto_X = np.atleast_2d(pareto_solutions)
    
    ideal = np.min(pareto_F, axis=0)
    nadir = np.max(pareto_F, axis=0)
    spread = np.where(nadir - ideal == 0, 1.0, nadir - ideal)
    normalized = (pareto_F - ideal) / spread
    best_idx = int(np.argmin(np.linalg.norm(normalized, axis=1)))
    
    best_chromosome = pareto_X[best_idx].astype(int)
    best_movie_list = [candidate_pool[idx] for idx in best_chromosome]
    
    # Return best list and full Pareto fitness logs for analysis
    pareto_metrics = []
    for f in pareto_F:
        pareto_metrics.append({
            "dopm": float(-f[0]),
            "serendipity": float(-f[1]),
            "fairness": float(-f[2])
        })
        
    return best_movie_list, pareto_metrics



