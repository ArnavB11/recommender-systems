import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.population import Population

class DuplicateRepair(Repair):
    """
    A custom repair operator for pymoo to ensure that all chromosomes 
    (which represent recommendation lists of item indices) contain 
    completely unique movies with zero duplicates.
    """
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size

    def _do(self, problem, X, **kwargs):
        # X is the matrix of chromosomes with shape (pop_size, n_var)
        for i in range(len(X)):
            chromosome = X[i].astype(int)
            unique_vals, indices = np.unique(chromosome, return_index=True)
            
            # If duplicates exist, replace them with unique elements from the pool
            if len(unique_vals) < len(chromosome):
                available_pool = set(range(self.pool_size)) - set(unique_vals)
                # Sort available pool for reproducibility
                available_list = sorted(list(available_pool))
                
                # Identify which positions in the chromosome are duplicates
                duplicate_positions = set(range(len(chromosome))) - set(indices)
                for pos in duplicate_positions:
                    if available_list:
                        new_val = np.random.choice(available_list)
                        available_list.remove(new_val)
                        chromosome[pos] = new_val
                
                X[i] = chromosome
        return X

class RecommendationListProblem(Problem):
    """
    Custom pymoo Problem for optimizing a recommendation list of size N.
    
    Decision Variables:
    - An integer vector X of size N, where each element is an index in the candidate movie pool [0, K-1].
    
    Conflicting Objectives to Minimize (using negative values for maximization):
    1. f1 = -Accuracy (Mean NCF sigmoid prediction score)
    2. f2 = -DOPM Novelty (Mean genre-based novelty score)
    3. f3 = -Serendipity (Mean review-based unexpectedness/relevance score)
    """
    def __init__(self, user_idx, candidate_pool, ncf_scores, dopm_recommender, serendipity_model, user_history, N=10):
        self.user_idx = user_idx
        self.candidate_pool = candidate_pool  # List of actual item_idx
        self.ncf_scores = ncf_scores          # NCF scores for this user across all items
        self.dopm = dopm_recommender
        self.serendipity = serendipity_model
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
            
            # 1. Accuracy: average NCF score
            acc_scores = [self.ncf_scores[m] for m in actual_movies]
            f1[p] = -np.mean(acc_scores)
            
            # 2. Diversity: average DOPM novelty
            nov_scores = [self.dopm.calculate_novelty(self.user_idx, m) for m in actual_movies]
            f2[p] = -np.mean(nov_scores)
            
            # 3. Serendipity: average review-based serendipity
            ser_scores = [self.serendipity.calculate_serendipity(self.user_history, self.user_pref_vector, m) for m in actual_movies]
            f3[p] = -np.mean(ser_scores)
            
        # Pymoo minimizes all objectives
        out["F"] = np.column_stack([f1, f2, f3])

def run_nsga2_optimization(user_idx, candidate_pool, ncf_scores, dopm_recommender, 
                           serendipity_model, user_history, initial_population_variants, 
                           N=10, pop_size=50, n_generations=40):
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
        user_history=user_history,
        N=N
    )
    
    # Custom repair ensuring duplicate-free chromosomes
    repair = DuplicateRepair(pool_size)
    
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
    
    # Repair initial population to enforce unique constraints
    initial_chromosomes = repair._do(problem, initial_chromosomes)
    
    # Convert to pymoo Population object
    pop = Population.new("X", initial_chromosomes)
    
    # 2. Setup NSGA2 Algorithm
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=pop,  # Seed with our high-quality PSNR population
        crossover=SBX(prob=0.9, eta=15, repair=repair),
        mutation=PM(prob=1.0/N, eta=20, repair=repair),
        eliminate_duplicates=True
    )
    
    # 3. Optimize
    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', n_generations),
        seed=42,
        verbose=False
    )
    
    # 4. Extract Pareto frontier and select a balanced recommendation list
    pareto_solutions = res.X
    pareto_fitness = res.F  # Shape: (num_solutions, 3) -> [-Acc, -Nov, -Ser]
    
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
            "accuracy": float(-f[0]),
            "diversity": float(-f[1]),
            "serendipity": float(-f[2])
        })
        
    return best_movie_list, pareto_metrics
