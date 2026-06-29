import time
import numpy as np
import torch
from data_loader import MovieLensData
from ncf_inference import get_pure_signal, load_trained_model
from serendipity import SerendipityModel
from dataset_linker import DatasetLinker
from dopm import DOPMRecommender
from fairness import FairnessObjective
from genetic_algorithm import run_nsga2_optimization

print("Loading data...")
data = MovieLensData()
model = load_trained_model(data)

user_idx = 0
all_items = torch.arange(data.n_items)
user_tensor = torch.full((data.n_items,), user_idx, dtype=torch.long)
with torch.no_grad():
    R_hat_row = model(user_tensor, all_items).numpy()

pure_signal = get_pure_signal(R_hat_row, data.user_history.get(user_idx, set()))
candidate_pool = pure_signal["item_indices"]

print("Initializing models...")
dopm_system = DOPMRecommender()
movie_genres = {
    item_idx: data.items_df.loc[data.idx2item[item_idx], "genres"].split("|")
    for item_idx in range(data.n_items)
}
dopm_system.fit(data.user_history, movie_genres)

linker = DatasetLinker()
serendipity_system = SerendipityModel(data=data, linker=linker)

fairness_model = FairnessObjective(data)
fairness_model.fit()

# Build a small initial population from the top candidates
initial_variants = [candidate_pool[:10] for _ in range(5)]

print("Running NSGA-II for 10 generations...")
start = time.perf_counter()
best_list, pareto_metrics = run_nsga2_optimization(
    user_idx=user_idx,
    candidate_pool=candidate_pool,
    ncf_scores=R_hat_row,
    dopm_recommender=dopm_system,
    serendipity_model=serendipity_system,
    fairness_model=fairness_model,
    user_history=list(data.user_history.get(user_idx, set())),
    initial_population_variants=initial_variants,
    N=10,
    pop_size=5,
    n_generations=3
)
print(f"Completed in {time.perf_counter() - start:.2f} seconds.")
print("Best recommendation list:", best_list)
