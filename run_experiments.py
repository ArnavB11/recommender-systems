import os
import json
import time
import tracemalloc
import numpy as np
import torch
from collections import defaultdict
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD

from data_loader import MovieLensData
from ncf_inference import get_pure_signal, load_trained_model
from dataset_linker import DatasetLinker
from serendipity import SerendipityModel
from genetic_algorithm import run_nsga2_optimization
from dopm import DOPMRecommender
from fairness import FairnessObjective
from weighted_roulette import generate_weighted_roulette_lists
from psnr_filter import filter_by_psnr_sweetspot
from evaluator import evaluate_list

def run_experiments():
    print("Initializing Data and Models...")
    data = MovieLensData()
    model = load_trained_model(data)

    print("Generating R_hat...")
    all_users = torch.arange(data.n_users).repeat_interleave(data.n_items)
    all_items = torch.arange(data.n_items).repeat(data.n_users)
    with torch.no_grad():
        all_scores = model(all_users, all_items).detach().cpu().numpy().reshape(data.n_users, data.n_items)
    R_hat = all_scores

    dopm_system = DOPMRecommender()
    movie_genres = {item_idx: data.items_df.loc[data.idx2item[item_idx], "genres"].split("|") for item_idx in range(data.n_items)}
    dopm_system.fit(data.user_history, movie_genres)
    
    linker = DatasetLinker()
    serendipity_system = SerendipityModel(data=data, linker=linker)
    fairness_model = FairnessObjective(data)
    fairness_model.fit()

    # Extract relevant items from validation set (ground truth for accuracy)
    val_df = data.val_df
    val_relevant = val_df[val_df["label"] == 1].groupby("user_idx")["item_idx"].apply(list).to_dict()

    # Select a few test users (e.g., 10 users) to keep runtime reasonable for demo
    test_users = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    
    # Tracking variables
    results = {
        "pareto_fronts": [],
        "hypervolume": [],
        "igd": [],
        "top_n_metrics": {
            "ncf": defaultdict(lambda: defaultdict(list)),
            "nsga2": defaultdict(lambda: defaultdict(list))
        },
        "runtime": [],
        "memory": []
    }

    n_generations = 50
    pop_size = 50
    # The reference point for HV (since we maximize by minimizing negative, pymoo minimizes by default)
    # We negated the objectives in Problem, so the values are negative. 
    # The reference point should be larger than any possible objective value.
    # Max DOPM/Serendipity/Fairness is ~1.0, so negative is -1.0. Worst is 0.0, so ref point is [0, 0, 0].
    hv_indicator = Hypervolume(ref_point=np.array([0.0, 0.0, 0.0]))

    for u_idx in test_users:
        if u_idx not in val_relevant or len(val_relevant[u_idx]) == 0:
            continue
            
        print(f"\n--- Processing User {u_idx} ---")
        tracemalloc.start()
        start_time = time.perf_counter()
        
        pure_signal = get_pure_signal(R_hat[u_idx], data.user_history.get(u_idx, set()))
        candidate_items = pure_signal["item_indices"]
        candidate_scores = np.asarray(pure_signal["scores"], dtype=np.float64)
        
        movie_scores_dict = {item_idx: float(score) for item_idx, score in zip(candidate_items, candidate_scores)}
        roulette_lists = generate_weighted_roulette_lists(
            candidate_scores.reshape(1, -1), np.zeros((1, len(candidate_items)), dtype=np.int8),
            [u_idx], candidate_items, N=30, variations=50, pool_size=50
        )
        
        accepted_variants, _, _ = filter_by_psnr_sweetspot(
            candidate_items[:30], roulette_lists[u_idx], movie_scores_dict, 10, 90, 1.0
        )
        initial_variants = [v["list"] for v in accepted_variants]
        if not initial_variants:
            initial_variants = [candidate_items[:30]]

        best_list, pareto, history = run_nsga2_optimization(
            u_idx, candidate_items, R_hat[u_idx], dopm_system, serendipity_system, fairness_model, 
            list(data.user_history.get(u_idx, set())), initial_variants, N=30, pop_size=pop_size, n_generations=n_generations
        )
        
        elapsed_time = time.perf_counter() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        results["runtime"].append(elapsed_time)
        results["memory"].append(peak_mem / (1024 * 1024)) # MB
        
        # 1. Pareto Front Data
        # Format: (Accuracy (proxied by mean NCF score), Fairness, Serendipity)
        user_pareto = []
        if history:
            last_gen = history[-1]
            for i, sol in enumerate(last_gen.opt):
                # Calculate mean NCF score for this solution to proxy Accuracy
                sol_items = [candidate_items[idx] for idx in sol.X.astype(int)]
                acc = np.mean([R_hat[u_idx][m] for m in sol_items])
                # f values are negative in our formulation
                user_pareto.append({
                    "accuracy": float(acc),
                    "fairness": float(-sol.F[2]),
                    "serendipity": float(-sol.F[1])
                })
        results["pareto_fronts"].append(user_pareto)

        # 2. Hypervolume Evolution
        user_hv = []
        if history:
            for gen_algo in history:
                # Get the pareto front of this generation
                F = gen_algo.opt.get("F")
                if len(F) > 0:
                    hv_val = hv_indicator.do(F)
                    user_hv.append(float(hv_val))
                else:
                    user_hv.append(0.0)
        results["hypervolume"].append(user_hv)
        
        # 2.5 IGD Evolution (Inverted Generational Distance)
        user_igd = []
        if history:
            # Use the Pareto front of the final generation as the reference front
            reference_front = history[-1].opt.get("F")
            if len(reference_front) > 0:
                igd_indicator = IGD(reference_front)
                for gen_algo in history:
                    F = gen_algo.opt.get("F")
                    if len(F) > 0:
                        user_igd.append(float(igd_indicator.do(F)))
                    else:
                        # Fallback for empty front (shouldn't happen)
                        user_igd.append(1.0)
        results["igd"].append(user_igd)
        
        # 3. Top-N Metrics (K=5..30)
        user_pref_vec = serendipity_system.compute_user_preference_vector(data.user_history.get(u_idx, set()))
        user_hist_list = list(data.user_history.get(u_idx, set()))
        rel_items = val_relevant[u_idx]
        
        ncf_base_list = candidate_items[:30]
        
        for k in range(5, 31, 5):
            ncf_metrics = evaluate_list(ncf_base_list, rel_items, dopm_system, serendipity_system, fairness_model, u_idx, user_hist_list, user_pref_vec, R_hat[u_idx], k)
            nsga2_metrics = evaluate_list(best_list, rel_items, dopm_system, serendipity_system, fairness_model, u_idx, user_hist_list, user_pref_vec, R_hat[u_idx], k)
            
            for m_name, val in ncf_metrics.items():
                results["top_n_metrics"]["ncf"][k][m_name].append(val)
                
            for m_name, val in nsga2_metrics.items():
                results["top_n_metrics"]["nsga2"][k][m_name].append(val)

    # Average top-N metrics across users
    final_top_n = {"ncf": {}, "nsga2": {}}
    for model_name in ["ncf", "nsga2"]:
        for k in results["top_n_metrics"][model_name]:
            final_top_n[model_name][k] = {m: float(np.mean(vals)) for m, vals in results["top_n_metrics"][model_name][k].items()}
            
    results["top_n_metrics"] = final_top_n
    
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_data.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Experiments completed. Data saved to results/experiment_data.json")

if __name__ == "__main__":
    run_experiments()
