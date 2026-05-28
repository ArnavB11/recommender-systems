import time
import numpy as np
import torch

from weighted_roulette import generate_weighted_roulette_lists
from psnr_filter import filter_by_psnr_sweetspot
from data_loader import MovieLensData
from ncf_inference import get_pure_signal, load_trained_model

# Import our new multi-objective serendipity optimization components
from dataset_linker import DatasetLinker
from serendipity import SerendipityModel
from genetic_algorithm import run_nsga2_optimization
from dopm import DOPMRecommender

N_LIST_SIZE = 10
NUM_VARIATIONS = 1000
ROULETTE_POOL_SIZE = 50
SAMPLE_USERS = [0, 99, 499]


def generate_full_r_hat(model, n_users, n_items):
    all_users = torch.arange(n_users).repeat_interleave(n_items)
    all_items = torch.arange(n_items).repeat(n_users)

    with torch.no_grad():
        all_scores = model(all_users, all_items)

    return all_scores.detach().cpu().numpy().reshape(n_users, n_items)


def get_movie_titles(data, item_indices):
    return [
        data.items_df.loc[data.idx2item[item_idx], "title"]
        for item_idx in item_indices
    ]


def run_pipeline():
    print("--- Phase 1: Loading Data and Generating NCF Scores ---")
    data = MovieLensData()

    model = load_trained_model(data)

    start_time = time.perf_counter()
    R_hat = generate_full_r_hat(model, data.n_users, data.n_items)
    end_time = time.perf_counter()

    print(f"Generated R_hat with shape {R_hat.shape} in {(end_time - start_time):.2f} seconds")

    print("\n--- Phase 2: Initializing DOPM & BERT Serendipity Models ---")
    # Initialize the Stakeholder DOPM Recommender and Fit
    dopm_system = DOPMRecommender()
    movie_genres = {
        item_idx: data.items_df.loc[data.idx2item[item_idx], "genres"].split("|")
        for item_idx in range(data.n_items)
    }
    dopm_system.fit(data.user_history, movie_genres)
    
    # Initialize Dataset Linker and BERT Serendipity Model
    linker = DatasetLinker()
    serendipity_system = SerendipityModel(data=data, linker=linker)

    print("\n--- Phase 3: Building Per-User Initial Chromosomes (P0) ---")
    P0 = {}
    sample_results = {}

    start_time = time.perf_counter()
    for u_idx in range(data.n_users):
        pure_signal = get_pure_signal(
            R_hat[u_idx],
            data.user_history.get(u_idx, set()),
        )

        candidate_items = pure_signal["item_indices"]
        candidate_scores = np.asarray(pure_signal["scores"], dtype=np.float64)

        if len(candidate_items) == 0:
            P0[u_idx] = []
            continue

        movie_scores_dict = {
            item_idx: float(score)
            for item_idx, score in zip(candidate_items, candidate_scores)
        }

        roulette_lists = generate_weighted_roulette_lists(
            pure_signal_scores=candidate_scores.reshape(1, -1),
            user_watch_matrix=np.zeros((1, len(candidate_items)), dtype=np.int8),
            users=[u_idx],
            movies=candidate_items,
            N=N_LIST_SIZE,
            variations=NUM_VARIATIONS,
            pool_size=ROULETTE_POOL_SIZE,
        )

        original_ncf_list = candidate_items[:N_LIST_SIZE]
        accepted_variants, low, high = filter_by_psnr_sweetspot(
            original_list=original_ncf_list,
            variant_lists=roulette_lists[u_idx],
            movie_scores_dict=movie_scores_dict,
            lower_percentile=25,
            upper_percentile=75,
            max_val=1.0,
        )

        P0[u_idx] = [variant["list"] for variant in accepted_variants]

        if u_idx in SAMPLE_USERS:
            sample_results[u_idx] = {
                "candidate_count": len(candidate_items),
                "original_ncf_list": original_ncf_list,
                "accepted_variants": accepted_variants,
                "lower_threshold": low,
                "upper_threshold": high,
            }

    end_time = time.perf_counter()
    print(f"Built P0 for {len(P0):,} users in {(end_time - start_time):.2f} seconds")

    print("\n--- Phase 4: Running Pymoo Multi-Objective NSGA-II Optimization ---")
    
    for u_idx in SAMPLE_USERS:
        if u_idx not in sample_results:
            continue
            
        print(f"\nRunning NSGA-II optimization for User {u_idx}...")
        
        result = sample_results[u_idx]
        pure_signal = get_pure_signal(R_hat[u_idx], data.user_history.get(u_idx, set()))
        candidate_pool = pure_signal["item_indices"]
        
        initial_variants = P0[u_idx]
        if not initial_variants:
            print(f"Skipping User {u_idx} due to empty P0 chromosome population.")
            continue
            
        ga_start = time.perf_counter()
        best_movie_list, pareto_metrics = run_nsga2_optimization(
            user_idx=u_idx,
            candidate_pool=candidate_pool,
            ncf_scores=R_hat[u_idx],
            dopm_recommender=dopm_system,
            serendipity_model=serendipity_system,
            user_history=list(data.user_history.get(u_idx, set())),
            initial_population_variants=initial_variants,
            N=N_LIST_SIZE,
            pop_size=50,
            n_generations=40
        )
        ga_elapsed = time.perf_counter() - ga_start
        print(f"User {u_idx} NSGA-II optimization completed in {ga_elapsed:.2f} seconds.")
        
        # Calculate metric scores for original vs optimized lists
        user_pref_vec = serendipity_system.compute_user_preference_vector(data.user_history.get(u_idx, set()))
        user_hist_list = list(data.user_history.get(u_idx, set()))
        
        def evaluate_list(movie_list):
            acc = np.mean([R_hat[u_idx][m] for m in movie_list])
            nov = np.mean([dopm_system.calculate_novelty(u_idx, m) for m in movie_list])
            ser = np.mean([serendipity_system.calculate_serendipity(user_hist_list, user_pref_vec, m) for m in movie_list])
            return acc, nov, ser
            
        orig_acc, orig_nov, orig_ser = evaluate_list(result["original_ncf_list"])
        opt_acc, opt_nov, opt_ser = evaluate_list(best_movie_list)
        
        # Calculate Multi-Objective Harmonic Score (MOHS)
        # MOHS is a 3-variable harmonic mean, punishing poor performance in any single metric.
        def calculate_mohs(acc, nov, ser):
            acc = max(acc, 1e-6)
            nov = max(nov, 1e-6)
            ser = max(ser, 1e-6)
            return 3.0 / (1.0 / acc + 1.0 / nov + 1.0 / ser)
            
        orig_mohs = calculate_mohs(orig_acc, orig_nov, orig_ser)
        opt_mohs = calculate_mohs(opt_acc, opt_nov, opt_ser)
        mohs_improvement = ((opt_mohs - orig_mohs) / orig_mohs) * 100.0
        
        print(f"\n{'=' * 75}")
        print(f"Comparison Summary for User {u_idx}")
        print(f"{'=' * 75}")
        print(f"{'Objective Metric':<20} | {'Original NCF List':<22} | {'NSGA-II Optimized List':<22}")
        print(f"{'-' * 75}")
        print(f"{'NCF Rating Accuracy':<20} | {orig_acc:<22.4f} | {opt_acc:<22.4f}")
        print(f"{'DOPM Genre Diversity':<20} | {orig_nov:<22.4f} | {opt_nov:<22.4f}")
        print(f"{'BERT Serendipity':<20} | {orig_ser:<22.4f} | {opt_ser:<22.4f}")
        print(f"{'-' * 75}")
        print(f"{'MOHS Quality Score':<20} | {orig_mohs:<22.4f} | {opt_mohs:<22.4f}")
        print(f"{'MOHS Net Improvement':<20} | {'-':<22} | {f'+{mohs_improvement:.2f}%':<22}")
        print(f"{'-' * 75}")
        
        print("\nOriginal Top NCF Recommendation Titles:")
        print(get_movie_titles(data, result["original_ncf_list"]))
        
        print("\nNSGA-II Multi-Objective Optimized Recommendation Titles:")
        print(get_movie_titles(data, best_movie_list))
        print(f"{'=' * 75}\n")
        
    return P0


if __name__ == "__main__":
    run_pipeline()
