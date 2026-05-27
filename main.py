import time
import numpy as np
import torch

from weighted_roulette import generate_weighted_roulette_lists
from psnr_filter import filter_by_psnr_sweetspot
from data_loader import MovieLensData
from ncf_inference import get_pure_signal, load_trained_model

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

    print("\n--- Phase 2: Building Per-User Initial Chromosomes ---")
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

    for u_idx in SAMPLE_USERS:
        if u_idx not in sample_results:
            continue

        result = sample_results[u_idx]
        accepted_variants = result["accepted_variants"]

        print(f"\n{'=' * 60}")
        print(f"User Index {u_idx}")
        print(f"{'=' * 60}")
        print(f"S* candidate count: {result['candidate_count']}")
        print(f"P0 chromosome count: {len(P0[u_idx])}")
        print(
            "PSNR sweet spot: "
            f"{result['lower_threshold']:.2f} to {result['upper_threshold']:.2f}"
        )

        print(f"\nOriginal NCF Top {N_LIST_SIZE}:")
        print(get_movie_titles(data, result["original_ncf_list"]))

        for idx, variant in enumerate(accepted_variants[:2], start=1):
            print(f"\nP0 Chromosome {idx} [PSNR {variant['psnr']:.2f}]:")
            print(get_movie_titles(data, variant["list"]))

    return P0


if __name__ == "__main__":
    run_pipeline()
