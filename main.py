import os
import time
import numpy as np
import torch
from dopm import DOPMRecommender
from weighted_roulette import generate_weighted_roulette_lists
from psnr_filter import filter_by_psnr_sweetspot
from data_loader import MovieLensData
from ncf_inference import load_trained_model

def run_pipeline():
    print("--- Initializing MovieLens Data ---")
    data = MovieLensData()
    
    # 1. Map NCF dataframe to DOPM format
    print("\n--- Mapping Data for DOPM ---")
    # DOPM needs: movie_genres = {item_idx: [genre1, genre2...]}
    movie_genres = {}
    for item_idx, item_id in data.idx2item.items():
        genres_str = data.items_df.loc[item_id, "genres"]
        movie_genres[item_idx] = genres_str.split("|")
        
    # user_history is already provided perfectly by data.user_history
    user_history = data.user_history

    # 2. Initialize and Fit DOPM
    print("\n--- Initializing DOPM Recommender ---")
    dopm_system = DOPMRecommender(epsilon=0.01)
    
    start_time = time.perf_counter()
    dopm_system.fit(user_history, movie_genres)
    end_time = time.perf_counter()
    
    print(f"DOPM Precomputation Finished in {(end_time - start_time)*1000:.2f} ms")
    
    U = data.n_users
    M = data.n_items
    
    # 3. Predict authentic NCF R_hat_matrix
    print("\n--- Running Authentic PyTorch NCF Inference ---")
    model = load_trained_model(data)
    
    # Generate all (User, Item) pairs to create the full matrix
    all_users = torch.arange(U).repeat_interleave(M)
    all_items = torch.arange(M).repeat(U)
    
    start_time = time.perf_counter()
    with torch.no_grad():
        # Compute the authentic neural network predictions!
        all_scores = model(all_users, all_items)
        
    R_hat = all_scores.numpy().reshape(U, M)
    end_time = time.perf_counter()
    print(f"Generated real predictions for {U * M:,} pairs in {(end_time - start_time):.2f} seconds")
    
    # 4. Generate Weighted Roulette Lists
    print("\n--- Generating Weighted Roulette Variants ---")
    
    # We first extract the DOPM scores across all (U, M) pairs
    dopm_scores = dopm_system.calculate_dopm_batch(R_hat)
    
    # Generate variations
    N_list_size = 10
    num_variations = 1000
    pool_size = 50
    
    start_time = time.perf_counter()
    roulette_lists = generate_weighted_roulette_lists(
        dopm_scores=dopm_scores,
        user_watch_matrix=dopm_system.W,
        users=dopm_system.users,
        movies=dopm_system.movies,
        N=N_list_size,
        variations=num_variations,
        pool_size=pool_size
    )
    end_time = time.perf_counter()
    print(f"Generated {num_variations} variants per user in {(end_time - start_time):.4f} seconds")
    
    # 5. PSNR Sweet-Spot Filtering
    print("\n--- Filtering lists dynamically using PSNR ---")
    
    # Inspect a few random users by their actual int indices (0-indexed now)
    sample_users = [0, 99, 499] 
    
    for u_idx in sample_users:
        if u_idx not in roulette_lists:
            continue
            
        print(f"\n{'='*60}\nEvaluating User Index {u_idx}\n{'='*60}")
        
        user_predicted_ratings = R_hat[u_idx, :]
        
        movie_scores_dict = {
            m_idx: user_predicted_ratings[m_idx] 
            for m_idx in dopm_system.movies
        }
        
        # Unseen movies mask
        unseen_movies_mask = (dopm_system.W[u_idx, :] == 0)
        unseen_ratings = np.where(unseen_movies_mask, user_predicted_ratings, -1.0)
        
        top_N_indices = np.argpartition(unseen_ratings, -N_list_size)[-N_list_size:]
        top_N_indices = top_N_indices[np.argsort(-unseen_ratings[top_N_indices])]
        
        original_ncf_list = [dopm_system.movies[idx] for idx in top_N_indices]
        
        print(f"Original purely-accurate NCF list (Top {N_list_size}):")
        # Map item indices to their actual titles
        titles_ncf = [data.items_df.loc[data.idx2item[m], "title"] for m in original_ncf_list]
        print(titles_ncf)
        
        accepted_variants, low, high = filter_by_psnr_sweetspot(
            original_list=original_ncf_list,
            variant_lists=roulette_lists[u_idx],
            movie_scores_dict=movie_scores_dict,
            lower_percentile=25,
            upper_percentile=75
        )
        
        print(f"\nDynamically Calculated Sweet Spot:")
        print(f"  Lower Bound (25th Percentile): {low:.2f}")
        print(f"  Upper Bound (75th Percentile): {high:.2f}")
        
        print(f"\n{len(accepted_variants)} lists passed the sweet-spot test!")
        # Print first 2 accepted variants to keep terminal clean
        for idx, item in enumerate(accepted_variants[:2]):
            print(f"\nAccepted Variant {idx+1} [PSNR {item['psnr']:.2f}]:")
            variant_titles = [data.items_df.loc[data.idx2item[m], "title"] for m in item['list']]
            print(variant_titles)
        print(f"... and {len(accepted_variants) - 2} more variants.")

if __name__ == "__main__":
    run_pipeline()
