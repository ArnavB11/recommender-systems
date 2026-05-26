import numpy as np

def generate_weighted_roulette_lists(dopm_scores, user_watch_matrix, users, movies, N=10, variations=5, pool_size=50):
    """
    Generates multiple variant recommendation lists per user using a 
    Weighted Roulette selection over the DOPM scores.
    
    Args:
        dopm_scores (np.ndarray): Shape (U, M) containing calculated DOPM scores.
        user_watch_matrix (np.ndarray): Shape (U, M) indicating if user watched a movie 
                                        (1 if watched, 0 otherwise).
        users (list): List of user IDs corresponding to matrix rows.
        movies (list): List of movie IDs corresponding to matrix columns.
        N (int): The length of each recommendation list (e.g., top 10).
        variations (int): How many distinct lists to generate per user.
        pool_size (int): Restricts the roulette selection to the top `pool_size`
                         highest scoring movies. This ensures lists stay highly accurate 
                         while introducing randomized diversity. If None, samples all.
                         
    Returns:
        dict: Mapping from user_id to a list of `variations` lists, each of length `N`.
    """
    # 1. Mask out movies the user has already watched
    unseen_mask = (user_watch_matrix == 0)
    
    # Ensure scores are non-negative for probability calculation
    valid_scores = np.where(unseen_mask & (dopm_scores > 0), dopm_scores, 0.0)
    
    user_lists = {}
    for u_idx, u in enumerate(users):
        user_scores = valid_scores[u_idx]
        
        if pool_size is not None:
            # Find indices of top pool_size movies
            actual_pool = min(pool_size, np.count_nonzero(user_scores))
            if actual_pool > 0:
                top_indices = np.argpartition(user_scores, -actual_pool)[-actual_pool:]
                # Only keep scores for these top indices
                pool_scores = np.zeros_like(user_scores)
                pool_scores[top_indices] = user_scores[top_indices]
                user_scores = pool_scores
            
        total_score = user_scores.sum()
        if total_score == 0:
            user_lists[u] = [[] for _ in range(variations)]
            continue
            
        probabilities = user_scores / total_score
        # np.random.choice strictly requires float64 that sums to 1.0 exactly
        probabilities = probabilities.astype(np.float64) 
        probabilities /= probabilities.sum() 
        
        u_variations = []
        for _ in range(variations):
            num_available = np.count_nonzero(probabilities)
            sample_size = min(N, num_available)
            
            if sample_size == 0:
                u_variations.append([])
                continue
                
            sampled_idx = np.random.choice(
                len(movies), 
                size=sample_size, 
                replace=False, 
                p=probabilities
            )
            
            # Convert matrix indices back to actual movie string/int IDs
            sampled_movies = [movies[idx] for idx in sampled_idx]
            u_variations.append(sampled_movies)
            
        user_lists[u] = u_variations
        
    return user_lists
