import numpy as np

class DOPMRecommender:
    """
    Extremely optimized, fully vectorized implementation of the Multi-Stakeholder 
    and Multi-Objective Recommendation System Objective Function.
    
    Optimizations included:
    - 32-bit floats (np.float32) to double CPU cache hits and utilize AVX instructions.
    - Advanced indexing for O(1) bulk matrix assignments instead of python loops over arrays.
    - Full linear algebra formulation for Novelty and IGF scores.
    """
    
    def __init__(self, epsilon=0.01):
        """
        Initialize the DOPM Recommender.
        
        Args:
            epsilon (float): Baseline value to prevent the novelty score from dropping 
                             to zero in extreme cases.
        """
        self.epsilon = np.float32(epsilon)
        
    def fit(self, user_history, movie_genres):
        """
        Precomputes the Global Filtering (IGF) values and all user genre statistics.
        Runs in extremely fast C-level loops via NumPy.
        """
        self.users = list(user_history.keys())
        self.movies = list(movie_genres.keys())
        
        all_genres = set()
        for genres in movie_genres.values():
            all_genres.update(genres)
        self.genres = list(all_genres)
        
        self.user2idx = {u: i for i, u in enumerate(self.users)}
        self.movie2idx = {m: i for i, m in enumerate(self.movies)}
        self.genre2idx = {g: i for i, g in enumerate(self.genres)}
        
        U = len(self.users)
        M = len(self.movies)
        G = len(self.genres)
        
        # =====================================================================
        # 1. Build matrices using ultra-fast bulk array indexing & float32
        # =====================================================================
        mg_rows, mg_cols = [], []
        for m, genres in movie_genres.items():
            idx_m = self.movie2idx[m]
            for g in genres:
                mg_rows.append(idx_m)
                mg_cols.append(self.genre2idx[g])
        
        self.MG = np.zeros((M, G), dtype=np.float32)
        if mg_rows:
            self.MG[mg_rows, mg_cols] = 1.0
                
        w_rows, w_cols = [], []
        for u, history in user_history.items():
            idx_u = self.user2idx[u]
            for m in history:
                if m in self.movie2idx:
                    w_rows.append(idx_u)
                    w_cols.append(self.movie2idx[m])
                    
        self.W = np.zeros((U, M), dtype=np.float32)
        if w_rows:
            self.W[w_rows, w_cols] = 1.0
                    
        # =====================================================================
        # 2. Global Platform Filtering (IGF)
        # =====================================================================
        T_g = self.MG.sum(axis=0) 
        safe_T_g = np.where(T_g == 0, 1.0, T_g)
        self.igf = (1.0 + np.log(M / safe_T_g)).astype(np.float32)
        self.igf = np.where(T_g == 0, 1.0, self.igf)
        
        # =====================================================================
        # 3. Vectorized User Statistics
        # =====================================================================
        self.Nu_g = self.W.dot(self.MG) 
        self.Nu_total = self.W.sum(axis=1, keepdims=True)
        
        safe_Nu_total = np.where(self.Nu_total == 0, 1.0, self.Nu_total)
        self.fatigue = np.where(self.Nu_total == 0, 0.0, self.Nu_g / safe_Nu_total)
        
        self.genre_novelty = np.maximum(self.epsilon, 1.0 - self.fatigue)
        
        # =====================================================================
        # 4. Precompute final multiplier matrix
        # =====================================================================
        self.score_matrix = self.genre_novelty * self.igf 
        self.sum_matrix = self.score_matrix.dot(self.MG.T) 
        
        self.G_counts = self.MG.sum(axis=1, keepdims=True).T 
        safe_G_counts = np.where(self.G_counts == 0, 1.0, self.G_counts)
        
        self.dopm_multipliers = self.sum_matrix / safe_G_counts
        self.dopm_multipliers = np.where(self.G_counts == 0, 1.0, self.dopm_multipliers)

    def calculate_novelty(self, u, i):
        if u not in self.user2idx or i not in self.movie2idx:
            return 0.0
            
        u_idx = self.user2idx[u]
        i_idx = self.movie2idx[i]
        
        novelty_sum = self.genre_novelty[u_idx, :].dot(self.MG[i_idx, :])
        g_count = self.MG[i_idx, :].sum()
        
        if g_count == 0:
            return 0.0
        return float(novelty_sum / g_count)

    def calculate_dopm(self, u, i, predicted_rating):
        if predicted_rating is None:
            return None
            
        if u not in self.user2idx or i not in self.movie2idx:
            return predicted_rating
            
        u_idx = self.user2idx[u]
        i_idx = self.movie2idx[i]
        
        return float(predicted_rating * self.dopm_multipliers[u_idx, i_idx])
        
    def calculate_dopm_batch(self, R_hat_matrix):
        """
        Calculates the DOPM score for all users and all movies at once.
        R_hat_matrix must be shape (U, M)
        """
        return R_hat_matrix * self.dopm_multipliers

# Example Usage
if __name__ == "__main__":
    import time
    
    # Dummy data setup
    movie_genres = {
        'm1': ['Action', 'Sci-Fi'],
        'm2': ['Romance', 'Comedy'],
        'm3': ['Documentary'], 
        'm4': ['Action', 'Comedy']
    }
    
    user_history = {
        'u1': ['m1', 'm4'],
        'u2': ['m2']
    }
    
    cf_predictions = {
        ('u1', 'm2'): 4.5,
        ('u1', 'm3'): 4.0, 
    }
    
    # Initialize and fit
    dopm_system = DOPMRecommender(epsilon=0.01)
    
    # Time the fast matrix precomputation
    start_time = time.perf_counter()
    dopm_system.fit(user_history, movie_genres)
    end_time = time.perf_counter()
    
    print(f"--- Vectorized Precomputation Finished in {(end_time - start_time)*1000:.4f} ms ---")
    
    # Dummy R_hat matrix (U x M) representing NCF output
    # Let's say all ratings are randomly between 3.0 and 5.0
    R_hat = np.random.uniform(3.0, 5.0, size=(len(dopm_system.users), len(dopm_system.movies)))
    
    # Let's explicitly set the ones from our cf_predictions for testing
    u1_idx, u2_idx = dopm_system.user2idx['u1'], dopm_system.user2idx['u2']
    m2_idx, m3_idx = dopm_system.movie2idx['m2'], dopm_system.movie2idx['m3']
    R_hat[u1_idx, m2_idx] = 4.5
    R_hat[u1_idx, m3_idx] = 4.0
    
