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
        # Pointer to Paper: Implements Equation 1 (IGF = 1 + log(|T| / |T_g|))
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
        # Pointer to Paper: Implements the multiplier part of Equation 3 
        # (Combining the novelty fatigue terms with the IGF terms)
        self.score_matrix = self.genre_novelty * self.igf 
        self.sum_matrix = self.score_matrix.dot(self.MG.T) 
        
        self.G_counts = self.MG.sum(axis=1, keepdims=True).T 
        safe_G_counts = np.where(self.G_counts == 0, 1.0, self.G_counts)
        
        self.dopm_multipliers = self.sum_matrix / safe_G_counts
        self.dopm_multipliers = np.where(self.G_counts == 0, 1.0, self.dopm_multipliers)

    def calculate_novelty(self, u, i):
        """
        Pointer to Paper: Implements Equation 2 (Personalized Novelty score)
        Computes the standalone novelty score without the IGF factor.
        """
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
        """
        Pointer to Paper: Implements Equation 3 (Final DOPM score)
        Multiplies the predicted rating (r_hat) with the combined Novelty and IGF scores.
        """
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

