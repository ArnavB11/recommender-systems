"""
baselines/base_recommender.py
Abstract base class for all baseline recommender implementations.
"""

import time
import numpy as np
from abc import ABC, abstractmethod
from baselines.cf_utils import (
    get_train_user_history, popularity_fallback_topn, get_candidate_pool
)


class BaseRecommender(ABC):
    """
    Abstract base class for all baseline recommenders.
    
    Every baseline must implement:
        - fit(seed): train/prepare the model
        - recommend(user_idx, n): return top-n item indices
    
    Provides:
        - recommend_extended(): return top-30 for top-N curve evaluation
        - recommend_all_users(): batch recommend with cold-start + timeout
    """

    def __init__(self, data, name, top_n=10, candidate_pool_size=200):
        self.data = data
        self.name = name
        self.top_n = top_n
        self.candidate_pool_size = candidate_pool_size
        self.train_history = get_train_user_history(data)
        self._timeout_count = 0

    @abstractmethod
    def fit(self, seed=42):
        """Train or prepare the recommender model."""
        pass

    @abstractmethod
    def recommend(self, user_idx, n=10):
        """
        Generate a recommendation list for a single user.
        
        Args:
            user_idx: user index
            n: number of items to recommend
        
        Returns:
            list of item indices (length n)
        """
        pass

    def recommend_extended(self, user_idx, n=30):
        """
        Return an extended recommendation list (top-30) for top-N curve evaluation.
        Default: calls recommend() with n=30.
        Override in subclasses that have limitations on list length.
        """
        return self.recommend(user_idx, n=n)

    def recommend_all_users(self, user_indices, n=10, timeout=120, min_interactions=5):
        """
        Generate recommendations for all specified users with:
        - Cold-start fallback for users with < min_interactions
        - Timeout guard per user (120s default)
        
        Args:
            user_indices: list of user indices
            n: number of items to recommend per user
            timeout: max seconds per user
            min_interactions: minimum training interactions to not use fallback
        
        Returns:
            dict: user_idx -> list of recommended item indices
        """
        results = {}
        self._timeout_count = 0

        for u_idx in user_indices:
            history = self.train_history.get(u_idx, set())

            # Cold-start fallback
            if len(history) < min_interactions:
                results[u_idx] = popularity_fallback_topn(self.data, history, n=n)
                continue

            start = time.time()
            try:
                rec = self.recommend(u_idx, n=n)
                elapsed = time.time() - start
                if elapsed > timeout:
                    self._timeout_count += 1
                results[u_idx] = rec
            except Exception as e:
                # Fallback on any error
                results[u_idx] = popularity_fallback_topn(self.data, history, n=n)

        return results

    @property
    def timeout_count(self):
        return self._timeout_count
