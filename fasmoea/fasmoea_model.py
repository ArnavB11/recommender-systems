"""
FAS-MOEA objective functions.

Implements the accuracy, fairness, and serendipity objectives described by
Khaitan & Shrivastava (2026), adapted to the existing MovieLens/NCF pipeline
where predicted ratings are normalized sigmoid scores in [0, 1].
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import MovieLensData

ALPHA = 0.5
LAMBDA_THRESHOLD = 0.5
EVENING_MULT = 1.2
MORNING_MULT = 1.0
OTHER_MULT = 0.8


class FASMOEAObjectives:
    """
    Computes FAS-MOEA scores for one user's recommendation list.

    The same object is used inside NSGA-II and during post-hoc evaluation, so all
    dataset-level statistics are precomputed once in fit().
    """

    def __init__(self, data: MovieLensData, predicted_ratings_matrix: np.ndarray):
        self.data = data
        self.R_hat = predicted_ratings_matrix
        self._is_fit = False

        self.long_tail_items = set()
        self.global_genre_dist = {}
        self.user_peak_hours = {}
        self.max_popularity = 0

    def fit(self):
        """Precompute popularity, genre-distribution, and user context data."""
        pop_counts = self.data.item_popularity
        self.max_popularity = max(pop_counts.values(), default=1)
        median_pop = np.median(list(pop_counts.values())) if pop_counts else 0.0
        self.long_tail_items = {
            item_idx for item_idx, cnt in pop_counts.items() if cnt <= median_pop
        }

        genre_totals = {}
        for item_id in self.data.items_df.index:
            genres_str = self.data.items_df.loc[item_id, "genres"]
            if not isinstance(genres_str, str):
                continue
            for genre in genres_str.split("|"):
                if genre and genre != "unknown":
                    genre_totals[genre] = genre_totals.get(genre, 0) + 1
        total_genre_count = sum(genre_totals.values()) or 1
        self.global_genre_dist = {
            genre: count / total_genre_count
            for genre, count in genre_totals.items()
        }

        self._fit_user_peak_hours()
        self._is_fit = True
        return self

    def _fit_user_peak_hours(self):
        """Load MovieLens timestamps and infer each user's modal interaction hour."""
        import pandas as pd
        import data_loader as dl_module

        ratings_path = os.path.join(dl_module.DATA_DIR, "u.data")
        try:
            df = pd.read_csv(
                ratings_path,
                sep="\t",
                names=["user_id", "item_id", "rating", "timestamp"],
            )
            df["hour"] = pd.to_datetime(df["timestamp"], unit="s").dt.hour
            user_modal_hour = df.groupby("user_id")["hour"].agg(
                lambda values: values.mode().iloc[0] if len(values) > 0 else 12
            )
            for user_id, hour in user_modal_hour.items():
                user_idx = self.data.user2idx.get(user_id)
                if user_idx is not None:
                    self.user_peak_hours[user_idx] = int(hour)
        except Exception:
            self.user_peak_hours = {}

        for user_idx in range(self.data.n_users):
            self.user_peak_hours.setdefault(user_idx, 10)

    def _context_multiplier(self, user_idx: int) -> float:
        hour = self.user_peak_hours.get(user_idx, 10)
        if 18 <= hour <= 23:
            return EVENING_MULT
        if 6 <= hour <= 12:
            return MORNING_MULT
        return OTHER_MULT

    def _novelty_score(self, item_idx: int) -> float:
        pop = self.data.item_popularity.get(item_idx, 0)
        if self.max_popularity <= 1:
            return 1.0
        return float(1.0 - (np.log(pop + 1) / np.log(self.max_popularity + 1)))

    def _get_item_genres(self, item_idx: int) -> list:
        item_id = self.data.idx2item.get(item_idx)
        if item_id is None or item_id not in self.data.items_df.index:
            return []
        genres_str = self.data.items_df.loc[item_id, "genres"]
        if not isinstance(genres_str, str):
            return []
        return [
            genre for genre in genres_str.split("|")
            if genre and genre != "unknown"
        ]

    def compute_accuracy(self, user_idx: int, rec_list: list) -> float:
        """Accuracy objective: alpha-weighted long-tail and non-long-tail scores."""
        if not rec_list:
            return 0.0
        n_items = len(rec_list)
        sum_long_tail = 0.0
        sum_relevant = 0.0

        for item_idx in rec_list:
            if self.R_hat is None:
                r_hat = 0.5
            else:
                r_hat = float(self.R_hat[user_idx, item_idx])

            if item_idx in self.long_tail_items:
                sum_long_tail += r_hat
            else:
                sum_relevant += r_hat

        along_tail = sum_long_tail / n_items
        arelevant = sum_relevant / n_items
        return float(ALPHA * along_tail + (1.0 - ALPHA) * arelevant)

    def compute_fairness(self, user_idx: int, rec_list: list) -> float:
        """Fairness objective: recommendation genre distribution alignment."""
        if not rec_list:
            return 0.0

        rec_genre_counts = {}
        for item_idx in rec_list:
            for genre in self._get_item_genres(item_idx):
                rec_genre_counts[genre] = rec_genre_counts.get(genre, 0) + 1
        total_rec_genres = sum(rec_genre_counts.values()) or 1
        p_user = {
            genre: count / total_rec_genres
            for genre, count in rec_genre_counts.items()
        }

        history_genre_counts = {}
        for item_idx in self.data.user_history.get(user_idx, set()):
            for genre in self._get_item_genres(item_idx):
                history_genre_counts[genre] = history_genre_counts.get(genre, 0) + 1
        total_history_genres = sum(history_genre_counts.values()) or 1
        user_weight = max(
            (count / total_history_genres for count in history_genre_counts.values()),
            default=1.0,
        )

        all_genres = set(self.global_genre_dist) | set(p_user)
        deviation = sum(
            abs(p_user.get(genre, 0.0) - self.global_genre_dist.get(genre, 0.0))
            for genre in all_genres
        )
        fairness = 1.0 - (user_weight * deviation / 2.0)
        return float(np.clip(fairness, 0.0, 1.0))

    def compute_serendipity(self, user_idx: int, rec_list: list) -> float:
        """Serendipity objective: unseen, relevant, popularity-novel recommendations."""
        if not rec_list:
            return 0.0

        history = self.data.user_history.get(user_idx, set())
        context_multiplier = self._context_multiplier(user_idx)
        total = 0.0

        for item_idx in rec_list:
            if item_idx in history:
                continue
            r_hat = float(self.R_hat[user_idx, item_idx]) if self.R_hat is not None else 0.0
            if r_hat < LAMBDA_THRESHOLD:
                continue
            total += self._novelty_score(item_idx) * context_multiplier

        max_possible = len(rec_list) * EVENING_MULT
        if max_possible <= 0:
            return 0.0
        return float(np.clip(total / max_possible, 0.0, 1.0))

    def compute_all(self, user_idx: int, rec_list: list) -> dict:
        """Return all FAS-MOEA objectives for one user/list pair."""
        return {
            "accuracy": self.compute_accuracy(user_idx, rec_list),
            "fairness": self.compute_fairness(user_idx, rec_list),
            "serendipity": self.compute_serendipity(user_idx, rec_list),
        }
