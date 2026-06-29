"""
baselines/ae_mc_nsga2.py
Baseline 6: AE-MC + NSGA-II (Rajput et al., 2025)
"NSGA-II optimized deep autoencoders for enhanced multi-criteria recommendation system."

Train a simple autoencoder on the user-item rating matrix.
Two objectives: Accuracy (AE predicted rating) and Coverage (genre coverage).
"""

import numpy as np
import torch
import torch.nn as nn
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize

from baselines.base_recommender import BaseRecommender
from baselines.cf_utils import build_rating_matrix, get_genre_matrix, get_candidate_pool
from data_loader import GENRE_COLS


class RatingAutoencoder(nn.Module):
    """
    Simple autoencoder for rating reconstruction.
    Encoder: n_items -> 256 -> ReLU -> 64 -> ReLU
    Decoder: 64 -> 256 -> ReLU -> n_items -> Sigmoid
    """

    def __init__(self, n_items):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_items, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, n_items),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AEMCNSGA2Problem(Problem):
    """
    AE-MC: 2-objective problem.
    f1 = -accuracy (mean AE predicted rating)
    f2 = -coverage (unique genres / total genres)
    """

    def __init__(self, candidate_pool, R_hat_row, genre_matrix, total_genres, n=10):
        self.candidate_pool = candidate_pool
        self.R_hat_row = R_hat_row
        self.genre_matrix = genre_matrix
        self.total_genres = total_genres
        self.n = n
        self.pool_size = len(candidate_pool)

        super().__init__(
            n_var=n, n_obj=2,
            xl=0, xu=self.pool_size - 1, vtype=int,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        pop_size = len(X)
        F = np.zeros((pop_size, 2))

        for p in range(pop_size):
            indices = X[p].astype(int) % self.pool_size
            items = [self.candidate_pool[idx] for idx in indices]

            # f1: Accuracy
            scores = [self.R_hat_row[i] for i in items]
            F[p, 0] = -np.mean(scores)

            # f2: Coverage
            all_genres = set()
            for i in items:
                genres_present = np.where(self.genre_matrix[i] > 0)[0]
                all_genres.update(genres_present)
            coverage = len(all_genres) / self.total_genres if self.total_genres > 0 else 0
            F[p, 1] = -coverage

        out["F"] = F


class AEMCNSGA2(BaseRecommender):
    """AE-MC + NSGA-II baseline: Autoencoder + NSGA-II (Accuracy + Coverage)."""

    def __init__(self, data, pop_size=80, n_gen=150, ae_epochs=30,
                 top_n=10, candidate_pool_size=200):
        super().__init__(data, name="AE-MC+NSGA-II", top_n=top_n, candidate_pool_size=candidate_pool_size)
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.ae_epochs = ae_epochs
        self.R_hat = None
        self.genre_matrix = None
        self.total_genres = len(GENRE_COLS)

    def fit(self, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"  [{self.name}] Training autoencoder ({self.ae_epochs} epochs)...")

        rating_matrix, _ = build_rating_matrix(self.data)

        # Normalize to [0, 1] for sigmoid output
        max_rating = rating_matrix.max()
        if max_rating > 0:
            norm_matrix = rating_matrix / max_rating
        else:
            norm_matrix = rating_matrix

        mask = (rating_matrix > 0).astype(np.float32)

        # Train autoencoder
        model = RatingAutoencoder(self.data.n_items)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss(reduction='none')

        input_tensor = torch.FloatTensor(norm_matrix)
        mask_tensor = torch.FloatTensor(mask)

        model.train()
        for epoch in range(self.ae_epochs):
            optimizer.zero_grad()
            output = model(input_tensor)
            loss = (criterion(output, input_tensor) * mask_tensor).sum() / mask_tensor.sum()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"    AE Epoch {epoch + 1}/{self.ae_epochs}, Loss: {loss.item():.4f}")

        # Generate predictions
        model.eval()
        with torch.no_grad():
            self.R_hat = model(input_tensor).numpy() * max_rating

        self.genre_matrix = get_genre_matrix(self.data)
        print(f"  [{self.name}] Fit complete. R_hat shape: {self.R_hat.shape}")

    def recommend(self, user_idx, n=10):
        candidates = get_candidate_pool(
            user_idx, self.R_hat[user_idx], self.train_history,
            pool_size=self.candidate_pool_size
        )
        if len(candidates) < n:
            return candidates

        problem = AEMCNSGA2Problem(
            candidates, self.R_hat[user_idx], self.genre_matrix,
            self.total_genres, n=n
        )

        algorithm = NSGA2(
            pop_size=self.pop_size,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
            mutation=PM(eta=20, vtype=float, repair=RoundingRepair()),
            eliminate_duplicates=True,
        )

        res = minimize(problem, algorithm, termination=("n_gen", self.n_gen), seed=None, verbose=False)

        if res.X is None or len(res.X) == 0:
            return candidates[:n]

        # Selection: 0.5 * accuracy + 0.5 * coverage
        F = res.F
        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        F_range = F_max - F_min
        F_range = np.where(F_range == 0, 1, F_range)
        F_norm = (F - F_min) / F_range

        weighted = 0.5 * F_norm[:, 0] + 0.5 * F_norm[:, 1]
        best_idx = np.argmin(weighted)

        best_chrom = res.X[best_idx].astype(int) % len(candidates)
        return [candidates[idx] for idx in best_chrom]
