"""
baselines/dcrlrec.py
Baseline 9: DCRLRec (Bai et al., 2025)
"DCRLRec: Dual-domain contrastive reinforcement large language model for recommendation."

Simplified approximation:
- Domain A: Rating-based MF embeddings (rank=32)
- Domain B: Content-based genre embeddings via MLP
- Cross-domain InfoNCE alignment
- REINFORCE fine-tuning
- Direct ranking model (no GA)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Func

from baselines.base_recommender import BaseRecommender
from baselines.cf_utils import build_rating_matrix, get_genre_matrix, get_train_user_history
from data_loader import GENRE_COLS


class DualDomainModel(nn.Module):
    """
    Dual-domain embedding model:
    Domain A: Rating-based user/item embeddings (rank=32)
    Domain B: Content-based item embeddings via MLP (19 → 64 → ReLU → 32)
              + user embeddings as mean of interacted item content embeddings
    """

    def __init__(self, n_users, n_items, n_genres=19, emb_dim=32):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim

        # Domain A: Rating-based
        self.user_emb_a = nn.Embedding(n_users, emb_dim)
        self.item_emb_a = nn.Embedding(n_items, emb_dim)

        # Domain B: Content-based MLP for items
        self.content_mlp = nn.Sequential(
            nn.Linear(n_genres, 64),
            nn.ReLU(),
            nn.Linear(64, emb_dim),
        )
        # Domain B: User embeddings (learned)
        self.user_emb_b = nn.Embedding(n_users, emb_dim)

        nn.init.xavier_uniform_(self.user_emb_a.weight)
        nn.init.xavier_uniform_(self.item_emb_a.weight)
        nn.init.xavier_uniform_(self.user_emb_b.weight)

    def get_item_content_emb(self, genre_matrix_tensor):
        """Get content-based item embeddings from genre vectors."""
        return self.content_mlp(genre_matrix_tensor)

    def forward_a(self, user_ids, item_ids):
        """Domain A predictions: dot product of rating embeddings."""
        u_emb = self.user_emb_a(user_ids)
        i_emb = self.item_emb_a(item_ids)
        return (u_emb * i_emb).sum(dim=1)

    def forward_b(self, user_ids, item_content_embs):
        """Domain B predictions: dot product of user emb with content item emb."""
        u_emb = self.user_emb_b(user_ids)
        return (u_emb * item_content_embs).sum(dim=1)


class DCRLRec(BaseRecommender):
    """
    DCRLRec baseline: Dual-domain contrastive + reinforcement learning recommendation.
    Simplified approximation for MovieLens 100k (no LLM).
    """

    def __init__(self, data, emb_dim=32, n_epochs=20, reinforce_steps=5,
                 lr=0.001, contrastive_weight=0.1,
                 top_n=10, candidate_pool_size=200):
        super().__init__(data, name="DCRLRec", top_n=top_n, candidate_pool_size=candidate_pool_size)
        self.emb_dim = emb_dim
        self.n_epochs = n_epochs
        self.reinforce_steps = reinforce_steps
        self.lr = lr
        self.contrastive_weight = contrastive_weight
        self.user_emb_a = None
        self.item_emb_a = None
        self.user_emb_b = None
        self.item_emb_b = None

    def fit(self, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"  [{self.name}] Training dual-domain model ({self.n_epochs} epochs)...")

        rating_matrix, train_interactions = build_rating_matrix(self.data)
        genre_matrix = get_genre_matrix(self.data)
        genre_tensor = torch.FloatTensor(genre_matrix)

        model = DualDomainModel(self.data.n_users, self.data.n_items,
                                len(GENRE_COLS), self.emb_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # Prepare training data (positive interactions)
        interactions = list(train_interactions)

        # Get test ratings for REINFORCE reward
        from baselines.cf_utils import get_test_relevant_items
        original_ratings = {}
        from data_loader import load_ratings
        orig_df = load_ratings()
        for _, row in orig_df.iterrows():
            u_idx = self.data.user2idx.get(row["user_id"])
            i_idx = self.data.item2idx.get(row["item_id"])
            if u_idx is not None and i_idx is not None:
                original_ratings[(u_idx, i_idx)] = row["rating"]

        model.train()
        for epoch in range(self.n_epochs):
            np.random.shuffle(interactions)
            total_loss = 0.0
            batch_size = 512

            for start in range(0, len(interactions), batch_size):
                batch = interactions[start:start + batch_size]
                users = torch.LongTensor([u for u, i in batch])
                items = torch.LongTensor([i for u, i in batch])

                # Negative samples
                neg_items = []
                for u, _ in batch:
                    seen = self.train_history.get(u, set())
                    neg = np.random.randint(0, self.data.n_items)
                    while neg in seen:
                        neg = np.random.randint(0, self.data.n_items)
                    neg_items.append(neg)
                neg_items = torch.LongTensor(neg_items)

                # Domain A: BPR-style loss
                pos_scores_a = model.forward_a(users, items)
                neg_scores_a = model.forward_a(users, neg_items)
                loss_a = -Func.logsigmoid(pos_scores_a - neg_scores_a).mean()

                # Domain B: Content-based loss
                item_content = model.get_item_content_emb(genre_tensor[items])
                neg_content = model.get_item_content_emb(genre_tensor[neg_items])
                pos_scores_b = model.forward_b(users, item_content)
                neg_scores_b = model.forward_b(users, neg_content.detach())
                loss_b = -Func.logsigmoid(pos_scores_b - neg_scores_b).mean()

                # Cross-domain contrastive alignment (InfoNCE)
                item_emb_a = model.item_emb_a(items)
                item_emb_b = model.get_item_content_emb(genre_tensor[items])

                item_emb_a_norm = Func.normalize(item_emb_a, dim=1)
                item_emb_b_norm = Func.normalize(item_emb_b, dim=1)

                temperature = 0.2
                pos_sim = (item_emb_a_norm * item_emb_b_norm).sum(dim=1) / temperature
                all_sim = torch.mm(item_emb_a_norm, item_emb_b_norm.t()) / temperature
                contrastive_loss = (-pos_sim + torch.logsumexp(all_sim, dim=1)).mean()

                loss = loss_a + loss_b + self.contrastive_weight * contrastive_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                n_batches = max(1, len(interactions) // batch_size)
                print(f"    Epoch {epoch + 1}/{self.n_epochs}, Loss: {total_loss / n_batches:.4f}")

        # REINFORCE fine-tuning (simplified)
        print(f"  [{self.name}] Running REINFORCE fine-tuning ({self.reinforce_steps} steps)...")
        model.train()
        reinforce_optimizer = torch.optim.Adam(
            list(model.user_emb_a.parameters()) + list(model.item_emb_a.parameters()),
            lr=self.lr * 0.1
        )

        val_pos = self.data.val_df[self.data.val_df["label"] == 1]
        val_interactions = []
        for _, row in val_pos.iterrows():
            u = int(row["user_idx"])
            i = int(row["item_idx"])
            rating = original_ratings.get((u, i), 0)
            val_interactions.append((u, i, rating / 5.0))  # Normalize reward

        if val_interactions:
            for step in range(self.reinforce_steps):
                np.random.shuffle(val_interactions)
                batch = val_interactions[:min(256, len(val_interactions))]
                users = torch.LongTensor([u for u, i, r in batch])
                items = torch.LongTensor([i for u, i, r in batch])
                rewards = torch.FloatTensor([r for u, i, r in batch])

                scores = model.forward_a(users, items)
                log_probs = Func.logsigmoid(scores)
                policy_loss = -(log_probs * rewards).mean()

                reinforce_optimizer.zero_grad()
                policy_loss.backward()
                reinforce_optimizer.step()

        # Store final embeddings
        model.eval()
        with torch.no_grad():
            self.user_emb_a = model.user_emb_a.weight.numpy()
            self.item_emb_a = model.item_emb_a.weight.numpy()
            self.user_emb_b = model.user_emb_b.weight.numpy()
            self.item_emb_b = model.get_item_content_emb(genre_tensor).numpy()

        print(f"  [{self.name}] Fit complete.")

    def recommend(self, user_idx, n=10):
        """Score = 0.5 * dot(u_A, i_A) + 0.5 * dot(u_B, i_B), top-N by score."""
        scores_a = self.user_emb_a[user_idx] @ self.item_emb_a.T
        scores_b = self.user_emb_b[user_idx] @ self.item_emb_b.T
        scores = 0.5 * scores_a + 0.5 * scores_b

        seen = self.train_history.get(user_idx, set())
        for i in seen:
            scores[i] = -np.inf

        top_n_idx = np.argsort(-scores)[:n]
        return top_n_idx.tolist()
