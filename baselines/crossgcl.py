"""
baselines/crossgcl.py
Baseline 8: CrossGCL (Ye & Xu, 2024)
"CrossGCL: Cross-pairwise graph contrastive learning for unbiased recommendation."

LightGCN-style graph propagation + BPR loss + InfoNCE contrastive loss.
Direct ranking model (no GA).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from baselines.base_recommender import BaseRecommender
from baselines.cf_utils import get_train_user_history


class BPRDataset(Dataset):
    """Dataset of (user, pos_item, neg_item) triples for BPR training."""

    def __init__(self, train_interactions, n_items, user_history):
        self.interactions = list(train_interactions)
        self.n_items = n_items
        self.user_history = user_history

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        u, i = self.interactions[idx]
        # Sample negative item
        seen = self.user_history.get(u, set())
        neg = np.random.randint(0, self.n_items)
        while neg in seen:
            neg = np.random.randint(0, self.n_items)
        return u, i, neg


class LightGCN(nn.Module):
    """
    LightGCN-style model with 2-layer propagation.
    Final embedding = mean of all layer outputs.
    """

    def __init__(self, n_users, n_items, emb_dim=64, n_layers=2):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        self.norm_adj = None  # Set externally

    def set_norm_adj(self, norm_adj_sparse):
        """Set the normalized adjacency matrix (sparse tensor)."""
        self.norm_adj = norm_adj_sparse

    def forward(self):
        """Propagate through the graph and return final user/item embeddings."""
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)

        emb_list = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            emb_list.append(all_emb)

        final_emb = torch.stack(emb_list, dim=0).mean(dim=0)
        user_final = final_emb[:self.n_users]
        item_final = final_emb[self.n_users:]
        return user_final, item_final

    def bpr_loss(self, user_emb, pos_emb, neg_emb):
        """BPR loss for implicit feedback."""
        pos_scores = (user_emb * pos_emb).sum(dim=1)
        neg_scores = (user_emb * neg_emb).sum(dim=1)
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        return loss

    def infonce_loss(self, anchor, positive, temperature=0.2):
        """InfoNCE contrastive loss."""
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)

        # Similarity with positive
        pos_sim = (anchor * positive).sum(dim=1) / temperature
        # Similarity with all others (negative samples)
        all_sim = torch.mm(anchor, positive.t()) / temperature
        # InfoNCE: log(exp(pos) / sum(exp(all)))
        loss = -pos_sim + torch.logsumexp(all_sim, dim=1)
        return loss.mean()


class CrossGCL(BaseRecommender):
    """
    CrossGCL baseline: LightGCN + BPR + Cross-pairwise InfoNCE contrastive learning.
    Direct ranking model - no GA needed.
    """

    def __init__(self, data, emb_dim=64, n_layers=2, n_epochs=30,
                 batch_size=512, lr=0.001, contrastive_weight=0.1,
                 temperature=0.2, top_n=10, candidate_pool_size=200):
        super().__init__(data, name="CrossGCL", top_n=top_n, candidate_pool_size=candidate_pool_size)
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.user_emb_final = None
        self.item_emb_final = None

    def _build_norm_adj(self, train_interactions):
        """Build normalized adjacency matrix D^{-0.5} A D^{-0.5}."""
        n = self.data.n_users + self.data.n_items

        rows, cols, vals = [], [], []
        for u, i in train_interactions:
            # User -> Item edge
            rows.append(u)
            cols.append(self.data.n_users + i)
            vals.append(1.0)
            # Item -> User edge
            rows.append(self.data.n_users + i)
            cols.append(u)
            vals.append(1.0)

        rows = np.array(rows)
        cols = np.array(cols)
        vals = np.array(vals, dtype=np.float32)

        # Degree vector
        degree = np.zeros(n, dtype=np.float32)
        for r in rows:
            degree[r] += 1

        # D^{-0.5}
        d_inv_sqrt = np.zeros(n, dtype=np.float32)
        mask = degree > 0
        d_inv_sqrt[mask] = 1.0 / np.sqrt(degree[mask])

        # Normalize: D^{-0.5} * A * D^{-0.5}
        norm_vals = vals * d_inv_sqrt[rows] * d_inv_sqrt[cols]

        indices = torch.LongTensor(np.stack([rows, cols]))
        values = torch.FloatTensor(norm_vals)
        norm_adj = torch.sparse_coo_tensor(indices, values, (n, n))

        return norm_adj

    def fit(self, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"  [{self.name}] Training LightGCN + CrossGCL ({self.n_epochs} epochs)...")

        # Build training interactions
        train_pos = self.data.train_df[self.data.train_df["label"] == 1]
        train_interactions = set()
        for _, row in train_pos.iterrows():
            train_interactions.add((int(row["user_idx"]), int(row["item_idx"])))

        # Build normalized adjacency
        norm_adj = self._build_norm_adj(train_interactions)

        # Model
        model = LightGCN(self.data.n_users, self.data.n_items, self.emb_dim, self.n_layers)
        model.set_norm_adj(norm_adj)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # Dataset
        dataset = BPRDataset(train_interactions, self.data.n_items, self.train_history)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        model.train()
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            for users, pos_items, neg_items in loader:
                user_final, item_final = model()

                u_emb = user_final[users]
                pos_emb = item_final[pos_items]
                neg_emb = item_final[neg_items]

                # BPR loss
                bpr = model.bpr_loss(u_emb, pos_emb, neg_emb)

                # Cross-pairwise contrastive loss (simplified InfoNCE)
                cl_loss = model.infonce_loss(u_emb, pos_emb, self.temperature)

                loss = bpr + self.contrastive_weight * cl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch + 1}/{self.n_epochs}, Loss: {total_loss / len(loader):.4f}")

        # Store final embeddings
        model.eval()
        with torch.no_grad():
            user_final, item_final = model()
            self.user_emb_final = user_final.numpy()
            self.item_emb_final = item_final.numpy()

        print(f"  [{self.name}] Fit complete. User emb: {self.user_emb_final.shape}, Item emb: {self.item_emb_final.shape}")

    def recommend(self, user_idx, n=10):
        """Top-N by dot product of user/item embeddings."""
        scores = self.user_emb_final[user_idx] @ self.item_emb_final.T
        seen = self.train_history.get(user_idx, set())

        # Mask seen items
        for i in seen:
            scores[i] = -np.inf

        top_n_idx = np.argsort(-scores)[:n]
        return top_n_idx.tolist()
