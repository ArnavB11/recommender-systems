import os
import numpy as np
import torch
from ncf_model import NeuMF

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE = os.path.join(BASE_DIR, "saved_model", "ncf_model.pt")
TOP_K = 10
CANDIDATE_POOL = 200

def load_trained_model(data):
    model = NeuMF(data.n_users, data.n_items)
    model.load_state_dict(torch.load(MODEL_SAVE, map_location="cpu"))
    model.eval()
    return model

def get_pure_signal(R_hat_row, user_history):
    scores = np.asarray(R_hat_row)
    if scores.ndim != 1:
        raise ValueError("R_hat_row must be a 1D array of item scores.")

    unseen_mask = np.ones(scores.shape[0], dtype=bool)
    if user_history:
        unseen_mask[list(user_history)] = False

    unseen_indices = np.flatnonzero(unseen_mask)
    if len(unseen_indices) == 0:
        return {
            "item_indices": [],
            "scores": [],
        }

    unseen_scores = scores[unseen_indices]
    top_count = min(CANDIDATE_POOL, len(unseen_indices))
    top_positions = np.argpartition(unseen_scores, -top_count)[-top_count:]
    top_positions = top_positions[
        np.argsort(-unseen_scores[top_positions], kind="mergesort")
    ]
    top_item_indices = unseen_indices[top_positions]

    return {
        "item_indices": [int(item_idx) for item_idx in top_item_indices],
        "scores": [float(score) for score in unseen_scores[top_positions]],
    }

def display_recommendations(pure_signal, data, top_n=TOP_K):
    print(f"\n{'Rank':<6}{'Title':<45}{'Year':<7}{'Genre':<15}{'NCF Score'}")
    print("-" * 80)

    for rank, (item_idx, score) in enumerate(
        zip(pure_signal["item_indices"][:top_n],
            pure_signal["scores"][:top_n]),
        start=1,
    ):
        item_id = data.idx2item[item_idx]
        row = data.items_df.loc[item_id]
        title = row["title"][:42]
        year = int(row["year"]) if row["year"] == row["year"] else "N/A"
        genre = row["primary_genre"]
        print(f"{rank:<6}{title:<45}{year!s:<7}{genre:<15}{score:.4f}")

if __name__ == "__main__":
    from data_loader import MovieLensData

    data = MovieLensData()
    model = load_trained_model(data)

    user_idx = 0
    all_items = torch.arange(data.n_items)
    user_tensor = torch.full((data.n_items,), user_idx, dtype=torch.long)
    with torch.no_grad():
        R_hat_row = model(user_tensor, all_items).numpy()

    pure_signal = get_pure_signal(R_hat_row, data.user_history.get(user_idx, set()))
    display_recommendations(pure_signal, data)

    print(f"\nTotal unseen items : {len(set(range(data.n_items)) - data.user_history.get(user_idx, set()))}")
    print(f"Pool size          : {len(pure_signal['item_indices'])}")
    print(f"Score min          : {min(pure_signal['scores']):.4f}")
    print(f"Score max          : {max(pure_signal['scores']):.4f}")
