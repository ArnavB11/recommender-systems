import os
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

def get_pure_signal(model, data, user_idx):
    unseen = list(set(range(data.n_items)) - data.user_history.get(user_idx, set()))
    unseen_tensor = torch.LongTensor(unseen)
    user_tensor = torch.LongTensor([user_idx] * len(unseen))

    scores = []
    with torch.no_grad():
        for start in range(0, len(unseen), 512):
            end = start + 512
            batch_scores = model(user_tensor[start:end],
                                 unseen_tensor[start:end])
            scores.append(batch_scores)

    scores = torch.cat(scores).numpy()

    order = scores.argsort()[::-1]
    top_indices = order[:CANDIDATE_POOL]

    return {
        "user_idx": user_idx,
        "item_indices": [int(unseen[i]) for i in top_indices],
        "scores": [float(scores[i]) for i in top_indices],
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
    pure_signal = get_pure_signal(model, data, user_idx)
    display_recommendations(pure_signal, data)

    print(f"\nTotal items scored : {len(set(range(data.n_items)) - data.user_history.get(user_idx, set()))}")
    print(f"Pool size          : {len(pure_signal['item_indices'])}")
    print(f"Score min          : {min(pure_signal['scores']):.4f}")
    print(f"Score max          : {max(pure_signal['scores']):.4f}")
