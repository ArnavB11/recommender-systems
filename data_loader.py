import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "ml-100k")
RATINGS_FILE = os.path.join(DATA_DIR, "u.data")
ITEMS_FILE = os.path.join(DATA_DIR, "u.item")
USERS_FILE = os.path.join(DATA_DIR, "u.user")
BATCH_SIZE = 256
NEG_SAMPLES = 4
TEST_SIZE = 0.2
SEED = 42

GENRE_COLS = [
    "unknown", "Action", "Adventure", "Animation", "Children",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]

def load_ratings():
    return pd.read_csv(
        RATINGS_FILE, sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )

def load_items():
    col_names = [
        "item_id", "title", "release_date", "video_date", "external_url",
    ] + GENRE_COLS

    df = pd.read_csv(
        ITEMS_FILE, sep="|", encoding="latin-1",
        names=col_names, header=None,
    )

    df["year"] = pd.to_datetime(
        df["release_date"], format="%d-%b-%Y", errors="coerce"
    ).dt.year

    def _primary(row):
        for genre in GENRE_COLS:
            if row[genre] == 1:
                return genre
        return "unknown"

    df["primary_genre"] = df.apply(_primary, axis=1)
    df["genres"] = df.apply(
        lambda row: "|".join(genre for genre in GENRE_COLS if row[genre] == 1) or "unknown",
        axis=1,
    )
    df["age"] = 2025 - df["year"]
    df["age"] = df["age"].fillna(df["age"].median())

    keep = ["item_id", "title", "year", "age", "primary_genre", "genres"] + GENRE_COLS
    return df[keep]

def load_users():
    return pd.read_csv(
        USERS_FILE, sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"],
    )

class MovieLensData:
    def __init__(self):
        ratings_df = load_ratings()
        items_df = load_items()
        users_df = load_users()

        sorted_users = sorted(ratings_df["user_id"].unique())
        sorted_items = sorted(ratings_df["item_id"].unique())

        self.user2idx = {user_id: idx for idx, user_id in enumerate(sorted_users)}
        self.item2idx = {item_id: idx for idx, item_id in enumerate(sorted_items)}
        self.idx2item = {idx: item_id for item_id, idx in self.item2idx.items()}

        self.n_users = len(self.user2idx)
        self.n_items = len(self.item2idx)

        ratings_df["user_idx"] = ratings_df["user_id"].map(self.user2idx)
        ratings_df["item_idx"] = ratings_df["item_id"].map(self.item2idx)

        self.items_df = items_df.set_index("item_id")
        self.users_df = users_df.set_index("user_id")

        self.user_history = (
            ratings_df.groupby("user_idx")["item_idx"]
            .apply(set).to_dict()
        )

        self.item_avg_rating = (
            ratings_df.groupby("item_idx")["rating"]
            .mean().to_dict()
        )
        self.item_popularity = (
            ratings_df.groupby("item_idx")["rating"]
            .count().to_dict()
        )
        self.pop_max = max(self.item_popularity.values())

        train_df, val_df = train_test_split(
            ratings_df, test_size=TEST_SIZE, random_state=SEED,
        )
        self.train_df = self._add_negatives(train_df)
        self.val_df = self._add_negatives(val_df)

        total_possible = self.n_users * self.n_items
        sparsity = (1 - len(ratings_df) / total_possible) * 100
        print(f"  Users : {self.n_users}")
        print(f"  Items : {self.n_items}")
        print(f"  Train : {len(self.train_df):,} rows (with negatives)")
        print(f"  Val   : {len(self.val_df):,} rows (with negatives)")
        print(f"  Sparsity : {sparsity:.2f}%")

    def _add_negatives(self, df):
        rng = np.random.default_rng(SEED)
        all_items = set(range(self.n_items))
        rows = []

        for _, row in df.iterrows():
            user = int(row["user_idx"])
            item = int(row["item_idx"])
            rows.append((user, item, 1))
            neg_pool = list(all_items - self.user_history.get(user, set()))
            negs = rng.choice(
                neg_pool,
                size=min(NEG_SAMPLES, len(neg_pool)),
                replace=False,
            )
            for neg_item in negs:
                rows.append((user, int(neg_item), 0))

        return pd.DataFrame(rows, columns=["user_idx", "item_idx", "label"])

class InteractionDataset(Dataset):
    def __init__(self, df):
        self.users = torch.LongTensor(df["user_idx"].values)
        self.items = torch.LongTensor(df["item_idx"].values)
        self.labels = torch.FloatTensor(df["label"].values)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

def get_dataloaders(data):
    train_ds = InteractionDataset(data.train_df)
    val_ds = InteractionDataset(data.val_df)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    return train_loader, val_loader

if __name__ == "__main__":
    data = MovieLensData()
    train_loader, val_loader = get_dataloaders(data)

    users, items, labels = next(iter(train_loader))
    print(f"\nBatch shapes - users: {users.shape}, items: {items.shape}, "
          f"labels: {labels.shape}")

    print("\n-- Sample items " + "-" * 40)
    sample = data.items_df.head(5)[["title", "year", "age", "primary_genre"]]
    print(sample.to_string())

    print("\n-- Top-5 most popular (by rating count) " + "-" * 20)
    top5 = sorted(data.item_popularity.items(), key=lambda item: -item[1])[:5]
    for idx, count in top5:
        item_id = data.idx2item[idx]
        title = data.items_df.loc[item_id, "title"]
        print(f"  {title:40s}  ratings={count}")