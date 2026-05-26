import os
import torch
import torch.nn as nn
from data_loader import get_dataloaders
from ncf_model import NeuMF

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE = os.path.join(BASE_DIR, "saved_model", "ncf_model.pt")
EPOCHS = 20
LR = 0.001
SEED = 42

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for users, items, labels in loader:
        users = users.to(device)
        items = items.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        preds = model(users, items)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)

    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for users, items, labels in loader:
            users = users.to(device)
            items = items.to(device)
            labels = labels.to(device)

            preds = model(users, items)
            loss = criterion(preds, labels)
            total_loss += loss.item() * len(labels)
            n_samples += len(labels)

    return total_loss / n_samples

def train_ncf(data):
    torch.manual_seed(SEED)

    model = NeuMF(data.n_users, data.n_items)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    train_loader, val_loader = get_dataloaders(data)

    os.makedirs(os.path.dirname(MODEL_SAVE), exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE)
            marker = "  [SAVED]"

        print(f"Epoch {epoch:>2}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}{marker}")

    print(f"\nBest val loss : {best_val_loss:.4f}")
    return model, data

if __name__ == "__main__":
    from data_loader import MovieLensData
    train_ncf(MovieLensData())