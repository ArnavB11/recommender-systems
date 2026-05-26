import torch
import torch.nn as nn

EMB_DIM = 32
MLP_LAYERS = [64, 32, 16]
DROPOUT = 0.2

class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=EMB_DIM,
                 mlp_layers=MLP_LAYERS, dropout=DROPOUT):
        super().__init__()

        self.gmf_user_emb = nn.Embedding(n_users, emb_dim)
        self.gmf_item_emb = nn.Embedding(n_items, emb_dim)

        self.mlp_user_emb = nn.Embedding(n_users, emb_dim)
        self.mlp_item_emb = nn.Embedding(n_items, emb_dim)

        layers = []
        input_size = emb_dim * 2
        for hidden_size in mlp_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_size = hidden_size
        self.mlp_tower = nn.Sequential(*layers)

        self.fusion_layer = nn.Linear(emb_dim + mlp_layers[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, user_ids, item_ids):
        gmf_out = self.gmf_user_emb(user_ids) * self.gmf_item_emb(item_ids)

        mlp_input = torch.cat([
            self.mlp_user_emb(user_ids),
            self.mlp_item_emb(item_ids),
        ], dim=-1)
        mlp_out = self.mlp_tower(mlp_input)

        fused = torch.cat([gmf_out, mlp_out], dim=-1)
        logit = self.fusion_layer(fused)
        return torch.sigmoid(logit).squeeze(-1)

if __name__ == "__main__":
    model = NeuMF(943, 1682)
    total_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(model)
    print(f"\nTotal trainable parameters: {total_params:,}")

    dummy_users = torch.randint(0, 943, (8,))
    dummy_items = torch.randint(0, 1682, (8,))
    out = model(dummy_users, dummy_items)
    print(f"\nOutput shape : {out.shape}")
    print(f"Output range : [{out.min().item():.4f}, {out.max().item():.4f}]")