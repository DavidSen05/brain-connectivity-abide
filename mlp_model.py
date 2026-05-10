import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── network ───────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128),       nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64),        nn.ReLU(),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x)).squeeze(-1)   # raw logit — no sigmoid


# ── helpers ───────────────────────────────────────────────────────────────────

def _pos_weight(y: np.ndarray) -> torch.Tensor:
    n_pos = max((y == 1).sum(), 1)
    n_neg = (y == 0).sum()
    pw    = n_neg / n_pos
    print(f"    pos_weight={pw:.2f}  (n_pos={n_pos}, n_neg={n_neg})")
    return torch.tensor([pw], dtype=torch.float32).to(DEVICE)


def _weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    counts  = np.bincount(y.astype(int))
    weights = 1.0 / counts[y.astype(int)]
    return WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.float64),
        num_samples=len(weights), replacement=True)


def _to_tensor(X, y, dtype_y=torch.float32):
    return TensorDataset(
        torch.tensor(X, dtype=torch.float32).to(DEVICE),
        torch.tensor(y, dtype=dtype_y).to(DEVICE))


# ── classification ────────────────────────────────────────────────────────────

def train_mlp_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    lr: float        = 1e-3,
    batch_size: int  = 32,
    max_epochs: int  = 100,
    patience: int    = 10,
    val_frac: float  = 0.10,
    random_seed: int = 42,
) -> MLP:
    print(f"  Training MLP classifier on {DEVICE}...")
    torch.manual_seed(random_seed)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_frac,
        random_state=random_seed, stratify=y_train)

    pw       = _pos_weight(y_tr)
    sampler  = _weighted_sampler(y_tr)
    train_dl = DataLoader(_to_tensor(X_tr, y_tr.astype(np.float32)),
                          batch_size=batch_size, sampler=sampler)
    val_dl   = DataLoader(_to_tensor(X_val, y_val.astype(np.float32)),
                          batch_size=256, shuffle=False)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    model     = MLP(input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=4, factor=0.5)

    best_loss, best_state, pat = float("inf"), None, 0

    for epoch in range(max_epochs):
        model.train()
        for X_b, y_b in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        vl, ns = 0.0, 0
        with torch.no_grad():
            for X_b, y_b in val_dl:
                vl += criterion(model(X_b), y_b).item() * len(y_b)
                ns += len(y_b)
        vl /= max(ns, 1)
        scheduler.step(vl)

        if vl < best_loss - 1e-5:
            best_loss  = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat        = 0
        else:
            pat += 1
        if pat >= patience:
            print(f"    Early stop epoch {epoch+1} (val={best_loss:.4f})")
            break

    model.load_state_dict(best_state)
    return model


def predict_mlp_classifier(model: MLP, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits  = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
        y_proba = torch.sigmoid(logits).cpu().numpy()   # sigmoid at inference
    return (y_proba >= 0.5).astype(int), y_proba


# ── regression ────────────────────────────────────────────────────────────────

def train_mlp_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    lr: float        = 1e-3,
    batch_size: int  = 32,
    max_epochs: int  = 100,
    patience: int    = 10,
    val_frac: float  = 0.10,
    random_seed: int = 42,
) -> MLP:
    print(f"  Training MLP regressor on {DEVICE}...")
    torch.manual_seed(random_seed)

    # normalise age to [0,1] for stable gradient flow
    y_min, y_max = y_train.min(), y_train.max()
    y_norm = (y_train - y_min) / (y_max - y_min + 1e-8)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_norm, test_size=val_frac, random_state=random_seed)

    train_dl  = DataLoader(_to_tensor(X_tr,  y_tr.astype(np.float32)),
                           batch_size=batch_size, shuffle=True)
    val_dl    = DataLoader(_to_tensor(X_val, y_val.astype(np.float32)),
                           batch_size=256, shuffle=False)
    criterion = nn.MSELoss()
    model     = MLP(input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=4, factor=0.5)

    best_loss, best_state, pat = float("inf"), None, 0

    for epoch in range(max_epochs):
        model.train()
        for X_b, y_b in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        vl, ns = 0.0, 0
        with torch.no_grad():
            for X_b, y_b in val_dl:
                vl += criterion(model(X_b), y_b).item() * len(y_b); ns += len(y_b)
        vl /= max(ns, 1)
        scheduler.step(vl)

        if vl < best_loss - 1e-5:
            best_loss  = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat        = 0
        else:
            pat += 1
        if pat >= patience:
            print(f"    Early stop epoch {epoch+1} (val={best_loss:.4f})")
            break

    model.load_state_dict(best_state)
    # store scale for inverse transform at predict time
    model._age_min = float(y_min)
    model._age_max = float(y_max)
    return model


def predict_mlp_regressor(model: MLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    return out * (model._age_max - model._age_min) + model._age_min


# ── public API ────────────────────────────────────────────────────────────────

def run_mlp(split, random_seed: int = 42) -> Dict:
    print("\n── MLP v2 (pos_weight + weighted sampler) ─────────────")
    X_tr, X_te = split.X_train_pca, split.X_test_pca
    dim        = X_tr.shape[1]

    clf        = train_mlp_classifier(X_tr, split.y_cls_train, dim, random_seed=random_seed)
    y_pred, y_proba = predict_mlp_classifier(clf, X_te)

    reg        = train_mlp_regressor(X_tr, split.y_age_train, dim, random_seed=random_seed)
    y_age      = predict_mlp_regressor(reg, X_te)

    return {
        "model_name":  "MLP",
        "y_pred_cls":  y_pred,
        "y_proba_cls": y_proba,
        "y_true_cls":  split.y_cls_test,
        "y_pred_age":  y_age,
        "y_true_age":  split.y_age_test,
    }