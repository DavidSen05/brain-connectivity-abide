import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K  = 10    # top-k positive correlations per node


# ── graph construction ────────────────────────────────────────────────────────

def subject_to_graph(subject):
    """
    Convert one SubjectData into (node_features, edge_index, edge_weight).
    Node features are the Fisher-z transformed FC profile for each ROI plus
    compact node-level summaries. Top-k positive edges define the graph.
    """
    fc = subject.fc_matrix       # (200, 200)

    fc_clean = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    np.fill_diagonal(fc_clean, 0.0)

    fc_z = np.arctanh(np.clip(fc_clean, -0.999999, 0.999999))
    pos_strength = np.maximum(fc_clean, 0.0).sum(axis=1, keepdims=True)
    neg_strength = np.abs(np.minimum(fc_clean, 0.0)).sum(axis=1, keepdims=True)
    mean_conn = fc_clean.mean(axis=1, keepdims=True)
    std_conn = fc_clean.std(axis=1, keepdims=True)
    node_feat = np.concatenate(
        [fc_z, pos_strength, neg_strength, mean_conn, std_conn],
        axis=1,
    ).astype(np.float32)

    n = fc_clean.shape[0]
    edges = {}
    for i in range(n):
        row = fc_clean[i].copy()
        row[i] = -1.0
        top_idx = np.argsort(row)[-TOP_K:]
        for j in top_idx:
            if row[j] > 0:
                a, b = sorted((i, int(j)))
                edges[(a, b)] = max(float(row[j]), edges.get((a, b), 0.0))

    if not edges:
        rows = np.arange(n, dtype=np.int64)
        cols = np.arange(n, dtype=np.int64)
        wts = np.ones(n, dtype=np.float32)
    else:
        rows = np.array([e[0] for e in edges.keys()], dtype=np.int64)
        cols = np.array([e[1] for e in edges.keys()], dtype=np.int64)
        wts = np.array(list(edges.values()), dtype=np.float32)

    edge_index  = np.stack([np.concatenate([rows, cols]),
                             np.concatenate([cols, rows])], axis=0)
    edge_weight = np.concatenate([wts, wts])

    return (torch.tensor(node_feat,   dtype=torch.float32),
            torch.tensor(edge_index,  dtype=torch.long),
            torch.tensor(edge_weight, dtype=torch.float32))


# ── try torch-geometric; dense fallback otherwise ────────────────────────────

try:
    from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data
    _USE_PYG = True
    print("  [GCN] torch-geometric found")

    class GCNModel(nn.Module):
        def __init__(self, in_channels: int = 204):
            super().__init__()
            self.conv1 = GCNConv(in_channels, 128)
            self.bn1   = nn.BatchNorm1d(128)
            self.conv2 = GCNConv(128, 64)
            self.bn2   = nn.BatchNorm1d(64)
            self.fc1   = nn.Linear(128, 64)
            self.head  = nn.Linear(64, 1)
            self.drop  = nn.Dropout(0.3)

        def forward(self, data):
            x, ei, ew, batch = data.x, data.edge_index, data.edge_attr, data.batch
            x = F.relu(self.bn1(self.conv1(x, ei, ew)))
            x = self.drop(x)
            x = F.relu(self.bn2(self.conv2(x, ei, ew)))
            x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
            x = F.relu(self.fc1(x))
            x = self.drop(x)
            return self.head(x).squeeze(-1)   # raw logit — no sigmoid

    def _make_data_list(subjects, labels):
        out = []
        for s, lbl in zip(subjects, labels):
            nf, ei, ew = subject_to_graph(s)
            out.append(Data(x=nf, edge_index=ei, edge_attr=ew,
                            y=torch.tensor([lbl], dtype=torch.float32)))
        return out

    def _make_loader(data_list, batch_size, shuffle, sampler=None):
        from torch_geometric.loader import DataLoader as PygLoader
        return PygLoader(data_list, batch_size=batch_size,
                         shuffle=(shuffle and sampler is None), sampler=sampler)

except ImportError:
    _USE_PYG = False
    print("  [GCN] torch-geometric not found — dense fallback")

    class GCNModel(nn.Module):
        def __init__(self, in_channels: int = 204):
            super().__init__()
            self.W1   = nn.Linear(in_channels, 128, bias=False)
            self.bn1  = nn.BatchNorm1d(128)
            self.W2   = nn.Linear(128, 64, bias=False)
            self.bn2  = nn.BatchNorm1d(64)
            self.fc1  = nn.Linear(128, 64)
            self.head = nn.Linear(64,  1)
            self.drop = nn.Dropout(0.3)

        def _gcn(self, A_hat, X, W):
            return torch.einsum("bij,bjd->bid", A_hat, W(X))

        def _bn_nodes(self, x, bn):
            b, n, d = x.shape
            return bn(x.reshape(b * n, d)).reshape(b, n, d)

        def forward(self, batch):
            A_hat, X = batch
            h = F.relu(self._bn_nodes(self._gcn(A_hat, X, self.W1), self.bn1))
            h = self.drop(h)
            h = F.relu(self._bn_nodes(self._gcn(A_hat, h, self.W2), self.bn2))
            h = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=1)
            h = F.relu(self.fc1(h))
            h = self.drop(h)
            return self.head(h).squeeze(-1)   # raw logit — no sigmoid

    def _norm_adj(fc: np.ndarray) -> np.ndarray:
        n = fc.shape[0]
        A = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            row = fc[i].copy(); row[i] = -1.0
            for j in np.argsort(row)[-TOP_K:]:
                if row[j] > 0:
                    A[i, j] = row[j]; A[j, i] = row[j]
        A += np.eye(n, dtype=np.float32)
        d  = np.diag(1.0 / np.sqrt(A.sum(1) + 1e-8))
        return d @ A @ d

    def _make_data_list(subjects, labels):
        items = []
        for s, lbl in zip(subjects, labels):
            nf, _, _ = subject_to_graph(s)
            adj = _norm_adj(s.fc_matrix)
            items.append((adj, nf.numpy(), float(lbl)))
        return items

    class _DenseLoader:
        def __init__(self, items, batch_size, shuffle):
            self.items = items; self.batch_size = batch_size; self.shuffle = shuffle
        def __iter__(self):
            idx = np.arange(len(self.items))
            if self.shuffle: np.random.shuffle(idx)
            for s in range(0, len(idx), self.batch_size):
                b = [self.items[i] for i in idx[s:s+self.batch_size]]
                adjs, nfs, lbls = zip(*b)
                A = torch.tensor(np.stack(adjs), dtype=torch.float32).to(DEVICE)
                X = torch.tensor(np.stack(nfs),  dtype=torch.float32).to(DEVICE)
                y = torch.tensor(list(lbls),     dtype=torch.float32).to(DEVICE)
                yield (A, X), y

    def _make_loader(data_list, batch_size, shuffle, sampler=None):
        return _DenseLoader(data_list, batch_size, shuffle)


# ── helpers ───────────────────────────────────────────────────────────────────

def _pos_weight(labels: np.ndarray) -> torch.Tensor:
    n_pos = max((labels == 1).sum(), 1)
    n_neg = (labels == 0).sum()
    pw    = n_neg / n_pos
    print(f"    pos_weight={pw:.2f}  (n_pos={n_pos}, n_neg={n_neg})")
    return torch.tensor([pw], dtype=torch.float32).to(DEVICE)


def _weighted_sampler(labels: np.ndarray):
    from torch.utils.data import WeightedRandomSampler
    counts  = np.bincount(labels.astype(int))
    weights = 1.0 / counts[labels.astype(int)]
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.float64),
                                 num_samples=len(weights), replacement=True)


# ── training loops ────────────────────────────────────────────────────────────

def _train(model, data_list, labels, task,
           lr=1e-3, batch_size=16, max_epochs=80,
           patience=12, val_frac=0.10, seed=42):
    torch.manual_seed(seed)

    idx = np.arange(len(data_list))
    stratify = labels if task == "classification" else None
    tr_idx, vl_idx = train_test_split(
        idx, test_size=val_frac, random_state=seed, stratify=stratify
    )
    tr_data = [data_list[i] for i in tr_idx]
    vl_data = [data_list[i] for i in vl_idx]
    tr_lbl = labels[tr_idx] if labels is not None else None

    if task == "classification":
        pw        = _pos_weight(tr_lbl)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        sampler   = _weighted_sampler(tr_lbl)
        train_dl  = _make_loader(tr_data, batch_size, shuffle=False, sampler=sampler)
    else:
        criterion = nn.MSELoss()
        train_dl  = _make_loader(tr_data, batch_size, shuffle=True)

    val_dl    = _make_loader(vl_data, batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5)

    best_loss, best_state, pat_cnt = float("inf"), None, 0

    for epoch in range(max_epochs):
        model.train()
        for batch in train_dl:
            optimizer.zero_grad()
            if _USE_PYG:
                batch  = batch.to(DEVICE)
                logits = model(batch)
                y      = batch.y.squeeze()
            else:
                (A, X), y = batch
                logits = model((A, X))
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        vl, ns = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                if _USE_PYG:
                    batch  = batch.to(DEVICE)
                    logits = model(batch); y = batch.y.squeeze()
                else:
                    (A, X), y = batch; logits = model((A, X))
                vl += criterion(logits, y).item() * len(y); ns += len(y)
        vl /= max(ns, 1)
        scheduler.step(vl)

        if vl < best_loss - 1e-5:
            best_loss  = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat_cnt    = 0
        else:
            pat_cnt += 1
        if pat_cnt >= patience:
            print(f"    Early stop epoch {epoch+1} (val={best_loss:.4f})")
            break

    model.load_state_dict(best_state)
    return model


def _infer(model, data_list, task):
    model.eval()
    preds  = []
    loader = _make_loader(data_list, 32, False)
    with torch.no_grad():
        for batch in loader:
            if _USE_PYG:
                batch = batch.to(DEVICE); out = model(batch)
            else:
                (A, X), _ = batch; out = model((A, X))
            if task == "classification":
                out = torch.sigmoid(out)   # sigmoid at inference only
            preds.append(out.cpu().numpy())
    return np.concatenate(preds)


# ── public API ────────────────────────────────────────────────────────────────

def run_gcn(split, random_seed: int = 42) -> Dict:
    print("\n── GCN v2 (pos_weight + top-k graph) ─────────────────")

    # classification
    cls_lbl   = split.y_cls_train.astype(float)
    train_cls = _make_data_list(split.subjects_train, cls_lbl)
    test_cls  = _make_data_list(split.subjects_test, split.y_cls_test.astype(float))

    print(f"  Training GCN classifier on {DEVICE}...")
    in_channels = train_cls[0].x.shape[1] if _USE_PYG else train_cls[0][1].shape[1]
    clf = GCNModel(in_channels=in_channels).to(DEVICE)
    clf = _train(clf, train_cls, split.y_cls_train, task="classification",
                 seed=random_seed)
    y_proba = _infer(clf, test_cls, "classification")
    y_pred  = (y_proba >= 0.5).astype(int)

    # regression — normalise age to [0,1]
    amin, amax = split.y_age_train.min(), split.y_age_train.max()
    anorm      = (split.y_age_train - amin) / (amax - amin + 1e-8)
    train_reg  = _make_data_list(split.subjects_train, anorm)
    test_reg   = _make_data_list(split.subjects_test, np.zeros(len(split.subjects_test)))

    print(f"  Training GCN regressor on {DEVICE}...")
    reg = GCNModel(in_channels=in_channels).to(DEVICE)
    reg = _train(reg, train_reg, anorm, task="regression", seed=random_seed)
    y_age_norm = _infer(reg, test_reg, "regression")
    y_age      = y_age_norm * (amax - amin) + amin

    return {
        "model_name":  "GCN",
        "y_pred_cls":  y_pred,
        "y_proba_cls": y_proba,
        "y_true_cls":  split.y_cls_test,
        "y_pred_age":  y_age,
        "y_true_age":  split.y_age_test,
    }
