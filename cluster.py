"""
cluster.py
----------
Unsupervised K-Means clustering on PCA-reduced FC features.
Produces:
  - Elbow curve + Silhouette Score plot  (Figure 1 in paper)
  - t-SNE visualization colored by cluster and by diagnosis (Figure 2)
  - Per-cluster ASD/control breakdown table (Table 1)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from typing import Tuple
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ── colour palette (colour-blind friendly) ───────────────────────────────────
CLUSTER_COLORS = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7",
                  "#C4AD66", "#77BEDB", "#F7A54A", "#E98EC2"]
ASD_COLORS     = {"ASD": "#D65F5F", "Control": "#4878CF"}


def select_k(
    X: np.ndarray,
    k_range: range = range(2, 11),
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Evaluate K-Means for each k using Elbow (inertia) and Silhouette Score.

    Returns
    -------
    inertias      : array of within-cluster SSE for each k
    sil_scores    : array of Silhouette Scores for each k
    best_k        : k with highest Silhouette Score
    """
    inertias   = []
    sil_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        n_unique = len(set(labels))
        if n_unique < 2:
            sil_scores.append(-1.0)
        else:
            sil_scores.append(silhouette_score(X, labels, sample_size=min(500, len(X))))

    best_k = k_range[int(np.argmax(sil_scores))]
    return np.array(inertias), np.array(sil_scores), best_k


def fit_kmeans(
    X: np.ndarray,
    k: int,
    random_seed: int = 42,
) -> np.ndarray:
    """Fit K-Means with the chosen k; return cluster labels."""
    km = KMeans(n_clusters=k, random_state=random_seed, n_init=20)
    return km.fit_predict(X)


def plot_selection(
    k_range: range,
    inertias: np.ndarray,
    sil_scores: np.ndarray,
    best_k: int,
    save_path: str = "figures/fig1_cluster_selection.png",
) -> None:
    """Figure 1: side-by-side Elbow + Silhouette plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    ks = list(k_range)

    # Elbow
    ax1.plot(ks, inertias, "o-", color="#4878CF", linewidth=1.8, markersize=5)
    ax1.axvline(best_k, color="#D65F5F", linestyle="--", linewidth=1.2,
                label=f"best k = {best_k}")
    ax1.set_xlabel("Number of clusters k", fontsize=10)
    ax1.set_ylabel("Within-cluster SSE", fontsize=10)
    ax1.set_title("(a) Elbow Method", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.tick_params(labelsize=9)

    # Silhouette
    ax2.plot(ks, sil_scores, "s-", color="#6ACC65", linewidth=1.8, markersize=5)
    ax2.axvline(best_k, color="#D65F5F", linestyle="--", linewidth=1.2,
                label=f"best k = {best_k}")
    ax2.set_xlabel("Number of clusters k", fontsize=10)
    ax2.set_ylabel("Silhouette Score", fontsize=10)
    ax2.set_title("(b) Silhouette Score", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.tick_params(labelsize=9)

    fig.suptitle(
        "Figure 1: Cluster count selection using Elbow Method and Silhouette Score",
        fontsize=9, y=1.01, color="#555"
    )
    plt.tight_layout()
    import os; os.makedirs("figures", exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {save_path}")


def plot_tsne(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    y_cls: np.ndarray,
    best_k: int,
    random_seed: int = 42,
    save_path: str   = "figures/fig2_tsne.png",
) -> None:
    """Figure 2: t-SNE scatter — left: by cluster, right: by ASD/control."""
    print("  Running t-SNE (may take ~30 s on synthetic data)...")
    tsne  = TSNE(n_components=2, random_state=random_seed,
                 perplexity=30, max_iter=1000)
    emb   = tsne.fit_transform(X)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # — by cluster
    for c in range(best_k):
        mask = cluster_labels == c
        ax1.scatter(emb[mask, 0], emb[mask, 1],
                    c=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                    s=12, alpha=0.7, label=f"Cluster {c}", rasterized=True)
    ax1.set_title("(a) Colored by cluster", fontsize=10, fontweight="bold")
    ax1.legend(markerscale=2, fontsize=8, framealpha=0.6)
    ax1.axis("off")

    # — by ASD / control
    for lbl, name in [(1, "ASD"), (0, "Control")]:
        mask = y_cls == lbl
        ax2.scatter(emb[mask, 0], emb[mask, 1],
                    c=ASD_COLORS[name], s=12, alpha=0.7,
                    label=name, rasterized=True)
    ax2.set_title("(b) Colored by diagnosis", fontsize=10, fontweight="bold")
    ax2.legend(markerscale=2, fontsize=8, framealpha=0.6)
    ax2.axis("off")

    fig.suptitle("Figure 2: t-SNE projection of functional connectivity features",
                 fontsize=9, y=1.01, color="#555")
    plt.tight_layout()
    import os; os.makedirs("figures", exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {save_path}")


def cluster_summary(
    cluster_labels: np.ndarray,
    y_cls: np.ndarray,
    y_age: np.ndarray,
    best_k: int,
) -> None:
    """Print Table 1: per-cluster ASD%, mean age (proxy for ADOS score)."""
    print(f"\n{'─'*52}")
    print(f"{'Cluster':>8} {'N':>6} {'ASD%':>8} {'Mean age':>10}")
    print(f"{'─'*52}")
    for c in range(best_k):
        mask   = cluster_labels == c
        n      = mask.sum()
        asd_pct = y_cls[mask].mean() * 100
        mean_age = y_age[mask].mean()
        print(f"{c:>8} {n:>6} {asd_pct:>7.1f}% {mean_age:>10.1f}")
    print(f"{'─'*52}\n")


def run_clustering(split, random_seed: int = 42):
    """
    Full clustering pipeline called from main.py.

    Parameters
    ----------
    split : ProcessedSplit from preprocess.py

    Returns
    -------
    cluster_labels_train : np.ndarray shape (N_train,)
    cluster_labels_test  : np.ndarray shape (N_test,)  — assigned by nearest centroid
    best_k               : int
    """
    print("\n── Clustering ─────────────────────────────────────────")
    X_train = split.X_train_pca

    # 1. select k
    k_range   = range(2, 11)
    inertias, sil_scores, best_k = select_k(X_train, k_range, random_seed)
    print(f"  Best k (Silhouette): {best_k}")

    # 2. plots
    plot_selection(k_range, inertias, sil_scores, best_k)
    
    # 3. fit final model
    cluster_labels_train = fit_kmeans(X_train, best_k, random_seed)

    # assign test subjects to nearest centroid (no leakage: centroids from train)
    km_final = KMeans(n_clusters=best_k, random_state=random_seed, n_init=20)
    km_final.fit(X_train)
    cluster_labels_test = km_final.predict(split.X_test_pca)

    # 4. t-SNE on train set
    plot_tsne(X_train, cluster_labels_train, split.y_cls_train, best_k, random_seed)

    # 5. summary table
    cluster_summary(cluster_labels_train, split.y_cls_train,
                    split.y_age_train, best_k)

    return cluster_labels_train, cluster_labels_test, best_k
