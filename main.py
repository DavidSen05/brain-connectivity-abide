import os
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error,
    confusion_matrix, roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synthetic_data import generate_dataset
from preprocess     import preprocess, load_real_data
from cluster        import run_clustering
from svm_model      import run_svm
from mlp_model      import run_mlp
from gcn_model      import run_gcn


def load_data(use_real: bool, data_dir: str) -> list:
    if use_real:
        print("Loading real ABIDE data from", data_dir)
        return load_real_data(data_dir)
    else:
        print("Generating synthetic ABIDE-like dataset...")
        return generate_dataset()


def evaluate_classification(results: dict) -> dict:
    y_true  = results["y_true_cls"]
    y_pred  = results["y_pred_cls"]
    y_proba = results["y_proba_cls"]
    return {
        "Model":     results["model_name"],
        "Accuracy":  round(accuracy_score(y_true, y_pred),            4),
        "Precision": round(precision_score(y_true, y_pred,
                                           zero_division=0),           4),
        "Recall":    round(recall_score(y_true, y_pred,
                                        zero_division=0),              4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_true, y_proba),            4),
    }


def evaluate_regression(results: dict) -> dict:
    y_true = results["y_true_age"]
    y_pred = results["y_pred_age"]
    return {
        "Model": results["model_name"],
        "MAE":   round(mean_absolute_error(y_true, y_pred),            4),
        "RMSE":  round(np.sqrt(mean_squared_error(y_true, y_pred)),    4),
    }


def print_table(rows: list, title: str) -> None:
    if not rows:
        return
    keys  = list(rows[0].keys())
    col_w = {k: max(len(k), max(len(str(r[k])) for r in rows)) for k in keys}
    sep   = "  ".join("-" * col_w[k] for k in keys)
    hdr   = "  ".join(k.ljust(col_w[k]) for k in keys)
    print(f"\n{'─'*len(sep)}\n{title}\n{'─'*len(sep)}")
    print(hdr)
    print(sep)
    for row in rows:
        print("  ".join(str(row[k]).ljust(col_w[k]) for k in keys))
    print('─'*len(sep))


def save_csv(rows: list, path: str) -> None:
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved → {path}")


def plot_roc_curves(all_results: list,
                    save_path: str = "figures/fig3_roc_curves.png"):
    """Figure 3: ROC curves — clean single AUC label per model."""
    fig, ax = plt.subplots(figsize=(6, 5))
    colors  = {"SVM": "#4878CF", "MLP": "#6ACC65", "GCN": "#D65F5F"}

    for res in all_results:
        name    = res["model_name"]
        y_true  = res["y_true_cls"]
        y_proba = res["y_proba_cls"]
        auc     = roc_auc_score(y_true, y_proba)
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        ax.plot(fpr, tpr, color=colors.get(name, "gray"),
                linewidth=1.8, label=f"{name} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance")
    ax.set_title("Figure 3: ROC Curves — ASD Classification",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate",  fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_confusion_matrix(best_result: dict,
                          save_path: str = "figures/fig4_confusion_matrix.png"):
    """Figure 4: Confusion matrix — wider so title doesn't get clipped."""
    name   = best_result["model_name"]
    y_true = best_result["y_true_cls"]
    y_pred = best_result["y_pred_cls"]
    cm     = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred Control", "Pred ASD"], fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True Control", "True ASD"], fontsize=9)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Figure 4: Confusion Matrix — {name}",
                 fontsize=10, fontweight="bold", pad=10)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_age_scatter(all_results: list,
                     save_path: str = "figures/fig5_age_scatter.png"):
    """Figure 5: Predicted vs. true age, colored by diagnosis."""
    n     = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    colors = {1: "#D65F5F", 0: "#4878CF"}

    for ax, res in zip(axes, all_results):
        y_true = res["y_true_age"]
        y_pred = res["y_pred_age"]
        y_cls  = res["y_true_cls"]
        mae    = mean_absolute_error(y_true, y_pred)

        for lbl, name in [(1, "ASD"), (0, "Control")]:
            mask = y_cls == lbl
            ax.scatter(y_true[mask], y_pred[mask],
                       c=colors[lbl], s=14, alpha=0.6,
                       label=name, rasterized=True)

        lims = [min(y_true.min(), y_pred.min()) - 2,
                max(y_true.max(), y_pred.max()) + 2]
        ax.plot(lims, lims, "k--", linewidth=0.8)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Chronological Age", fontsize=9)
        ax.set_ylabel("Predicted Age",     fontsize=9)
        ax.set_title(f"{res['model_name']}  (MAE = {mae:.2f})",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, markerscale=1.5)

    fig.suptitle("Figure 5: Predicted vs. Chronological Age",
                 fontsize=10, y=1.02, color="#555")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real",     action="store_true")
    parser.add_argument("--data_dir", type=str, default="data/roi_timeseries")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--skip_gcn", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  Brain Connectivity Prediction Pipeline")
    print("=" * 60)

    print("\n── Data Loading & Preprocessing ───────────────────────")
    dataset = load_data(use_real=args.real, data_dir=args.data_dir)
    split   = preprocess(dataset, random_seed=args.seed)

    cluster_labels_train, cluster_labels_test, best_k = \
        run_clustering(split, random_seed=args.seed)

    all_results = []

    svm_res = run_svm(split, random_seed=args.seed)
    all_results.append(svm_res)

    mlp_res = run_mlp(split, random_seed=args.seed)
    all_results.append(mlp_res)

    if not args.skip_gcn:
        gcn_res = run_gcn(split, random_seed=args.seed)
        all_results.append(gcn_res)
    else:
        print("\n  [GCN skipped]")

    print("\n── Evaluation ─────────────────────────────────────────")
    cls_rows = [evaluate_classification(r) for r in all_results]
    reg_rows = [evaluate_regression(r)     for r in all_results]

    print_table(cls_rows, "Table 2: ASD Classification Results")
    print_table(reg_rows, "Table 3: Brain Age Regression Results")

    save_csv(cls_rows, "results/classification_results.csv")
    save_csv(reg_rows, "results/regression_results.csv")

    print("\n── Generating Figures ──────────────────────────────────")
    plot_roc_curves(all_results)

    best = max(all_results,
               key=lambda r: roc_auc_score(r["y_true_cls"], r["y_proba_cls"]))
    plot_confusion_matrix(best)
    plot_age_scatter(all_results)

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("  Figures → figures/")
    print("  Results → results/")
    print("=" * 60)


if __name__ == "__main__":
    main()