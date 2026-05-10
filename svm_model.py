"""
svm_model.py
------------
Support Vector Machine baseline:
  - SVC (RBF kernel) for ASD classification
  - SVR (RBF kernel) for brain age regression

Hyperparameters C and gamma tuned with 5-fold CV on the training set.
"""

import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from typing import Tuple, Dict, Any


# ── hyperparameter grid (kept small so CV runs in < 5 min on CPU) ─────────────
PARAM_GRID = {
    "svm__C":     [0.1, 1.0, 10.0],
    "svm__gamma": ["scale", "auto"],
}


def _cv_search(estimator, X_train, y_train,
               scoring: str, random_seed: int) -> Tuple[Any, Dict]:
    """Run 5-fold CV grid search; return best estimator + best params."""
    pipe = Pipeline([("svm", estimator)])
    gs = GridSearchCV(
        pipe, PARAM_GRID,
        cv=5, scoring=scoring, n_jobs=-1, refit=True,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_


def train_svm_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_seed: int = 42,
):
    """
    Train SVC with probability calibration on PCA-reduced FC features.

    Returns
    -------
    model      : fitted sklearn Pipeline
    best_params: dict of best C and gamma
    """
    estimator = SVC(kernel="rbf", probability=True,
                    class_weight="balanced",
                    random_state=random_seed)
    model, best_params = _cv_search(
        estimator, X_train, y_train,
        scoring="roc_auc", random_seed=random_seed
    )
    print(f"  SVM classifier best params: {best_params}")
    return model, best_params


def train_svm_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_seed: int = 42,
):
    """
    Train SVR (epsilon-insensitive) for brain age regression.

    Returns
    -------
    model      : fitted sklearn Pipeline
    best_params: dict of best C and gamma
    """
    estimator = SVR(kernel="rbf", epsilon=0.5)
    param_grid_reg = {
        "svm__C":     [0.1, 1.0, 10.0],
        "svm__gamma": ["scale", "auto"],
        "svm__epsilon": [0.1, 0.5],
    }
    pipe = Pipeline([("svm", estimator)])
    from sklearn.model_selection import GridSearchCV
    gs = GridSearchCV(
        pipe, param_grid_reg,
        cv=5, scoring="neg_mean_absolute_error",
        n_jobs=-1, refit=True, verbose=0,
    )
    gs.fit(X_train, y_train)
    print(f"  SVM regressor best params: {gs.best_params_}")
    return gs.best_estimator_, gs.best_params_


def predict_classifier(model, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    y_pred  : hard class predictions (0 or 1)
    y_proba : predicted probability for class 1 (ASD)
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_proba


def predict_regressor(model, X_test: np.ndarray) -> np.ndarray:
    """Returns predicted age for each test subject."""
    return model.predict(X_test)


def run_svm(split, random_seed: int = 42) -> Dict:
    """
    Full SVM training + prediction called from main.py.

    Returns a results dict consumed by evaluate.py.
    """
    print("\n── SVM ────────────────────────────────────────────────")
    X_tr, X_te = split.X_train_pca, split.X_test_pca

    # — classification
    print("  Training SVM classifier...")
    clf, clf_params = train_svm_classifier(X_tr, split.y_cls_train, random_seed)
    y_pred_cls, y_proba_cls = predict_classifier(clf, X_te)

    # — regression
    print("  Training SVM regressor...")
    reg, reg_params = train_svm_regressor(X_tr, split.y_age_train, random_seed)
    y_pred_age = predict_regressor(reg, X_te)

    return {
        "model_name":    "SVM",
        "y_pred_cls":    y_pred_cls,
        "y_proba_cls":   y_proba_cls,
        "y_true_cls":    split.y_cls_test,
        "y_pred_age":    y_pred_age,
        "y_true_age":    split.y_age_test,
        "clf_params":    clf_params,
        "reg_params":    reg_params,
    }
