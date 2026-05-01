"""
src/models/definitions.py — Model Zoo
=======================================
Defines all classifiers used in the comparative study. Hyperparameters
have been tuned for small-to-medium clinical tabular datasets.
"""
from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import RANDOM_STATE

_RS = RANDOM_STATE


def get_base_models() -> dict[str, object]:
    """Return ordered dict of {name: unfitted estimator}."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, C=1.0, solver="lbfgs",
            class_weight="balanced", random_state=_RS,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_split=10,
            class_weight="balanced", random_state=_RS,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None,
            min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=_RS,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1, random_state=_RS,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05,
            max_depth=4, subsample=0.8, random_state=_RS,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05,
            max_depth=5, subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", verbosity=0, random_state=_RS,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.05,
            max_depth=5, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced",
            verbose=-1, random_state=_RS,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=200, learning_rate=0.5, random_state=_RS,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", C=10, gamma="scale",
            probability=True, class_weight="balanced",
            random_state=_RS,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=7, weights="distance",
            metric="minkowski", n_jobs=-1,
        ),
        "MLP Neural Net": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu", solver="adam",
            alpha=0.001, learning_rate_init=0.001,
            max_iter=500, early_stopping=True,
            validation_fraction=0.1, random_state=_RS,
        ),
    }


def get_stacking_model(
    base_models: dict[str, object] | None = None,
) -> StackingClassifier:
    """Stacking ensemble: RF + XGB + LGBM + SVM -> LogReg meta-learner."""
    if base_models is None:
        base_models = get_base_models()

    estimators = [
        ("rf", base_models["Random Forest"]),
        ("xgb", base_models["XGBoost"]),
        ("lgbm", base_models["LightGBM"]),
        ("svm", base_models["SVM (RBF)"]),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            max_iter=2000, C=0.1, random_state=_RS
        ),
        cv=5,
        stack_method="predict_proba",
        n_jobs=1,  # n_jobs=-1 hits joblib memmap bug on Windows
    )


def get_voting_model(
    base_models: dict[str, object] | None = None,
) -> VotingClassifier:
    """Soft-voting ensemble used as a comparison point to stacking."""
    if base_models is None:
        base_models = get_base_models()

    estimators = [
        ("rf", base_models["Random Forest"]),
        ("xgb", base_models["XGBoost"]),
        ("lgbm", base_models["LightGBM"]),
        ("mlp", base_models["MLP Neural Net"]),
    ]
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
