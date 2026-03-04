import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, roc_auc_score,
                             roc_curve, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

FEATURES = [
    "device", "load_time_s", "hour", "is_weekend",
    "is_new_user", "price_range", "n_pages_viewed",
    "has_coupon", "cart_value", "utm_source"
]
TARGET = "converted"

CATEGORICAL = ["device", "price_range", "utm_source"]


def preprocess(df: pd.DataFrame):
    df = df[FEATURES + [TARGET]].copy()
    le = LabelEncoder()
    for col in CATEGORICAL:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def train_all_models(df: pd.DataFrame):
    data = preprocess(df)
    X, y = data[FEATURES], data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Handle class imbalance with SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1
        ),
    }

    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        model.fit(X_res, y_res)
        cv_auc = cross_val_score(model, X, y, cv=skf, scoring="roc_auc").mean()
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "model":    model,
            "cv_auc":   round(cv_auc, 4),
            "test_auc": round(roc_auc_score(y_test, y_proba), 4),
            "report":   classification_report(y_test, y_pred, output_dict=True),
            "y_test":   y_test,
            "y_proba":  y_proba,
            "y_pred":   y_pred,
        }
        print(f"{name:25s} | CV AUC: {cv_auc:.4f} | Test AUC: {results[name]['test_auc']:.4f}")

    return results, X_test, y_test


def plot_roc_curves(results, save_path=None):
    plt.figure(figsize=(9, 6), facecolor="#0e0e10")
    ax = plt.gca(); ax.set_facecolor("#161618")
    colors = ["#ff6b35","#ffa552","#4caf8a","#3d7fff"]
    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(res["y_test"], res["y_proba"])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC={res['test_auc']:.3f})")
    ax.plot([0,1],[0,1],"--", color="#555")
    ax.set_xlabel("FPR", color="#888"); ax.set_ylabel("TPR", color="#888")
    ax.set_title("ROC Curves — All Models", color="#f0ede8", fontsize=13)
    ax.legend(facecolor="#1a1a1a", labelcolor="#ccc")
    ax.tick_params(colors="#666")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()


def shap_analysis(model, X_test, save_path=None):
    """Run SHAP for XGBoost or tree models."""
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_test)
    plt.figure(facecolor="#0e0e10")
    shap.summary_plot(shap_vals, X_test, plot_type="bar", show=False,
                      color="#ff6b35")
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def save_best_model(results, path="outputs/models/best_model.pkl"):
    best = max(results.items(), key=lambda x: x[1]["test_auc"])
    joblib.dump(best[1]["model"], path)
    print(f"Saved best model: {best[0]} (AUC={best[1]['test_auc']})")
    return best