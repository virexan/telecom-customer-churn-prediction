"""
Customer Churn Predictor
========================
Predicts whether a customer will churn (leave) based on their usage patterns.
Uses multiple ML models and compares their performance.


"""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score
)
from sklearn.pipeline import Pipeline
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. Data Generation (simulates real telecom data)
# ─────────────────────────────────────────────

def generate_churn_dataset(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic telecom churn dataset.
    In a real project, you'd load from CSV or a database.
    """
    np.random.seed(random_state)

    data = {
        "tenure_months":       np.random.randint(1, 72, n_samples),
        "monthly_charges":     np.round(np.random.uniform(20, 120, n_samples), 2),
        "total_charges":       None,  # will compute below
        "num_products":        np.random.randint(1, 5, n_samples),
        "has_internet":        np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        "has_phone":           np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        "contract_type":       np.random.choice(["Month-to-Month", "One Year", "Two Year"],
                                                 n_samples, p=[0.55, 0.25, 0.20]),
        "payment_method":      np.random.choice(
                                    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
                                    n_samples),
        "support_calls":       np.random.poisson(2, n_samples),
        "late_payments":       np.random.poisson(0.5, n_samples),
    }

    df = pd.DataFrame(data)

    # Derived features
    df["total_charges"] = np.round(df["tenure_months"] * df["monthly_charges"] * np.random.uniform(0.9, 1.1, n_samples), 2)
    df["avg_monthly_spend"] = np.round(df["total_charges"] / df["tenure_months"], 2)
    df["charges_per_product"] = (
        df["monthly_charges"] / df["num_products"]
    )

    df["customer_value"] = (
        df["total_charges"] * df["num_products"]
    )

    df["service_usage_score"] = (
        df["has_internet"] +
        df["has_phone"] +
        df["num_products"]
    )

    # Realistic churn label (higher churn for month-to-month, high charges, many support calls)
    churn_prob = (
    0.03
    + 0.35 * (df["contract_type"] == "Month-to-Month")
    + 0.15 * (df["monthly_charges"] > 80)
    + 0.12 * (df["support_calls"] > 3)
    + 0.10 * (df["late_payments"] > 1)
    - 0.10 * (df["tenure_months"] > 36)
    - 0.08 * (df["num_products"] > 2)
).clip(0.02, 0.85)

    df["churn"] = (np.random.rand(n_samples) < churn_prob).astype(int)

    return df


# ─────────────────────────────────────────────
# 2. Exploratory Data Analysis
# ─────────────────────────────────────────────

def run_eda(df: pd.DataFrame, plots_dir: str = "plots") -> None:
    """Generates and saves key EDA plots."""
    os.makedirs(plots_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Customer Churn – Exploratory Data Analysis", fontsize=16, fontweight="bold")

    # Churn distribution
    churn_counts = df["churn"].value_counts()
    axes[0, 0].pie(churn_counts, labels=["Retained", "Churned"],
                   autopct="%1.1f%%", colors=["#4CAF50", "#F44336"], startangle=90)
    axes[0, 0].set_title("Churn Distribution")

    # Monthly charges by churn
    df.groupby("churn")["monthly_charges"].plot(kind="hist", alpha=0.6, bins=20,
                                                 ax=axes[0, 1], legend=True)
    axes[0, 1].set_title("Monthly Charges by Churn")
    axes[0, 1].set_xlabel("Monthly Charges ($)")
    axes[0, 1].legend(["Retained", "Churned"])

    # Tenure by churn
    df.groupby("churn")["tenure_months"].plot(kind="hist", alpha=0.6, bins=20,
                                               ax=axes[0, 2], legend=True)
    axes[0, 2].set_title("Tenure by Churn")
    axes[0, 2].set_xlabel("Tenure (months)")
    axes[0, 2].legend(["Retained", "Churned"])

    # Churn by contract type
    contract_churn = df.groupby("contract_type")["churn"].mean() * 100
    contract_churn.plot(kind="bar", ax=axes[1, 0], color="#2196F3", edgecolor="black")
    axes[1, 0].set_title("Churn Rate by Contract Type (%)")
    axes[1, 0].set_xlabel("")
    axes[1, 0].tick_params(axis="x", rotation=15)

    # Support calls vs churn
    sns.boxplot(data=df, x="churn", y="support_calls", ax=axes[1, 1],
                palette=["#4CAF50", "#F44336"])
    axes[1, 1].set_title("Support Calls vs Churn")
    axes[1, 1].set_xticklabels(["Retained", "Churned"])

    # Correlation heatmap (numeric only)
    num_cols = df.select_dtypes(include=np.number).columns
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                ax=axes[1, 2], cbar=True, linewidths=0.5, annot_kws={"size": 7})
    axes[1, 2].set_title("Feature Correlation Heatmap")

    plt.tight_layout()
    path = os.path.join(plots_dir, "eda_overview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] EDA plot saved → {path}")


# ─────────────────────────────────────────────
# 3. Feature Engineering & Preprocessing
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    """Encodes categoricals, scales numerics, splits data."""
    df = df.copy()

    # Encode categorical columns
    le = LabelEncoder()
    for col in ["contract_type", "payment_method"]:
        df[col] = le.fit_transform(df[col])

    feature_cols = [c for c in df.columns if c != "churn"]
    X = df[feature_cols]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"  Churn rate in train: {y_train.mean():.2%} | test: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test, feature_cols


# ─────────────────────────────────────────────
# 4. Model Training & Evaluation
# ─────────────────────────────────────────────

def build_models() -> dict:
    """Returns pipelines for each model."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier( n_estimators=300, max_depth=10, min_samples_split=5, class_weight="balanced", random_state=42))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,random_state=42))
        ]),
    }


def evaluate_models(models: dict, X_train, X_test, y_train, y_test, plots_dir: str = "plots") -> dict:
    """Trains models, prints metrics, plots ROC curves."""
    results = {}

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="roc_auc")

        results[name] = {
            "pipeline": pipeline,
            "accuracy": acc,
            "roc_auc": auc,
            "cv_auc_mean": cv_scores.mean(),
            "cv_auc_std": cv_scores.std(),
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

        print(f"\n  ── {name} ──")
        print(f"     Accuracy  : {acc:.4f}")
        print(f"     ROC-AUC   : {auc:.4f}")
        print(f"     CV AUC    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – Model Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(plots_dir, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [✓] ROC curve saved → {path}")

    return results


def plot_feature_importance(results: dict, feature_cols: list, plots_dir: str = "plots") -> None:
    """Plots feature importances for the Random Forest model."""
    rf_pipeline = results["Random Forest"]["pipeline"]
    rf_clf = rf_pipeline.named_steps["clf"]
    importances = rf_clf.feature_importances_

    feat_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(feat_df["Feature"], feat_df["Importance"], color="#2196F3", edgecolor="black")
    ax.set_title("Feature Importances – Random Forest")
    ax.set_xlabel("Importance Score")
    ax.grid(True, axis="x", alpha=0.3)

    path = os.path.join(plots_dir, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Feature importance plot saved → {path}")


def plot_confusion_matrix(results: dict, y_test, plots_dir: str = "plots") -> None:
    """Plots confusion matrix for best model."""
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    y_pred = results[best_name]["y_pred"]

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Retained", "Churned"],
                yticklabels=["Retained", "Churned"])
    ax.set_title(f"Confusion Matrix – {best_name}")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

    path = os.path.join(plots_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Confusion matrix saved → {path}")


# ─────────────────────────────────────────────
# 5. Predict on New Customer
# ─────────────────────────────────────────────

def predict_customer(pipeline, sample: dict, feature_cols: list) -> None:
    """Predicts churn probability for a single new customer."""
    sample_df = pd.DataFrame([sample])[feature_cols]
    prob = pipeline.predict_proba(sample_df)[0][1]
    label = "CHURN" if prob > 0.5 else "RETAIN"
    print(f"\n  New Customer Prediction:")
    print(f"    Churn Probability : {prob:.2%}")
    print(f"    Decision          : {label}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("   CUSTOMER CHURN PREDICTOR")
    print("=" * 55)

    BASE_DIR  = os.path.join(os.path.dirname(__file__), "..")
    PLOTS_DIR = os.path.join(BASE_DIR, "plots")
    DATA_DIR  = os.path.join(BASE_DIR, "data")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("\n[1] Generating dataset...")
    df = generate_churn_dataset(n_samples=1500)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Overall churn rate: {df['churn'].mean():.2%}")
    df.to_csv(os.path.join(DATA_DIR, "churn_data.csv"), index=False)
    print("  [✓] Dataset saved → data/churn_data.csv")

    print("\n[2] Running EDA...")
    run_eda(df, plots_dir=PLOTS_DIR)

    print("\n[3] Preprocessing...")
    X_train, X_test, y_train, y_test, feature_cols = preprocess(df)

    print("\n[4] Training & evaluating models...")
    models = build_models()
    results = evaluate_models(models, X_train, X_test, y_train, y_test, plots_dir=PLOTS_DIR)

    print("\n[5] Generating additional plots...")
    plot_feature_importance(results, feature_cols, plots_dir=PLOTS_DIR)
    plot_confusion_matrix(results, y_test, plots_dir=PLOTS_DIR)

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\n[6] Best model: {best_name} (AUC = {results[best_name]['roc_auc']:.4f})")
    print(f"     Classification Report:\n")
    print(classification_report(y_test, results[best_name]["y_pred"],
                                 target_names=["Retained", "Churned"]))

    # Predict on a sample new customer
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(
        results[best_name]["pipeline"],
        os.path.join(MODELS_DIR, "best_churn_model.pkl")
    )

    print("\n[✓] Best model saved -> models/best_churn_model.pkl")

    sample_customer = {
        "tenure_months": 5,
        "monthly_charges": 95.0,
        "total_charges": 475.0,
        "num_products": 1,
        "has_internet": 1,
        "has_phone": 1,
        "contract_type": 0,
        "payment_method": 0,
        "support_calls": 5,
        "late_payments": 2,
        "avg_monthly_spend": 95.0,
        "charges_per_product": 95.0,
        "customer_value": 475.0,
        "service_usage_score": 3
    }
    predict_customer(results[best_name]["pipeline"], sample_customer, feature_cols)

    print("\n" + "=" * 55)
    print("   ALL DONE! Check the /plots folder for visuals.")
    print("=" * 55)


if __name__ == "__main__":
    main()
