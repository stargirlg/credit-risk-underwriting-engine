import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

try:
    from sklearn.frozen import FrozenEstimator
    HAS_FROZEN = True
except ImportError:
    HAS_FROZEN = False

FEATURE_COLUMNS = [
    "cibil_score",
    "monthly_income",
    "existing_emi",
    "loan_amount",
    "tenure_months",
    "credit_utilization",
    "recent_inquiries_6m",
    "dpd_30_plus_count",
    "foir_ratio",
    "loan_to_annual_income",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    monthly_rate = 0.12 / 12
    n = data["tenure_months"].clip(lower=6)
    proposed_emi = (
        data["loan_amount"]
        * monthly_rate
        * ((1 + monthly_rate) ** n)
        / (((1 + monthly_rate) ** n) - 1)
    )
    data["foir_ratio"] = ((data["existing_emi"] + proposed_emi) / data["monthly_income"].clip(lower=1)) * 100.0
    data["loan_to_annual_income"] = data["loan_amount"] / (data["monthly_income"].clip(lower=1) * 12.0)
    return data[FEATURE_COLUMNS]


def generate_synthetic_credit_data(n_samples: int = 12000, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    cibil_score = rng.normal(735, 65, n_samples).clip(520, 880)
    monthly_income = rng.lognormal(mean=10.9, sigma=0.45, size=n_samples).clip(18000, 350000)
    existing_emi = monthly_income * rng.uniform(0.0, 0.45, n_samples)
    loan_amount = monthly_income * rng.uniform(2.0, 14.0, n_samples)
    tenure_months = rng.choice([12, 24, 36, 48, 60], size=n_samples)
    credit_utilization = rng.beta(2.2, 4.5, n_samples) * 100.0
    recent_inquiries_6m = rng.poisson(1.4, n_samples).clip(0, 10)
    dpd_30_plus_count = rng.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.74, 0.14, 0.07, 0.03, 0.02])

    raw_df = pd.DataFrame({
        "cibil_score": cibil_score,
        "monthly_income": monthly_income,
        "existing_emi": existing_emi,
        "loan_amount": loan_amount,
        "tenure_months": tenure_months,
        "credit_utilization": credit_utilization,
        "recent_inquiries_6m": recent_inquiries_6m,
        "dpd_30_plus_count": dpd_30_plus_count,
    })

    X = engineer_features(raw_df)

    log_odds = (
        -2.4
        - 0.018 * (X["cibil_score"] - 720)
        + 0.042 * (X["foir_ratio"] - 40)
        + 0.025 * (X["credit_utilization"] - 35)
        + 0.38 * X["recent_inquiries_6m"]
        + 0.85 * X["dpd_30_plus_count"]
        + 0.45 * (X["loan_to_annual_income"] - 0.6)
    )
    prob_default = 1 / (1 + np.exp(-log_odds))
    y = rng.binomial(1, prob_default)
    return X, y


def train_and_save() -> None:
    os.makedirs("artifacts", exist_ok=True)
    X, y = generate_synthetic_credit_data()

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_cal, y_train, y_cal = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    baseline.fit(X_train, y_train)
    baseline_auc = roc_auc_score(y_test, baseline.predict_proba(X_test)[:, 1])

    monotone_constraints = (-1, -1, 1, 1, 0, 1, 1, 1, 1, 1)
    base_xgb = xgb.XGBClassifier(
        n_estimators=220,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        monotone_constraints=monotone_constraints,
        eval_metric="logloss",
        random_state=42,
    )
    base_xgb.fit(X_train, y_train)

    if HAS_FROZEN:
        calibrated_xgb = CalibratedClassifierCV(FrozenEstimator(base_xgb), method="sigmoid")
    else:
        calibrated_xgb = CalibratedClassifierCV(base_xgb, cv="prefit", method="sigmoid")
    calibrated_xgb.fit(X_cal, y_cal)

    cal_probs = calibrated_xgb.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, cal_probs)
    brier = brier_score_loss(y_test, cal_probs)

    print(f"Baseline Logistic Regression ROC-AUC : {baseline_auc:.4f}")
    print(f"Calibrated Monotonic XGBoost ROC-AUC : {xgb_auc:.4f}")
    print(f"Calibrated Brier Score (lower=better): {brier:.4f}")

    joblib.dump(
        {
            "base_model": base_xgb,
            "calibrated_model": calibrated_xgb,
            "feature_columns": FEATURE_COLUMNS,
            "metrics": {
                "baseline_auc": round(float(baseline_auc), 4),
                "xgboost_auc": round(float(xgb_auc), 4),
                "brier_score": round(float(brier), 4),
            },
        },
        "artifacts/credit_risk_bundle.joblib",
    )
    print("Saved model bundle to artifacts/credit_risk_bundle.joblib")


if __name__ == "__main__":
    train_and_save()
