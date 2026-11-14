import matplotlib
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel, Field

from rules_engine import evaluate_deterministic_rules
from train_model import engineer_features, generate_synthetic_credit_data

bundle = joblib.load("artifacts/credit_risk_bundle.joblib")
base_model = bundle["base_model"]
calibrated_model = bundle["calibrated_model"]
feature_columns = bundle["feature_columns"]

try:
    explainer = shap.TreeExplainer(base_model)
except Exception:
    explainer = None

app = FastAPI(
    title="Credit Risk Underwriting & Loan Triage Engine",
    description="API-first ML credit scoring (XGBoost + SHAP) paired with deterministic financial rules.",
    version="1.0.0",
)


class LoanApplication(BaseModel):
    applicant_id: str = Field(default="APP-1001")
    age: int = Field(default=31, ge=18, le=75)
    state: str = Field(default="Maharashtra")
    scheme_code: str = Field(default="SALARIED_PRIME")
    docs_uploaded: int = Field(default=12, ge=0, le=12)
    cibil_score: int = Field(default=765, ge=300, le=900)
    monthly_income: float = Field(default=85000.0, gt=0)
    existing_emi: float = Field(default=12000.0, ge=0)
    loan_amount: float = Field(default=400000.0, gt=0)
    tenure_months: int = Field(default=36, ge=6, le=84)
    credit_utilization: float = Field(default=24.0, ge=0, le=100)
    recent_inquiries_6m: int = Field(default=1, ge=0, le=20)
    dpd_30_plus_count: int = Field(default=0, ge=0, le=15)


def compute_shap_contributions(X_feat: pd.DataFrame) -> np.ndarray:
    if explainer is not None:
        shap_raw = explainer.shap_values(X_feat)
        shap_row = shap_raw[1][0] if isinstance(shap_raw, list) else shap_raw[0]
        return np.asarray(shap_row).ravel()
    contribs = base_model.get_booster().predict(xgb.DMatrix(X_feat), pred_contribs=True)
    return np.asarray(contribs[0][:-1]).ravel()


def score_single_application(payload: dict) -> tuple[dict, list[dict]]:
    raw_df = pd.DataFrame([payload])
    X_feat = engineer_features(raw_df)
    calibrated_pd = float(calibrated_model.predict_proba(X_feat)[0, 1])
    shap_row = compute_shap_contributions(X_feat)

    reason_codes = sorted(
        [
            {"feature": col, "shap_impact": round(float(val), 4), "direction": "Increases Risk" if val > 0 else "Lowers Risk"}
            for col, val in zip(feature_columns, shap_row)
        ],
        key=lambda x: abs(x["shap_impact"]),
        reverse=True,
    )

    result = evaluate_deterministic_rules(payload, calibrated_pd)
    result["applicant_id"] = payload.get("applicant_id", "APP-1001")
    result["top_shap_reason_codes"] = reason_codes[:4]
    return result, reason_codes


@app.get("/")
def health_check():
    return {
        "status": "online",
        "model_metrics": bundle["metrics"],
        "endpoints": {"swagger_api_docs": "/docs", "gradio_ml_ui": "/ui"},
    }


@app.post("/underwrite")
def underwrite_loan(application: LoanApplication):
    result, _ = score_single_application(application.model_dump())
    return result


@app.post("/triage-batch")
def triage_batch(applications: list[LoanApplication]):
    records = [a.model_dump() for a in applications]
    raw_df = pd.DataFrame(records)
    X_feat = engineer_features(raw_df)
    pds = calibrated_model.predict_proba(X_feat)[:, 1]

    outcomes = [evaluate_deterministic_rules(rec, float(pd_val)) for rec, pd_val in zip(records, pds)]
    summary = {
        "total_applications": len(outcomes),
        "auto_approved_stp": sum(1 for o in outcomes if o["decision"] == "AUTO_APPROVED_STP"),
        "routed_to_analyst_workbench": sum(1 for o in outcomes if o["decision"] == "EXCEPTION_REVIEW"),
        "auto_declined": sum(1 for o in outcomes if o["decision"] == "AUTO_DECLINE"),
    }
    summary["workbench_overload_reduction_pct"] = round(
        ((summary["auto_approved_stp"] + summary["auto_declined"]) / max(len(outcomes), 1)) * 100.0, 1
    )
    return {"batch_summary": summary, "sample_exceptions": [o for o in outcomes if o["decision"] == "EXCEPTION_REVIEW"][:10]}


def run_gradio_single(
    age, state, scheme_code, docs_uploaded, cibil_score,
    monthly_income, existing_emi, loan_amount, tenure_months,
    credit_utilization, recent_inquiries_6m, dpd_30_plus_count
):
    payload = {
        "applicant_id": "DEMO-LIVE-01",
        "age": age,
        "state": state,
        "scheme_code": scheme_code,
        "docs_uploaded": docs_uploaded,
        "cibil_score": cibil_score,
        "monthly_income": monthly_income,
        "existing_emi": existing_emi,
        "loan_amount": loan_amount,
        "tenure_months": tenure_months,
        "credit_utilization": credit_utilization,
        "recent_inquiries_6m": recent_inquiries_6m,
        "dpd_30_plus_count": dpd_30_plus_count,
    }
    result, reason_codes = score_single_application(payload)

    top_reasons = reason_codes[:6][::-1]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = ["#d9534f" if r["shap_impact"] > 0 else "#2e8b57" for r in top_reasons]
    ax.barh([r["feature"] for r in top_reasons], [r["shap_impact"] for r in top_reasons], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title("SHAP Feature Impact on Default Risk (Log-Odds)")
    ax.set_xlabel("Red = Increases Default Risk | Green = Lowers Default Risk")
    plt.tight_layout()

    headline = (
        f"### Decision: `{result['decision']}` -> Queue: `{result['routing_queue']}`\n"
        f"- **Calibrated Default Probability (PD):** `{result['calibrated_default_probability']:.2%}`\n"
        f"- **Checklist Gate:** `{result['checklist_status']}` | **FOIR:** `{result['financial_math']['foir_pct']}%` | "
        f"**Upfront Charges:** `Rs. {result['financial_math']['total_upfront_charges']:,.2f}`"
    )
    return headline, result, fig


def run_2000_case_simulation():
    X_sim, _ = generate_synthetic_credit_data(n_samples=2000, seed=99)
    rng = np.random.default_rng(99)
    docs = rng.choice([12, 11, 10], size=2000, p=[0.88, 0.09, 0.03])
    ages = rng.integers(22, 58, size=2000)
    states = rng.choice(["Maharashtra", "Karnataka", "Delhi"], size=2000)

    pds = calibrated_model.predict_proba(X_sim)[:, 1]
    rows = []
    for i in range(2000):
        rec = {
            "applicant_id": f"BATCH-{2000 + i}",
            "age": int(ages[i]),
            "state": str(states[i]),
            "scheme_code": "STANDARD",
            "docs_uploaded": int(docs[i]),
            "cibil_score": int(X_sim.iloc[i]["cibil_score"]),
            "monthly_income": float(X_sim.iloc[i]["monthly_income"]),
            "existing_emi": float(X_sim.iloc[i]["existing_emi"]),
            "loan_amount": float(X_sim.iloc[i]["loan_amount"]),
            "tenure_months": int(X_sim.iloc[i]["tenure_months"]),
        }
        out = evaluate_deterministic_rules(rec, float(pds[i]))
        rows.append({
            "Applicant ID": rec["applicant_id"],
            "Decision": out["decision"],
            "Queue": out["routing_queue"],
            "Docs": out["checklist_status"],
            "CIBIL": rec["cibil_score"],
            "PD (%)": round(out["calibrated_default_probability"] * 100, 2),
            "FOIR (%)": out["financial_math"]["foir_pct"],
            "Primary Flag": out["flagged_exceptions"][0] if out["flagged_exceptions"] else "Clean STP",
        })

    df_out = pd.DataFrame(rows)
    stp = int((df_out["Decision"] == "AUTO_APPROVED_STP").sum())
    exc = int((df_out["Decision"] == "EXCEPTION_REVIEW").sum())
    dec = int((df_out["Decision"] == "AUTO_DECLINE").sum())
    cleared_pct = ((stp + dec) / 2000) * 100

    summary_md = (
        f"### 2,000-Case 3-Day TAT Batch Simulation Results\n"
        f"- **Auto-Approved (Straight-Through Processing):** `{stp}` cases\n"
        f"- **Routed to Analyst Workbench (Exceptions Only):** `{exc}` cases\n"
        f"- **Auto-Declined (High Risk Ceiling):** `{dec}` cases\n"
        f"- **Analyst Workbench Overload Reduction:** **`{cleared_pct:.1f}%`** of files resolved automatically"
    )
    exceptions_only = df_out[df_out["Decision"] == "EXCEPTION_REVIEW"].head(200)
    return summary_md, exceptions_only


with gr.Blocks(title="Credit Risk Underwriting & Triage Engine") as demo:
    gr.Markdown("# Credit Risk Underwriting & Loan Triage Workbench")
    with gr.Tabs():
        with gr.Tab("1. Single Applicant Underwriter"):
            with gr.Row():
                with gr.Column():
                    age = gr.Slider(18, 70, value=31, step=1, label="Applicant Age")
                    state = gr.Dropdown(["Maharashtra", "Karnataka", "Delhi"], value="Maharashtra", label="State (Stamp Duty)")
                    scheme = gr.Dropdown(["STANDARD", "FESTIVE_ZERO_PF", "SALARIED_PRIME"], value="SALARIED_PRIME", label="Scheme")
                    docs = gr.Slider(0, 12, value=12, step=1, label="Documents Uploaded (12/12 Gate)")
                    cibil = gr.Slider(500, 880, value=765, step=5, label="CIBIL Score")
                    income = gr.Number(value=85000, label="Net Monthly Income (INR)")
                    emi = gr.Number(value=12000, label="Existing Monthly EMIs (INR)")
                    loan_amt = gr.Number(value=400000, label="Requested Loan Amount (INR)")
                    tenure = gr.Dropdown([12, 24, 36, 48, 60], value=36, label="Tenure (Months)")
                    util = gr.Slider(0, 100, value=24, step=1, label="Credit Utilization (%)")
                    inq = gr.Slider(0, 10, value=1, step=1, label="Recent Inquiries (6M)")
                    dpd = gr.Slider(0, 5, value=0, step=1, label="30+ DPD Count")
                    run_btn = gr.Button("Run Underwriting Engine", variant="primary")
                with gr.Column():
                    decision_md = gr.Markdown()
                    shap_plot = gr.Plot(label="SHAP Explainability")
                    json_out = gr.JSON(label="Audit & Commercials Payload")

            run_btn.click(
                run_gradio_single,
                inputs=[age, state, scheme, docs, cibil, income, emi, loan_amt, tenure, util, inq, dpd],
                outputs=[decision_md, json_out, shap_plot],
            )

        with gr.Tab("2. 2,000-Case Batch Triage"):
            gr.Markdown("Simulate a 2,000-application lending surge to measure Straight-Through Processing vs. Exception Queue routing.")
            batch_btn = gr.Button("Simulate 2,000-Case Batch Triage", variant="primary")
            batch_summary_md = gr.Markdown()
            exception_table = gr.Dataframe(label="Sample Flagged Exceptions Routed to Analysts")
            batch_btn.click(run_2000_case_simulation, outputs=[batch_summary_md, exception_table])

app = gr.mount_gradio_app(app, demo, path="/ui")
