import json
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).parent / "rules_config.json"


def load_rules_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_emi(loan_amount: float, annual_rate_pct: float, tenure_months: int) -> float:
    monthly_rate = (annual_rate_pct / 100.0) / 12.0
    if monthly_rate == 0:
        return round(loan_amount / tenure_months, 2)
    emi = (
        loan_amount
        * monthly_rate
        * ((1 + monthly_rate) ** tenure_months)
        / (((1 + monthly_rate) ** tenure_months) - 1)
    )
    return round(emi, 2)


def evaluate_deterministic_rules(applicant: dict[str, Any], calibrated_pd: float) -> dict[str, Any]:
    cfg = load_rules_config()
    exceptions: list[str] = []

    docs_uploaded = int(applicant["docs_uploaded"])
    req_docs = int(cfg["required_documents"])
    if docs_uploaded < req_docs:
        missing = req_docs - docs_uploaded
        exceptions.append(
            f"Incomplete KYC Packet ({docs_uploaded}/{req_docs} docs) - trigger single-file re-upload for {missing} missing doc(s)."
        )

    age = int(applicant["age"])
    if age < cfg["min_applicant_age"] or age > cfg["max_applicant_age"]:
        exceptions.append(f"Age {age} outside RBI policy window ({cfg['min_applicant_age']}-{cfg['max_applicant_age']}).")

    cibil = int(applicant["cibil_score"])
    if cibil < cfg["min_cibil_cutoff"]:
        exceptions.append(f"CIBIL score {cibil} below minimum policy cutoff ({cfg['min_cibil_cutoff']}).")

    scheme_code = applicant.get("scheme_code", "STANDARD").upper()
    scheme = cfg["schemes"].get(scheme_code, cfg["schemes"]["STANDARD"])
    effective_rate = round(scheme["base_interest_rate"] - scheme["discount_pct"], 2)

    loan_amount = float(applicant["loan_amount"])
    tenure_months = int(applicant["tenure_months"])
    monthly_income = float(applicant["monthly_income"])
    existing_emi = float(applicant["existing_emi"])

    proposed_emi = calculate_emi(loan_amount, effective_rate, tenure_months)
    total_emi = round(existing_emi + proposed_emi, 2)
    foir_pct = round((total_emi / max(monthly_income, 1.0)) * 100.0, 2)

    if foir_pct > cfg["max_foir_percent"]:
        exceptions.append(f"FOIR {foir_pct}% exceeds max debt-to-income cap ({cfg['max_foir_percent']}%).")

    state = applicant.get("state", "DEFAULT")
    stamp_rate = cfg["state_stamp_duty_rate"].get(state, cfg["state_stamp_duty_rate"]["DEFAULT"])
    stamp_duty = round(loan_amount * stamp_rate, 2)
    processing_fee = round(loan_amount * scheme["processing_fee_pct"], 2)
    agreement_fee = float(cfg["agreement_fee_flat"])
    total_upfront_charges = round(stamp_duty + processing_fee + agreement_fee, 2)

    auto_max = cfg["pd_thresholds"]["auto_approve_max"]
    review_max = cfg["pd_thresholds"]["manual_review_max"]

    if calibrated_pd > review_max:
        decision = "AUTO_DECLINE"
        queue = "DECLINED_ARCHIVE"
        exceptions.append(f"Calibrated Default Probability ({calibrated_pd:.1%}) exceeds maximum risk ceiling ({review_max:.0%}).")
    elif exceptions or calibrated_pd > auto_max:
        decision = "EXCEPTION_REVIEW"
        queue = "ANALYST_WORKBENCH"
        if calibrated_pd > auto_max and not exceptions:
            exceptions.append(
                f"Borderline ML Risk Score ({calibrated_pd:.1%} PD) - routed to credit officer with SHAP reason codes."
            )
    else:
        decision = "AUTO_APPROVED_STP"
        queue = "STRAIGHT_THROUGH_DISBURSAL"

    return {
        "decision": decision,
        "routing_queue": queue,
        "checklist_status": f"{docs_uploaded}/{req_docs}",
        "calibrated_default_probability": round(calibrated_pd, 4),
        "financial_math": {
            "scheme_applied": scheme_code,
            "effective_interest_rate_pct": effective_rate,
            "proposed_emi": proposed_emi,
            "total_monthly_emi": total_emi,
            "foir_pct": foir_pct,
            "state": state,
            "stamp_duty": stamp_duty,
            "processing_fee": processing_fee,
            "agreement_fee": agreement_fee,
            "total_upfront_charges": total_upfront_charges,
        },
        "flagged_exceptions": exceptions,
    }
