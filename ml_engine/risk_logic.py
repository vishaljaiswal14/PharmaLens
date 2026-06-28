"""Shared risk logic and feature preparation utilities.

This module centralizes feature engineering rules used by preprocessing,
real-time inference, and batch scoring so risk behavior stays consistent.
"""

from __future__ import annotations

import pandas as pd

from core.config import (
    ELDERLY_THRESHOLD,
    FEATURE_COLS,
    POLYPHARMACY_THRESHOLD,
    RISK_THRESHOLDS,
)
from core.pipeline_logging import get_logger


logger = get_logger(__name__)


FEATURE_LABELS = {
    "age": "Age",
    "sex_code": "Sex",
    "drug_count": "Drug count",
    "unique_drug_count": "Unique drug count",
    "drug_repeat_flag": "Repeated drug entries",
    "reaction_count": "Reaction count",
    "polypharmacy": "Polypharmacy",
    "elderly": "Elderly patient",
    "risk_score": "Composite risk score",
}


def normalize_sex_code(sex: str) -> int:
    """Convert a sex value to the numeric encoding used by the models."""
    normalized = str(sex).upper().strip()
    if normalized not in {"F", "M"}:
        raise ValueError("Sex must be either 'F' or 'M'.")
    return 0 if normalized == "F" else 1


def risk_level_from_prob(prob: float) -> str:
    """Convert probability to the dashboard risk level category."""
    p = float(prob)
    if p < RISK_THRESHOLDS["low"]:
        return "Low"
    if p < RISK_THRESHOLDS["medium"]:
        return "Medium"
    return "High"


def action_from_risk(risk: str) -> dict[str, str]:
    """Map a risk level to an operational decision-support action."""
    normalized = str(risk).strip().lower()
    if normalized == "high":
        return {
            "action": "Immediate attention",
            "priority": "High",
            "rationale": "Prioritize review, validate case details, and consider escalation for safety assessment.",
        }
    if normalized == "medium":
        return {
            "action": "Review",
            "priority": "Medium",
            "rationale": "Review contributing factors and monitor for additional seriousness indicators.",
        }
    return {
        "action": "Monitor",
        "priority": "Low",
        "rationale": "Continue routine monitoring unless new risk factors appear.",
    }


def confidence_from_probability(prob: float, threshold: float) -> dict[str, object]:
    """Score prediction confidence using threshold distance and probability certainty."""
    p = min(max(float(prob), 0.0), 1.0)
    t = min(max(float(threshold), 0.0), 1.0)
    threshold_distance = abs(p - t)
    certainty = abs(p - 0.5) * 2
    confidence_score = min(1.0, (0.55 * min(threshold_distance / 0.35, 1.0)) + (0.45 * certainty))
    if confidence_score >= 0.70:
        level = "High"
    elif confidence_score >= 0.40:
        level = "Moderate"
    else:
        level = "Low"
    return {
        "level": level,
        "score": round(confidence_score * 100, 1),
        "threshold_distance": round(threshold_distance, 3),
        "certainty": round(certainty, 3),
    }


def probability_to_risk(prob_series: pd.Series) -> pd.Series:
    """Map prediction probabilities to consistent categorical risk buckets."""
    return prob_series.apply(risk_level_from_prob)


def build_patient_profile(age: float, drug_count: float) -> dict[str, str]:
    """Build patient profile summary for the UI."""
    age_group = "Elderly" if age >= ELDERLY_THRESHOLD else ("Adult" if age >= 40 else "Young")
    poly = "Yes" if drug_count >= POLYPHARMACY_THRESHOLD else "No"
    burden = "High" if drug_count >= POLYPHARMACY_THRESHOLD else "Normal"
    return {
        "age_group": age_group,
        "polypharmacy": poly,
        "drug_burden": burden,
    }


def build_ai_interpretation(age: float, drug_count: float) -> tuple[str, list[str]]:
    """Build AI interpretation narrative."""
    parts: list[str] = []
    drivers: list[str] = []

    if drug_count >= POLYPHARMACY_THRESHOLD:
        parts.append("high drug burden increases risk")
        drivers.append("drug_count")

    if age >= ELDERLY_THRESHOLD:
        parts.append("age-related vulnerability in elderly patients raises adverse-event risk")
        drivers.append("age")

    if not parts:
        parts.append("no strong risk flags were detected from age and drug burden")

    sentence = "Risk is elevated due to " + " and ".join(parts) + "."
    if not drivers:
        sentence = "Risk appears lower because " + parts[0] + "."

    return sentence, drivers


def _feature_importance_lookup(importance_df: pd.DataFrame | None) -> dict[str, float]:
    """Build a normalized feature importance lookup from available explainability output."""
    if importance_df is None or importance_df.empty or "feature" not in importance_df.columns:
        return {}
    score_col = "importance"
    scores = importance_df.set_index("feature")[score_col].abs()
    total = float(scores.sum())
    if total <= 0:
        return {feature: 0.0 for feature in scores.index}
    return (scores / total).to_dict()


def explain_prediction(features: pd.DataFrame, risk: str, importance_df: pd.DataFrame | None = None, top_n: int = 4) -> dict[str, object]:
    """Generate a case-level explanation from feature values and model importance."""
    if features.empty:
        return {"summary": "No feature values were available for this prediction.", "drivers": []}

    row = features.iloc[0].to_dict()
    importance = _feature_importance_lookup(importance_df)
    candidates: list[dict[str, object]] = []

    rules = {
        "drug_count": float(row.get("drug_count", 0)) >= POLYPHARMACY_THRESHOLD,
        "unique_drug_count": float(row.get("unique_drug_count", 0)) >= POLYPHARMACY_THRESHOLD,
        "reaction_count": float(row.get("reaction_count", 0)) >= 3,
        "polypharmacy": int(row.get("polypharmacy", 0)) == 1,
        "elderly": int(row.get("elderly", 0)) == 1,
        "drug_repeat_flag": int(row.get("drug_repeat_flag", 0)) == 1,
        "risk_score": float(row.get("risk_score", 0)) >= 50,
        "age": float(row.get("age", 0)) >= ELDERLY_THRESHOLD,
    }

    for feature, triggered in rules.items():
        if feature not in row or not triggered:
            continue
        value = row[feature]
        model_weight = float(importance.get(feature, 0.0))
        intensity = _feature_intensity(feature, value)
        rule_weight = 1.0 if feature in {"polypharmacy", "elderly", "risk_score"} else 0.6
        candidates.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "value": value,
                "importance": round(model_weight, 4),
                "feature_intensity": round(intensity, 3),
                "score": (model_weight + 0.05) * intensity + rule_weight,
                "message": _driver_message(feature, value),
            }
        )

    if not candidates and importance:
        for feature, model_weight in sorted(importance.items(), key=lambda item: item[1], reverse=True)[:top_n]:
            if feature in row:
                intensity = _feature_intensity(feature, row[feature])
                candidates.append(
                    {
                        "feature": feature,
                        "label": FEATURE_LABELS.get(feature, feature),
                        "value": row[feature],
                        "importance": round(float(model_weight), 4),
                        "feature_intensity": round(intensity, 3),
                        "score": float(model_weight) * max(intensity, 0.1),
                        "message": _driver_message(feature, row[feature]),
                    }
                )

    drivers = sorted(candidates, key=lambda item: float(item["score"]), reverse=True)[:top_n]
    total_score = sum(max(float(item["score"]), 0.0) for item in drivers)
    for item in drivers:
        impact = (max(float(item["score"]), 0.0) / total_score * 100) if total_score > 0 else 0.0
        item["impact_pct"] = round(impact, 1)
        item["quantified_message"] = f"{item['message']} ({item['impact_pct']:.1f}% of explanation weight)"
    if drivers:
        summary = f"This case is {risk.lower()} risk because " + "; ".join(str(item["quantified_message"]) for item in drivers) + "."
    else:
        summary = f"This case is {risk.lower()} risk with no strong rule-based risk flags from the available features."

    return {"summary": summary, "drivers": drivers}


def _feature_intensity(feature: str, value: object) -> float:
    """Scale a feature value to a 0-1 risk intensity for explanation weighting."""
    numeric = float(value)
    if feature == "age":
        return min(max((numeric - 40) / 50, 0.0), 1.0)
    if feature == "drug_count":
        return min(max(numeric / 15, 0.0), 1.0)
    if feature == "unique_drug_count":
        return min(max(numeric / 12, 0.0), 1.0)
    if feature == "reaction_count":
        return min(max(numeric / 10, 0.0), 1.0)
    if feature == "risk_score":
        return min(max(numeric / 80, 0.0), 1.0)
    if feature in {"polypharmacy", "elderly", "drug_repeat_flag"}:
        return 1.0 if int(numeric) == 1 else 0.0
    return min(max(numeric, 0.0), 1.0)


def _driver_message(feature: str, value: object) -> str:
    """Convert one feature value into a clinician-readable driver message."""
    if feature == "drug_count":
        return f"drug count is high ({float(value):.0f} drugs)"
    if feature == "unique_drug_count":
        return f"unique medication burden is high ({float(value):.0f} unique drugs)"
    if feature == "reaction_count":
        return f"reaction count is elevated ({float(value):.0f} reactions)"
    if feature == "polypharmacy":
        return "polypharmacy is detected"
    if feature == "elderly":
        return "the patient is in an elderly age band"
    if feature == "drug_repeat_flag":
        return "the case contains repeated drug entries"
    if feature == "risk_score":
        return f"composite risk score is elevated ({float(value):.1f})"
    if feature == "age":
        return f"age is elevated ({float(value):.0f} years)"
    return f"{FEATURE_LABELS.get(feature, feature)} is {value}"


def _composite_risk_score(age: float, drug_count: float, unique_drug_count: float, reaction_count: float, drug_repeat_flag: int) -> float:
    """Build the same leakage-safe composite risk score used in preprocessing."""
    polypharmacy = int(float(drug_count) >= POLYPHARMACY_THRESHOLD)
    age_component = (min(max(float(age), 0.0), 100.0) / 100) * 25
    drug_component = (min(max(float(drug_count), 0.0), 20.0) / 20) * 30
    unique_drug_component = (min(max(float(unique_drug_count), 0.0), 20.0) / 20) * 15
    reaction_component = (min(max(float(reaction_count), 0.0), 10.0) / 10) * 20
    flag_component = (polypharmacy * 7) + (int(drug_repeat_flag) * 3)
    return round(age_component + drug_component + unique_drug_component + reaction_component + flag_component, 2)


def build_feature_frame(
    *,
    age: float,
    sex: str,
    drug_count: float,
    reaction_count: float = 0,
    unique_drug_count: float | None = None,
    drug_repeat_flag: int | None = None,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Create a single-row feature frame with all derived model inputs."""
    cols = feature_cols or FEATURE_COLS
    unique_drug_count = float(drug_count) if unique_drug_count is None else float(unique_drug_count)
    drug_repeat_flag = int(float(drug_count) > unique_drug_count) if drug_repeat_flag is None else int(drug_repeat_flag)
    risk_score = _composite_risk_score(float(age), float(drug_count), unique_drug_count, float(reaction_count), drug_repeat_flag)
    features = pd.DataFrame(
        [
            {
                "age": float(age),
                "sex_code": normalize_sex_code(sex),
                "drug_count": float(drug_count),
                "unique_drug_count": unique_drug_count,
                "drug_repeat_flag": drug_repeat_flag,
                "reaction_count": float(reaction_count),
                "polypharmacy": int(float(drug_count) >= POLYPHARMACY_THRESHOLD),
                "elderly": int(float(age) >= ELDERLY_THRESHOLD),
                "risk_score": risk_score,
            }
        ]
    )
    logger.info("Prediction feature frame built with columns=%s", list(features[cols].columns))
    return features[cols]


def prepare_external_features(upload_df: pd.DataFrame, required_features: list[str]) -> pd.DataFrame:
    """Prepare uploaded dataset into the feature schema required by the model."""
    ext = upload_df.copy()

    missing_core = [col for col in ("age", "drug_count") if col not in ext.columns]
    if missing_core:
        raise ValueError(f"Uploaded dataset is missing required columns: {missing_core}")

    if "sex_code" not in ext.columns:
        if "sex" not in ext.columns:
            raise ValueError("Provide either 'sex_code' or 'sex' column in uploaded dataset.")
        ext["sex"] = ext["sex"].astype(str).str.upper().str.strip()
        ext = ext[ext["sex"].isin(["F", "M"])].copy()
        ext["sex_code"] = ext["sex"].map({"F": 0, "M": 1})

    ext["age"] = pd.to_numeric(ext["age"], errors="coerce")
    ext["drug_count"] = pd.to_numeric(ext["drug_count"], errors="coerce")
    ext["sex_code"] = pd.to_numeric(ext["sex_code"], errors="coerce")
    if "reaction_count" not in ext.columns:
        ext["reaction_count"] = 0
    if "unique_drug_count" not in ext.columns:
        ext["unique_drug_count"] = ext["drug_count"]
    ext["reaction_count"] = pd.to_numeric(ext["reaction_count"], errors="coerce")
    ext["unique_drug_count"] = pd.to_numeric(ext["unique_drug_count"], errors="coerce")
    if "drug_repeat_flag" not in ext.columns:
        ext["drug_repeat_flag"] = (ext["drug_count"] > ext["unique_drug_count"]).astype(int)
    ext["drug_repeat_flag"] = pd.to_numeric(ext["drug_repeat_flag"], errors="coerce")

    if "polypharmacy" not in ext.columns:
        ext["polypharmacy"] = (ext["drug_count"] >= POLYPHARMACY_THRESHOLD).astype(int)
    if "elderly" not in ext.columns:
        ext["elderly"] = (ext["age"] >= ELDERLY_THRESHOLD).astype(int)

    ext["polypharmacy"] = pd.to_numeric(ext["polypharmacy"], errors="coerce")
    ext["elderly"] = pd.to_numeric(ext["elderly"], errors="coerce")
    if "risk_score" not in ext.columns:
        ext["risk_score"] = [
            _composite_risk_score(age, drug_count, unique_drug_count, reaction_count, repeat_flag)
            for age, drug_count, unique_drug_count, reaction_count, repeat_flag in zip(
                ext["age"],
                ext["drug_count"],
                ext["unique_drug_count"],
                ext["reaction_count"],
                ext["drug_repeat_flag"],
            )
        ]
    ext["risk_score"] = pd.to_numeric(ext["risk_score"], errors="coerce")

    ext = ext[
        (ext["age"] > 0)
        & (ext["age"] < 120)
        & (ext["drug_count"] >= 0)
        & (ext["unique_drug_count"] >= 0)
        & (ext["reaction_count"] >= 0)
    ].copy()
    ext = ext.dropna(subset=required_features).copy()
    ext["sex_code"] = ext["sex_code"].astype(int)
    ext["polypharmacy"] = ext["polypharmacy"].astype(int)
    ext["elderly"] = ext["elderly"].astype(int)
    ext["drug_repeat_flag"] = ext["drug_repeat_flag"].astype(int)
    return ext


def create_download_template() -> pd.DataFrame:
    """Create a sample CSV template for batch prediction upload."""
    return pd.DataFrame(
        {
            "age": [34, 71],
            "sex": ["F", "M"],
            "drug_count": [2, 7],
            "unique_drug_count": [2, 6],
            "reaction_count": [1, 4],
            "polypharmacy": [0, 1],
            "elderly": [0, 1],
        }
    )


def validate_upload_data(df: pd.DataFrame) -> tuple[bool, str]:
    """Validate uploaded batch-scoring data before inference."""
    required_columns = ["age", "drug_count"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"

    if "sex" not in df.columns and "sex_code" not in df.columns:
        return False, "Must include either 'sex' or 'sex_code' column."

    age = pd.to_numeric(df["age"], errors="coerce")
    drug_count = pd.to_numeric(df["drug_count"], errors="coerce")
    if age.isna().all() or drug_count.isna().all():
        return False, "Age and drug_count must contain numeric values."
    if ((age <= 0) | (age >= 120)).all():
        return False, "Age values must be between 1 and 119."
    if (drug_count < 0).all():
        return False, "Drug count cannot be negative."

    if "sex" in df.columns:
        normalized = df["sex"].astype(str).str.upper().str.strip()
        if not normalized.isin(["F", "M"]).any():
            return False, "Sex values must contain at least one valid 'F' or 'M' entry."

    return True, "Data validation passed."
