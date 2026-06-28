"""Decision support helpers for pharmacovigilance risk intelligence."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_engine.risk_logic import probability_to_risk


DRIVER_FEATURES = [
    "risk_score",
    "drug_count",
    "unique_drug_count",
    "reaction_count",
    "age",
    "polypharmacy",
    "elderly",
    "drug_repeat_flag",
]


def _safe_rate(df: pd.DataFrame, mask_col: str, target_col: str = "serious") -> tuple[float, float]:
    """Return target rates for flag true and false cohorts."""
    if mask_col not in df.columns or target_col not in df.columns:
        return 0.0, 0.0
    true_rate = df.loc[df[mask_col] == 1, target_col].mean()
    false_rate = df.loc[df[mask_col] == 0, target_col].mean()
    return float(true_rate or 0.0) * 100, float(false_rate or 0.0) * 100


def _relative_change(true_rate: float, false_rate: float) -> float | None:
    """Return relative percent change from false cohort to true cohort."""
    if false_rate <= 0:
        return None
    return ((true_rate - false_rate) / false_rate) * 100


def _relative_phrase(true_rate: float, false_rate: float) -> str:
    """Format a relative change safely when the comparison rate is zero."""
    rel = _relative_change(true_rate, false_rate)
    if rel is None:
        return "an undefined relative increase because the comparison group has 0.0% serious outcomes"
    return f"{rel:+.1f}% relative risk change"


def build_risk_segmentation(pred_output: pd.DataFrame) -> pd.DataFrame:
    """Summarize model-scored patients by probability risk segment."""
    if pred_output.empty or "predicted_probability" not in pred_output.columns:
        return pd.DataFrame()

    scored = pred_output.copy()
    scored["risk_segment"] = probability_to_risk(scored["predicted_probability"])
    feature_cols = [col for col in DRIVER_FEATURES if col in scored.columns]
    overall = scored[feature_cols].mean(numeric_only=True)

    rows = []
    for segment in ["Low", "Medium", "High"]:
        seg = scored[scored["risk_segment"] == segment]
        if seg.empty:
            rows.append(
                {
                    "risk_segment": segment,
                    "patients": 0,
                    "share_pct": 0.0,
                    "avg_probability_pct": 0.0,
                    "key_drivers": "No patients in segment",
                }
            )
            continue

        seg_means = seg[feature_cols].mean(numeric_only=True)
        driver_gaps = (seg_means - overall).abs().sort_values(ascending=False)
        key_drivers = ", ".join(driver_gaps.head(3).index.tolist())
        row = {
            "risk_segment": segment,
            "patients": int(len(seg)),
            "share_pct": round(len(seg) / len(scored) * 100, 2),
            "avg_probability_pct": round(seg["predicted_probability"].mean() * 100, 2),
            "key_drivers": key_drivers,
        }
        for col in feature_cols:
            row[f"avg_{col}"] = round(float(seg_means.get(col, 0.0)), 2)
        rows.append(row)

    return pd.DataFrame(rows)


def generate_clinical_insights(df: pd.DataFrame) -> list[str]:
    """Generate data-driven cohort insights from filtered FAERS data."""
    if df.empty:
        return ["No records are available for the current filter."]

    insights: list[str] = []
    overall_rate = float(df["serious"].mean()) * 100 if "serious" in df.columns else 0.0

    poly_rate, non_poly_rate = _safe_rate(df, "polypharmacy")
    if poly_rate or non_poly_rate:
        insights.append(
            f"Polypharmacy is associated with {_relative_phrase(poly_rate, non_poly_rate)} versus non-polypharmacy cases "
            f"({poly_rate:.1f}% vs {non_poly_rate:.1f}%, {poly_rate - non_poly_rate:+.1f} percentage points)."
        )

    elderly_rate, non_elderly_rate = _safe_rate(df, "elderly")
    if elderly_rate or non_elderly_rate:
        insights.append(
            f"Elderly status is associated with {_relative_phrase(elderly_rate, non_elderly_rate)} versus non-elderly patients "
            f"({elderly_rate:.1f}% vs {non_elderly_rate:.1f}%, {elderly_rate - non_elderly_rate:+.1f} percentage points)."
        )

    if "reaction_count" in df.columns:
        high_reaction_cutoff = max(3, float(df["reaction_count"].quantile(0.75)))
        high_reaction = df[df["reaction_count"] >= high_reaction_cutoff]
        low_reaction = df[df["reaction_count"] < high_reaction_cutoff]
        if not high_reaction.empty and not low_reaction.empty:
            high_rate = float(high_reaction["serious"].mean()) * 100
            low_rate = float(low_reaction["serious"].mean()) * 100
            insights.append(
                f"Cases with at least {high_reaction_cutoff:.0f} reactions show "
                f"{_relative_phrase(high_rate, low_rate)} compared with lower-reaction cases "
                f"({high_rate:.1f}% vs {low_rate:.1f}%)."
            )

    if "drug_repeat_flag" in df.columns:
        repeat_rate, non_repeat_rate = _safe_rate(df, "drug_repeat_flag")
        insights.append(
            f"Repeated drug entries appear in {df['drug_repeat_flag'].mean() * 100:.1f}% of cases; "
            f"they are associated with {_relative_phrase(repeat_rate, non_repeat_rate)} versus non-repeated drug lists "
            f"({repeat_rate:.1f}% vs {non_repeat_rate:.1f}%)."
        )

    insights.append(f"The current filtered cohort has an overall serious outcome rate of {overall_rate:.1f}%.")
    return insights


def failure_case_guidance() -> pd.DataFrame:
    """Describe situations where predictions should be interpreted cautiously."""
    return pd.DataFrame(
        [
            {
                "failure_case": "Missing or invalid clinical fields",
                "why_it_matters": "Age, sex, drug burden, and reaction counts are simplified proxies; missing values reduce context.",
                "mitigation": "Validate source completeness before acting on high-risk or low-risk outputs.",
            },
            {
                "failure_case": "No reaction semantics",
                "why_it_matters": "The model counts reactions but does not understand whether terms are clinically severe or redundant.",
                "mitigation": "Review reaction terms manually for high-risk cases and consider MedDRA grouping in future work.",
            },
            {
                "failure_case": "FAERS reporting bias",
                "why_it_matters": "FAERS is spontaneous reporting data and can overrepresent unusual or severe reports.",
                "mitigation": "Use outputs for triage and signal exploration, not causal claims.",
            },
            {
                "failure_case": "Drug name normalization limits",
                "why_it_matters": "Basic uppercase/trim normalization may treat equivalent product names as different drugs.",
                "mitigation": "Use standardized active ingredient mapping before regulatory-grade analysis.",
            },
            {
                "failure_case": "Low confidence predictions",
                "why_it_matters": "Probabilities near the tuned threshold are more sensitive to small feature changes.",
                "mitigation": "Prioritize manual review when confidence is Low or model explanations conflict with clinical context.",
            },
        ]
    )


def build_model_comparison_summary(eval_df: pd.DataFrame) -> list[str]:
    """Translate model metrics into plain-language model selection guidance."""
    if eval_df.empty:
        return ["Model comparison is unavailable for the current filter."]

    df = eval_df.copy()
    best_f1 = df.loc[df["f1"].idxmax()]
    best_recall = df.loc[df["recall"].idxmax()]
    best_precision = df.loc[df["precision"].idxmax()]
    best_auc = df.loc[df["roc_auc"].idxmax()] if "roc_auc" in df.columns else best_f1

    summaries = [
        f"{best_f1['model']} is the most balanced model by F1-score ({best_f1['f1']:.3f}).",
        f"{best_recall['model']} has the strongest serious-case capture with recall {best_recall['recall']:.3f}.",
        f"{best_precision['model']} is the most selective model with precision {best_precision['precision']:.3f}.",
        f"{best_auc['model']} separates serious and non-serious cases best by ROC-AUC ({best_auc['roc_auc']:.3f}).",
    ]

    low_thresholds = df[df["threshold"] < 0.35]
    if not low_thresholds.empty:
        names = ", ".join(low_thresholds["model"].tolist())
        summaries.append(f"Low selected thresholds for {names} indicate the system is prioritizing recall over precision.")

    return summaries


def build_data_quality_report(
    *,
    demo: pd.DataFrame,
    drug: pd.DataFrame,
    outc: pd.DataFrame,
    reac: pd.DataFrame,
    processed: pd.DataFrame,
) -> dict[str, pd.DataFrame | int | float]:
    """Create a compact data quality report for dashboard display."""
    source_rows = {
        "DEMO": len(demo),
        "DRUG": len(drug),
        "OUTC": len(outc),
        "REAC": len(reac),
    }
    source_df = pd.DataFrame({"source": list(source_rows.keys()), "rows": list(source_rows.values())})

    important_raw_cols = {
        "DEMO": [col for col in ["primaryid", "age", "sex"] if col in demo.columns],
        "DRUG": [col for col in ["primaryid", "drugname"] if col in drug.columns],
        "OUTC": [col for col in ["primaryid", "outc_cod"] if col in outc.columns],
        "REAC": [col for col in ["primaryid", "pt"] if col in reac.columns],
    }
    missing_rows = []
    for source, frame in {"DEMO": demo, "DRUG": drug, "OUTC": outc, "REAC": reac}.items():
        for col in important_raw_cols[source]:
            missing_rows.append(
                {
                    "source": source,
                    "column": col,
                    "missing_pct": round(float(frame[col].isna().mean()) * 100, 2),
                }
            )

    feature_cols = [
        "age",
        "drug_count",
        "unique_drug_count",
        "reaction_count",
        "risk_score",
    ]
    feature_cols = [col for col in feature_cols if col in processed.columns]
    feature_summary = processed[feature_cols].describe().transpose().reset_index().rename(columns={"index": "feature"})
    for col in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
        if col in feature_summary.columns:
            feature_summary[col] = feature_summary[col].round(2)

    return {
        "source_rows": source_df,
        "missing_values": pd.DataFrame(missing_rows),
        "feature_summary": feature_summary,
        "final_rows": int(len(processed)),
        "demo_rows_dropped": int(max(len(demo) - len(processed), 0)),
        "retention_pct": round(len(processed) / len(demo) * 100, 2) if len(demo) else 0.0,
    }
