"""Streamlit entry point for the pharmacovigilance risk intelligence system."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score

from core.config import AGE_BINS, AGE_LABELS, FEATURE_COLS
from data.data_processing import load_data, preprocess_data
from ml_engine.decision_support import (
    build_data_quality_report,
    build_model_comparison_summary,
    build_risk_segmentation,
    failure_case_guidance,
    generate_clinical_insights,
)
from ml_engine.ml_pipeline import prepare_external_features, run_ml_pipeline
from ml_engine.risk_logic import create_download_template, probability_to_risk, validate_upload_data
from frontend.ui_components import (
    build_final_stats,
    build_top_risk_segments,
    get_risk_theme,
    render_ai_decision_interface,
    render_comparative_analysis,
    render_correlation_outliers,
    render_distribution,
    render_executive_summary,
    render_patient_profile,
    render_risk_meter,
    render_what_if_simulation,
)
from core.utils import render_final_stats_section, render_key_metrics, render_insight_metrics
from frontend.visualizations import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curves,
    plot_risk_segment_distribution,
    plot_threshold_metrics,
    plot_precision_recall_curves,
    plot_pca_clusters,
    plot_model_comparison,
)


st.set_page_config(page_title="PharmaLens - Machine Learning-Powered Pharmacovigilance Intelligence", layout="wide")


st.markdown(
    """
    <style>
    .stApp {
        background-color: #121212;
        color: #F5F5F5;
    }
    header[data-testid="stHeader"] {
        background-color: #121212 !important;
    }
    .block-container {
        padding-top: 3.5rem;
        padding-bottom: 2rem;
        max-width: 1320px;
    }
    h1, h2, h3 {
        color: #F5F5F5 !important;
        letter-spacing: -0.01em;
    }
    p, label, div[data-testid="stCaptionContainer"] {
        color: #A3A3A3;
    }
    div[data-testid="stMetric"] {
        background: #1E1E1E;
        border: 1px solid #2C2C2C;
        border-radius: 12px;
        padding: 0.9rem 1rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #1E1E1E !important;
        border: 1px solid #2C2C2C !important;
        border-radius: 12px !important;
        padding: 1rem 1.1rem;
    }
    div[data-testid="stTabs"] button[role="tab"] {
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        border-radius: 0;
        color: #A3A3A3 !important;
        padding: 0.55rem 1rem;
        margin-right: 0.35rem;
        font-weight: 600;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background: transparent;
        border-bottom: 2px solid #22C55E !important;
        color: #F5F5F5 !important;
    }
    .stButton > button, div[data-testid="stDownloadButton"] > button {
        background: #1E1E1E;
        color: #F5F5F5;
        border: 1px solid #2C2C2C;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover, div[data-testid="stDownloadButton"] > button:hover {
        border-color: #22C55E;
        color: #22C55E;
    }
    .stSelectbox > div[data-baseweb="select"] > div,
    .stNumberInput input,
    div[data-baseweb="base-input"] input,
    div[data-baseweb="textarea"] textarea {
        background: #1E1E1E !important;
        border: 1px solid #2C2C2C !important;
        border-radius: 8px !important;
        color: #F5F5F5 !important;
    }
    .stExpander {
        border-radius: 12px;
        border: 1px solid #2C2C2C;
        background: #1E1E1E;
    }
    .risk-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 700;
        border: 1px solid;
    }
    .risk-dot {
        width: 0.5rem;
        height: 0.5rem;
        border-radius: 50%;
        background: currentColor;
    }
    .subtle-card {
        border-radius: 12px;
        border: 1px solid #2C2C2C;
        background: #121212;
        padding: 0.95rem 1rem;
        min-height: 130px;
        display: flex;
        align-items: center;
    }
    .decision-card {
        border-left: 4px solid;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        background: #121212;
        margin-top: 0.25rem;
    }
    .kv {
        display: flex;
        justify-content: space-between;
        gap: 0.8rem;
        padding: 0.35rem 0;
        border-bottom: 1px solid #2C2C2C;
    }
    .kv:last-child { border-bottom: 0; }
    .kv .k { color: #A3A3A3; }
    .kv .v {
        color: #F5F5F5;
        font-weight: 600;
        text-align: right;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid transparent;
    }
    .badge-dot {
        width: 0.4rem;
        height: 0.4rem;
        border-radius: 50%;
        background: currentColor;
    }
    .badge.good { color: #F5F5F5; background: rgba(34, 197, 94, 0.15); border-color: rgba(34, 197, 94, 0.3); }
    .badge.warn { color: #F5F5F5; background: rgba(245, 158, 11, 0.15); border-color: rgba(245, 158, 11, 0.3); }
    .badge.bad { color: #F5F5F5; background: rgba(239, 68, 68, 0.15); border-color: rgba(239, 68, 68, 0.3); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_ml_results(df_filtered: pd.DataFrame) -> dict:
    """Cache the heavy ML pipeline so UI changes remain responsive."""
    return run_ml_pipeline(df_filtered.copy())


@st.cache_data(show_spinner=False)
def load_data_cached() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cache dataset loading at the UI layer."""
    return load_data()


def prepare_uploaded_analysis_data(upload_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize an uploaded patient-level CSV into the full dashboard schema."""
    df_upload = upload_df.copy()
    is_valid, validation_message = validate_upload_data(df_upload)
    if not is_valid:
        raise ValueError(validation_message)

    if "serious" not in df_upload.columns:
        if "severity_label" in df_upload.columns:
            severity_map = {"serious": 1, "non-serious": 0, "non serious": 0}
            normalized_severity = df_upload["severity_label"].astype(str).str.lower().str.strip()
            df_upload["serious"] = normalized_severity.map(severity_map)
        else:
            raise ValueError("Uploaded analysis dataset must include either 'serious' or 'severity_label'.")

    normalized = prepare_external_features(df_upload, FEATURE_COLS)
    normalized["serious"] = pd.to_numeric(df_upload.loc[normalized.index, "serious"], errors="coerce")
    normalized = normalized.dropna(subset=["serious"]).copy()
    normalized["serious"] = normalized["serious"].astype(int).clip(0, 1)

    if "sex" in df_upload.columns:
        normalized["sex"] = (
            df_upload.loc[normalized.index, "sex"].astype(str).str.upper().str.strip()
        )
        normalized["sex"] = normalized["sex"].where(normalized["sex"].isin(["F", "M"]), pd.NA)
    else:
        normalized["sex"] = normalized["sex_code"].map({0: "F", 1: "M"})

    normalized["sex"] = normalized["sex"].fillna(normalized["sex_code"].map({0: "F", 1: "M"}))
    if "primaryid" in df_upload.columns:
        normalized["primaryid"] = df_upload.loc[normalized.index, "primaryid"].astype(str)
    else:
        normalized["primaryid"] = normalized.index.astype(str)

    normalized["age_group"] = pd.cut(normalized["age"], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True)
    normalized["severity_label"] = normalized["serious"].map({0: "Non-serious", 1: "Serious"})
    return normalized[
        [
            "primaryid",
            "age",
            "sex",
            "drug_count",
            "unique_drug_count",
            "drug_repeat_flag",
            "reaction_count",
            "serious",
            "polypharmacy",
            "elderly",
            "age_group",
            "sex_code",
            "risk_score",
            "severity_label",
        ]
    ].copy()


def build_uploaded_quality_report(processed: pd.DataFrame) -> dict[str, pd.DataFrame | int | float]:
    """Build a compact quality report for uploaded analysis datasets."""
    missing_rows = [
        {
            "source": "Uploaded CSV",
            "column": col,
            "missing_pct": round(float(processed[col].isna().mean()) * 100, 2),
        }
        for col in processed.columns
    ]
    feature_cols = [col for col in ["age", "drug_count", "unique_drug_count", "reaction_count", "risk_score"] if col in processed.columns]
    feature_summary = processed[feature_cols].describe().transpose().reset_index().rename(columns={"index": "feature"})
    for col in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
        if col in feature_summary.columns:
            feature_summary[col] = feature_summary[col].round(2)

    return {
        "source_rows": pd.DataFrame({"source": ["Uploaded CSV"], "rows": [len(processed)]}),
        "missing_values": pd.DataFrame(missing_rows),
        "feature_summary": feature_summary,
        "final_rows": int(len(processed)),
        "demo_rows_dropped": 0,
        "retention_pct": 100.0,
    }


demo_raw, drug_raw, outc_raw, reac_raw = load_data_cached()
default_df = preprocess_data(demo_raw, drug_raw, outc_raw, reac_raw)
default_quality_report = build_data_quality_report(
    demo=demo_raw,
    drug=drug_raw,
    outc=outc_raw,
    reac=reac_raw,
    processed=default_df,
)

with st.container(border=True):
    st.title("PharmaLens")
    st.caption(
        "Machine Learning-Powered Pharmacovigilance Intelligence Platform — Transforming adverse event data into explainable clinical intelligence."
    )

active_df = default_df.copy()
data_quality_report = default_quality_report

with st.container(border=True):
    filter_col, info_col = st.columns([3, 1])
    with filter_col:
        age_bounds = (int(active_df["age"].min()), int(active_df["age"].max()))
        age_range = st.slider(
            "Global Cohort Age Range (Years)",
            min_value=age_bounds[0],
            max_value=age_bounds[1],
            value=age_bounds,
        )
    with info_col:
        st.markdown("<div style='margin-top: 36px; color: #A3A3A3; font-size: 0.85rem;'>All metrics update based on this filter.</div>", unsafe_allow_html=True)

df_filtered = active_df[(active_df["age"] >= age_range[0]) & (active_df["age"] <= age_range[1])].copy()

try:
    ml_results = get_ml_results(df_filtered)
    ml_error = None
except Exception as exc:  # pragma: no cover - interactive guard
    ml_results = None
    ml_error = exc

eval_df = pd.DataFrame()
best_model_name = "Unavailable"
y_test = pd.Series(dtype="int64")
best_pred = pd.Series(dtype="int64")
best_prob = pd.Series(dtype="float64")
pred_output = pd.DataFrame()
risk_segment_summary = pd.DataFrame()

if ml_results is not None:
    eval_df = ml_results["eval_df"].copy()
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "brier_score"]
    eval_df[metric_cols] = eval_df[metric_cols].round(4)
    best_model_name = ml_results["best_model_name"]
    y_test = pd.Series(ml_results["y_test"]).reset_index(drop=True)
    best_pred = pd.Series(ml_results["best_pred"]).reset_index(drop=True)
    best_prob = pd.Series(ml_results["best_prob"]).reset_index(drop=True)

    pred_output = ml_results["X_test"].reset_index(drop=True).copy()
    pred_output["actual_serious"] = y_test.astype(int)
    pred_output["predicted_serious"] = best_pred.astype(int)
    pred_output["predicted_probability"] = best_prob
    pred_output["predicted_risk_level"] = ml_results["prob_risk"]
    pred_output["prediction_match"] = pred_output["actual_serious"] == pred_output["predicted_serious"]
    pred_output["actual_label"] = pred_output["actual_serious"].map({0: "Non-serious", 1: "Serious"})
    pred_output["predicted_label"] = pred_output["predicted_serious"].map({0: "Non-serious", 1: "Serious"})
    pred_output["confidence_%"] = (pred_output["predicted_probability"] * 100).round(2)
    pred_output["result"] = pred_output["prediction_match"].map({True: "Correct", False: "Mismatch"})
    risk_segment_summary = build_risk_segmentation(pred_output)

tab_overview, tab_eda, tab_comparative, tab_correlation, tab_case, tab_model, tab_quality = st.tabs(
    [
        "Overview",
        "EDA Explorer",
        "Comparative Analysis",
        "Correlation & Outliers",
        "Case Analysis",
        "Model Performance",
        "Data Quality",
    ]
)

with tab_overview:
    st.header("Overview")
    with st.container(border=True):
        render_key_metrics(df_filtered)

    st.markdown("### ")
    with st.container(border=True):
        render_insight_metrics(df_filtered)

    st.markdown("### ")
    with st.container(border=True):
        st.subheader("Key Insights")
        insights = generate_clinical_insights(df_filtered)[:3]
        if insights:
            insight_cols = st.columns(len(insights))
            for idx, insight in enumerate(insights):
                with insight_cols[idx]:
                    st.markdown(f'<div class="subtle-card">{insight}</div>', unsafe_allow_html=True)
        else:
            st.caption("No key insights available.")

    st.markdown("### ")
    with st.container(border=True):
        st.subheader("Cohort Overview")
        render_executive_summary(df_filtered)

    with st.expander("View cohort summary metrics", expanded=False):
        render_final_stats_section(df_filtered)
        st.markdown("### ")
        st.dataframe(build_top_risk_segments(df_filtered), use_container_width=True, hide_index=True)

with tab_case:
    h_col1, h_col2 = st.columns([1, 1])
    with h_col1:
        st.header("Case Analysis")
    with h_col2:
        st.markdown(f"<div style='text-align: right; margin-top: 2.2rem;'><span style='background: #1E1E1E; padding: 0.4rem 0.8rem; border-radius: 8px; border: 1px solid #2C2C2C; font-size: 0.85rem; color: #A3A3A3;'><span style='color: #4ade80;'>●</span> Active Inference Engine: <span style='color: #F5F5F5; font-weight: bold;'>{best_model_name}</span></span></div>", unsafe_allow_html=True)
    if ml_error is not None:
        st.error(f"Case analysis is unavailable for the current filter: {ml_error}")
    else:
        input_col, result_col = st.columns([1.0, 1.15], gap="large")

        with input_col:
            with st.container(border=True):
                rt, rt_age, rt_sex, rt_drug_count, rt_reaction_count = render_ai_decision_interface(
                    ml_results["best_model"],
                    ml_results["scaler"],
                    ml_results["feature_cols"],
                    ml_results,
                )

            st.markdown("### ")
            with st.container(border=True):
                st.subheader("Patient Profile")
                render_patient_profile(rt["profile"])

        with result_col:
            risk = str(rt["risk"])
            risk_theme = get_risk_theme(risk)
            action = rt.get("action", {})
            confidence = rt.get("confidence", {})

            with st.container(border=True):
                st.subheader("Risk Assessment")
                top_metrics = st.columns(2)
                top_metrics[0].metric("Risk Category", risk)
                top_metrics[1].metric("Risk Probability", f"{float(rt['prob']) * 100:.1f}%")
                st.markdown(
                    (
                        f'<span class="risk-pill" style="color:{risk_theme["accent"]};'
                        f'background:{risk_theme["soft"]};border-color:{risk_theme["border"]};">'
                        f'<span class="risk-dot"></span>{risk} risk</span>'
                    ),
                    unsafe_allow_html=True,
                )
                render_risk_meter(float(rt["prob"]))
                st.caption("Risk category is derived from predicted probability thresholds.")

            st.markdown("### ")
            with st.container(border=True):
                st.subheader("Key Risk Drivers")
                explanation = rt.get("explanation", {})
                st.write(explanation.get("summary", "No risk driver summary is available for this case."))
                st.caption("Drivers summarize the factors most associated with the current risk assessment.")
                with st.expander("View detailed risk drivers", expanded=False):
                    driver_rows = explanation.get("drivers", [])
                    if driver_rows:
                        driver_df = pd.DataFrame(driver_rows)
                        show_cols = [
                            col
                            for col in ["label", "value", "importance", "feature_intensity", "impact_pct", "message"]
                            if col in driver_df.columns
                        ]
                        st.dataframe(driver_df[show_cols], use_container_width=True, hide_index=True)
                    else:
                        st.caption("No detailed risk drivers are available for this assessment.")

            st.markdown("### ")
            with st.container(border=True):
                st.subheader("Decision Action")
                action_class = (
                    "bad" if action.get("priority") == "High" else "warn" if action.get("priority") == "Medium" else "good"
                )
                action_text = str(action.get("action", "Monitor"))
                action_text = {
                    "Immediate attention": "Immediate attention required",
                    "Review": "Review recommended",
                    "Monitor": "Routine monitoring sufficient",
                }.get(action_text, action_text)
                st.metric("Decision Action", action_text)
                st.markdown(
                    f"""
                    <div class="decision-card" style="border-left-color:{risk_theme['accent']};">
                        <div class="kv"><div class="k">Action priority</div><div class="v"><span class="badge {action_class}"><span class="badge-dot"></span>{action.get('priority', 'Low')}</span></div></div>
                        <div class="kv"><div class="k">Rationale</div><div class="v">{action.get('rationale', 'Continue monitoring.')}</div></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption("Decision action translates the current risk assessment into an operational next step.")

            st.markdown("### ")
            with st.container(border=True):
                st.subheader("Confidence")
                conf_cols = st.columns(3)
                conf_cols[0].metric("Confidence", confidence.get("level", "NA"))
                conf_cols[1].metric("Confidence Score", f"{confidence.get('score', 0):.1f}%")
                conf_cols[2].metric("Decision threshold", f"{ml_results.get('best_threshold', 0.5):.2f}")
                st.caption("Confidence reflects distance from the decision threshold and model certainty.")
                if confidence.get("level") == "Low":
                    st.info("Low confidence: prediction is near the decision threshold. Manual review recommended.")
                with st.expander("What-if simulation", expanded=False):
                    render_what_if_simulation(
                        ml_results["best_model"],
                        ml_results["scaler"],
                        ml_results["feature_cols"],
                        rt,
                        rt_age,
                        rt_sex,
                        rt_drug_count,
                        rt_reaction_count,
                        ml_results.get("model_explainability"),
                    )

        with st.expander("Batch risk assessment on uploaded CSV", expanded=False):
            template_df = create_download_template()
            st.caption("Use this explicitly for **batch ML scoring** on new, unlabelled patients. (The top dashboard uploader overrides the entire cohort).")
            st.download_button(
                "Download batch scoring template",
                data=template_df.to_csv(index=False).encode("utf-8"),
                file_name="risk_assessment_template.csv",
                mime="text/csv",
                use_container_width=True,
            )
            batch_file = st.file_uploader("Upload CSV for batch scoring", type=["csv"], key="batch_scoring_uploader")
            if batch_file is not None:
                try:
                    batch_df = pd.read_csv(batch_file)
                    is_valid, validation_message = validate_upload_data(batch_df)
                    if not is_valid:
                        st.error(validation_message)
                    else:
                        batch_features = prepare_external_features(batch_df, ml_results["feature_cols"])
                        if len(batch_features) == 0:
                            st.warning("Uploaded file has no valid rows after preprocessing.")
                        else:
                            batch_scaled = ml_results["scaler"].transform(batch_features[ml_results["feature_cols"]])
                            batch_prob = ml_results["best_model"].predict_proba(batch_scaled)[:, 1]
                            batch_pred = (batch_prob >= ml_results["best_threshold"]).astype(int)
                            batch_risk = probability_to_risk(pd.Series(batch_prob))
                            batch_out = batch_features.copy()
                            batch_out["predicted_label"] = pd.Series(batch_pred).map({0: "Non-serious", 1: "Serious"})
                            batch_out["predicted_probability"] = np.round(batch_prob, 4)
                            batch_out["risk_category"] = batch_risk
                            st.success(f"Risk assessment generated for {len(batch_out):,} uploaded rows.")
                            show_cols = ml_results["feature_cols"] + ["predicted_label", "predicted_probability", "risk_category"]
                            st.dataframe(batch_out[show_cols], use_container_width=True, hide_index=True)
                            st.download_button(
                                "Download batch scoring results",
                                data=batch_out[show_cols].to_csv(index=False).encode("utf-8"),
                                file_name="batch_risk_assessment.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )
                except Exception as batch_err:
                    st.error(f"Batch scoring could not be completed: {batch_err}")

with tab_eda:
    st.header("EDA Explorer")
    with st.container(border=True):
        st.subheader("Summary Metrics")
        render_final_stats_section(df_filtered)

    st.markdown("### ")
    with st.container(border=True):
        st.subheader("Feature Distributions")
        dist_cols11, dist_cols12 = st.columns(2)
        with dist_cols11:
            render_distribution(df_filtered, "Age Distribution")
        with dist_cols12:
            render_distribution(df_filtered, "Gender Distribution")
            
        st.markdown("---")
        dist_cols21, dist_cols22 = st.columns(2)
        with dist_cols21:
            render_distribution(df_filtered, "Drug Count Distribution")
        with dist_cols22:
            render_distribution(df_filtered, "Polypharmacy")
            
        st.markdown("---")
        render_distribution(df_filtered, "Severity")

    st.markdown("### ")
    with st.container(border=True):
        st.subheader("Risk Segmentation")
        if risk_segment_summary.empty:
            st.info("Risk category distribution is unavailable for the current cohort.")
        else:
            seg_left, seg_right = st.columns([1.1, 1.0])
            with seg_left:
                st.dataframe(risk_segment_summary, use_container_width=True, hide_index=True)
            with seg_right:
                fig_seg = plot_risk_segment_distribution(risk_segment_summary)
                st.pyplot(fig_seg, use_container_width=True)
            st.caption("Risk category counts are derived from predicted probability thresholds.")

    with st.expander("View advanced cohort statistics", expanded=False):
        cohort_stats = (
            df_filtered.groupby("severity_label", observed=False)
            .agg(
                cases=("primaryid", "count"),
                median_age=("age", "median"),
                avg_drug_count=("drug_count", "mean"),
                avg_reaction_count=("reaction_count", "mean"),
                serious_rate=("serious", "mean"),
            )
            .reset_index()
        )
        cohort_stats["avg_drug_count"] = cohort_stats["avg_drug_count"].round(2)
        cohort_stats["avg_reaction_count"] = cohort_stats["avg_reaction_count"].round(2)
        cohort_stats["serious_rate"] = (cohort_stats["serious_rate"] * 100).round(2)
        st.dataframe(cohort_stats, use_container_width=True, hide_index=True)
        st.markdown("### ")
        st.dataframe(build_top_risk_segments(df_filtered), use_container_width=True, hide_index=True)
        st.markdown("### ")
        st.dataframe(df_filtered.describe(include="all").transpose(), use_container_width=True)

with tab_comparative:
    st.header("Comparative Analysis")
    with st.container(border=True):
        st.subheader("Comparative Relationships")
        render_comparative_analysis(df_filtered)

with tab_correlation:
    st.header("Correlation & Outliers")
    with st.container(border=True):
        st.subheader("Correlation Heatmap & Outliers")
        render_correlation_outliers(df_filtered)

    st.markdown("### ")
    with st.container(border=True):
        st.subheader("Clustering & PCA Insights")
        if "cluster_viz_df" in ml_results:
            viz_left, viz_right = st.columns(2)
            with viz_left:
                fig_pca1 = plot_pca_clusters(ml_results["cluster_viz_df"], color_col="serious", title="PCA by Actual Severity")
                st.pyplot(fig_pca1, use_container_width=True)
            with viz_right:
                viz_df = ml_results["cluster_viz_df"].copy()
                viz_df["cluster_label"] = "Cluster " + viz_df["cluster"].astype(str)
                fig_pca2 = plot_pca_clusters(viz_df, color_col="cluster_label", title="KMeans Clusters")
                st.pyplot(fig_pca2, use_container_width=True)

with tab_model:
    st.header("Model Performance")
    if ml_error is not None:
        st.error(f"Model evaluation is unavailable for the current filter: {ml_error}")
    else:
        with st.container(border=True):
            st.markdown(f"### Best Model Selected: <span style='color: #4ade80;'>{best_model_name}</span>", unsafe_allow_html=True)
            st.markdown("<div style='color: #A3A3A3; font-size: 0.9rem; margin-top: -10px; margin-bottom: 20px;'>Selected autonomously by the AutoML pipeline based on peak F1 and Precision metrics.</div>", unsafe_allow_html=True)
            st.subheader("Model Metrics")
            st.dataframe(eval_df, use_container_width=True, hide_index=True)

        st.markdown("### ")
        with st.container(border=True):
            st.subheader("Model Comparison")
            comp_left, comp_right = st.columns([1, 1.5])
            with comp_left:
                st.markdown("**Guidance**")
                for summary in build_model_comparison_summary(eval_df):
                    st.info(summary)
            with comp_right:
                fig_comp = plot_model_comparison(eval_df)
                st.pyplot(fig_comp, use_container_width=True)

        st.markdown("### ")
        roc_col, cm_col = st.columns(2)
        with roc_col:
            with st.container(border=True):
                st.subheader("ROC Curve")
                fig_roc = plot_roc_curves(ml_results["roc_curve_df"], eval_df)
                st.pyplot(fig_roc, use_container_width=True)
        with cm_col:
            with st.container(border=True):
                st.subheader("Confusion Matrix")
                fig_cm = plot_confusion_matrix(y_test, best_pred)
                st.pyplot(fig_cm, use_container_width=True)

        st.markdown("### ")
        pr_col, cal_col = st.columns(2)
        with pr_col:
            with st.container(border=True):
                st.subheader("Precision-Recall Curve")
                if "pr_curve_df" in ml_results:
                    fig_pr = plot_precision_recall_curves(ml_results["pr_curve_df"])
                    st.pyplot(fig_pr, use_container_width=True)
                else:
                    st.info("PR curve not available.")
        with cal_col:
            with st.container(border=True):
                st.subheader("Calibration Curve")
                fig_cal = plot_calibration_curve(y_test, best_prob)
                st.pyplot(fig_cal, use_container_width=True)

        st.markdown("### ")
        with st.container(border=True):
            st.subheader("Threshold Analysis")
            threshold_models = ml_results["threshold_analysis"]["model"].drop_duplicates().tolist()
            threshold_model = st.selectbox(
                "Model",
                options=threshold_models,
                index=threshold_models.index(best_model_name) if best_model_name in threshold_models else 0,
                key="threshold_model_selector",
            )
            fig_threshold = plot_threshold_metrics(ml_results["threshold_analysis"], threshold_model)
            st.pyplot(fig_threshold, use_container_width=True)
            threshold_view = ml_results["threshold_analysis"]
            threshold_view = threshold_view[threshold_view["model"] == threshold_model].copy()
            threshold_view[["threshold", "precision", "recall", "f1"]] = threshold_view[
                ["threshold", "precision", "recall", "f1"]
            ].round(4)
            st.dataframe(threshold_view, use_container_width=True, hide_index=True)

        if not ml_results["imbalance_comparison"].empty:
            st.markdown("### ")
            with st.container(border=True):
                st.subheader("Class Imbalance Comparison")
                imbalance_view = ml_results["imbalance_comparison"].copy()
                imbalance_view[["precision", "recall", "f1", "roc_auc", "threshold"]] = imbalance_view[
                    ["precision", "recall", "f1", "roc_auc", "threshold"]
                ].round(4)
                st.dataframe(imbalance_view, use_container_width=True, hide_index=True)
                st.caption("Compares unweighted models against class-weight-balanced models at their selected thresholds.")

        with st.expander("View prediction output", expanded=False):
            st.subheader("Actual vs Predicted Output")
            controls_left, controls_right = st.columns([1.2, 1.0])
            with controls_left:
                row_limit = st.selectbox("Rows to display", [25, 50, 100, 200], index=1)
            with controls_right:
                show_mode = st.selectbox("Show records", ["All", "Only Mismatch", "Only Correct"], index=0)

            pred_view = pred_output.copy()
            if show_mode == "Only Mismatch":
                pred_view = pred_view[pred_view["prediction_match"] == False]
            elif show_mode == "Only Correct":
                pred_view = pred_view[pred_view["prediction_match"] == True]

            spotlight_left, spotlight_right = st.columns([1.2, 1.0])
            with spotlight_left:
                if len(pred_view) == 0:
                    st.info("No rows match the current prediction-output filter.")
                    spot_idx = None
                else:
                    spot_idx = st.selectbox(
                        "Spotlight record",
                        options=list(range(len(pred_view))),
                        index=0,
                        format_func=lambda i: f"Row {i + 1} • Age {int(pred_view.iloc[i]['age'])} • Drugs {int(pred_view.iloc[i]['drug_count'])}",
                    )
            with spotlight_right:
                if spot_idx is not None:
                    r = pred_view.iloc[int(spot_idx)]
                    is_match = bool(r["prediction_match"])
                    status_class = "good" if is_match else "bad"
                    status_text = "Correct" if is_match else "Mismatch"
                    risk = str(r["predicted_risk_level"])
                    risk_class = "good" if risk == "Low" else ("warn" if risk == "Medium" else "bad")
                    st.markdown(
                        f"""
                        <div class="kv"><div class="k">Actual label</div><div class="v"><span class="badge {'bad' if r['actual_label']=='Serious' else 'good'}"><span class="badge-dot"></span>{r['actual_label']}</span></div></div>
                        <div class="kv"><div class="k">Predicted label</div><div class="v"><span class="badge {'bad' if r['predicted_label']=='Serious' else 'good'}"><span class="badge-dot"></span>{r['predicted_label']}</span></div></div>
                        <div class="kv"><div class="k">Assessment outcome</div><div class="v"><span class="badge {status_class}"><span class="badge-dot"></span>{status_text}</span></div></div>
                        <div class="kv"><div class="k">Confidence</div><div class="v">{float(r['confidence_%']):.2f}%</div></div>
                        <div class="kv"><div class="k">Risk Category</div><div class="v"><span class="badge {risk_class}"><span class="badge-dot"></span>{risk}</span></div></div>
                        """,
                        unsafe_allow_html=True,
                    )

            readable_cols = [
                "actual_label",
                "predicted_label",
                "predicted_risk_level",
                "confidence_%",
                "result",
                "age",
                "drug_count",
                "unique_drug_count",
                "reaction_count",
                "drug_repeat_flag",
                "polypharmacy",
                "elderly",
            ]
            readable_cols = [col for col in readable_cols if col in pred_view.columns]
            view_tbl = pred_view[readable_cols].copy()
            for flag_col in ["polypharmacy", "elderly", "drug_repeat_flag"]:
                if flag_col in view_tbl.columns:
                    view_tbl[flag_col] = view_tbl[flag_col].map({0: "No", 1: "Yes"})
            st.dataframe(view_tbl.head(row_limit), use_container_width=True, hide_index=True)
            st.download_button(
                "Download prediction output",
                data=view_tbl.to_csv(index=False).encode("utf-8"),
                file_name="model_prediction_output.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with st.expander("View advanced model diagnostics", expanded=False):
            pre_left, pre_right = st.columns(2)
            with pre_left:
                st.markdown("**Feature Selection (Mutual Information)**")
                st.dataframe(ml_results["feature_scores"], use_container_width=True, hide_index=True)
            with pre_right:
                st.markdown("**Dimensionality Reduction (PCA)**")
                pca_view = ml_results["pca_df"].copy()
                pca_view[["explained_variance_ratio", "cumulative_variance_ratio"]] = pca_view[
                    ["explained_variance_ratio", "cumulative_variance_ratio"]
                ].round(4)
                st.dataframe(pca_view, use_container_width=True, hide_index=True)

            if ml_results["tree_feature_importance"] is not None:
                st.markdown("### ")
                st.markdown("**Tree-Based Feature Importance**")
                st.dataframe(ml_results["tree_feature_importance"], use_container_width=True, hide_index=True)
                fig_tree = plot_feature_importance(ml_results["tree_feature_importance"])
                st.pyplot(fig_tree, use_container_width=True)

            if ml_results["model_explainability"] is not None:
                st.markdown("### ")
                st.markdown("**Permutation Importance**")
                st.dataframe(ml_results["model_explainability"], use_container_width=True, hide_index=True)

        with st.expander("View statistical tests and clustering metrics", expanded=False):
            tests_df = ml_results["stats_tests"].copy()
            tests_df["significant_0_05"] = tests_df["p_value"] < 0.05
            tests_view = tests_df.copy()
            tests_view["statistic"] = tests_view["statistic"].astype(float).map(lambda x: f"{x:.4g}" if np.isfinite(x) else "NA")
            tests_view["p_value"] = tests_view["p_value"].astype(float).map(lambda p: f"{p:.3e}" if np.isfinite(p) else "NA")
            st.dataframe(tests_view, use_container_width=True, hide_index=True)
            st.markdown("### ")
            cluster_left, cluster_right = st.columns([0.8, 1.2])
            with cluster_left:
                st.subheader("Clustering metrics")
                st.metric("Silhouette Score", f"{ml_results['silhouette_score']:.3f}")
            with cluster_right:
                cluster_summary = ml_results["cluster_summary"].copy()
                for col in ["serious_rate", "polypharmacy_rate"]:
                    cluster_summary[col] = (cluster_summary[col] * 100).round(2)
                cluster_summary[["avg_age", "avg_drug_count"]] = cluster_summary[["avg_age", "avg_drug_count"]].round(2)
                st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

with tab_quality:
    st.header("Data Quality")
    with st.container(border=True):
        st.subheader("Dataset Overview")
        summary_cols = st.columns(4)
        summary_cols[0].metric("Final dataset size", f"{data_quality_report['final_rows']:,}")
        summary_cols[1].metric("Rows dropped", f"{data_quality_report['demo_rows_dropped']:,}")
        summary_cols[2].metric("Retention", f"{data_quality_report['retention_pct']:.1f}%")
        summary_cols[3].metric("Filtered columns", f"{df_filtered.shape[1]:,}")

    st.markdown("### ")
    quality_left, quality_right = st.columns(2)
    with quality_left:
        with st.container(border=True):
            st.subheader("Missing Values")
            missing_view = data_quality_report["missing_values"].copy().sort_values("missing_pct", ascending=False)
            if missing_view.empty:
                st.caption("No missing-value issues were identified in the monitored fields.")
            else:
                st.dataframe(missing_view.head(20), use_container_width=True, hide_index=True)
    with quality_right:
        with st.container(border=True):
            st.subheader("Limitations")
            failure_view = failure_case_guidance().copy()
            if failure_view.empty:
                st.caption("No limitations available.")
            else:
                st.dataframe(failure_view, use_container_width=True, hide_index=True)

    with st.expander("View full data quality details", expanded=False):
        st.subheader("Source Rows")
        st.dataframe(data_quality_report["source_rows"], use_container_width=True, hide_index=True)
        st.markdown("### ")
        st.subheader("Feature Summary")
        st.dataframe(data_quality_report["feature_summary"], use_container_width=True, hide_index=True)
        st.markdown("### ")
        st.subheader("Filtered Dataset Preview")
        st.dataframe(df_filtered.head(200), use_container_width=True, hide_index=True)
        st.markdown("### ")
        st.subheader("Distribution Metrics")
        st.dataframe(df_filtered.describe(include="all").transpose(), use_container_width=True)
        st.markdown("### ")
        missing_summary = df_filtered.isna().sum().reset_index()
        missing_summary.columns = ["column", "missing_count"]
        st.dataframe(missing_summary, use_container_width=True, hide_index=True)
