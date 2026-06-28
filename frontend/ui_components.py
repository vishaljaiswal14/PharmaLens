""""UI components and business logic for Drug Risk dashboard.

This module contains all user interface components and business logic
for seamless user interaction and data visualization.

Key Components:
- Executive Summary: Key insights and risk metrics dashboard
- AI Decision Interface: Real-time prediction with risk assessment
- Risk Visualization: Interactive meters and confidence displays
- What-if Simulation: Scenario analysis for clinical decision support
- Distribution Analysis: Demographic and risk factor visualizations
- Comparative Analysis: Gender-based and age-group comparisons
- Correlation Analysis: Feature relationships and outlier detection
- Patient Profiles: Detailed individual patient information cards

UI Features:
- Professional dark theme with consistent styling
- Real-time updates and interactive elements
- Responsive layout for different screen sizes
- Error handling and user feedback
- Data export and template generation

Purpose:
- Separates UI logic from backend processing
- Enables modular development and team collaboration
- Provides consistent user experience across all tabs
- Facilitates easy maintenance and updates
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from core.config import PALETTE, POLYPHARMACY_THRESHOLD, ELDERLY_THRESHOLD
from frontend.visualizations import (
    plot_hist_kde,
    plot_box,
    plot_bar,
    plot_correlation_heatmap,
    plot_stacked_gender_severity,
)


def get_risk_theme(risk: str) -> dict[str, str]:
    """Return consistent colors for each risk category."""
    themes = {
        "Low": {"accent": "#22C55E", "soft": "rgba(34, 197, 94, 0.15)", "border": "rgba(34, 197, 94, 0.3)"},
        "Medium": {"accent": "#F59E0B", "soft": "rgba(245, 158, 11, 0.15)", "border": "rgba(245, 158, 11, 0.3)"},
        "High": {"accent": "#EF4444", "soft": "rgba(239, 68, 68, 0.15)", "border": "rgba(239, 68, 68, 0.3)"},
    }
    return themes.get(risk, {"accent": "#A3A3A3", "soft": "#1E1E1E", "border": "#2C2C2C"})


def render_distribution(df_filtered: pd.DataFrame, selected_plot: str) -> None:
    """Render distribution plots based on selection."""
    if selected_plot == "Age Distribution":
        fig = plot_hist_kde(df_filtered["age"], "Age Distribution", "Age", PALETTE["primary"])
        st.pyplot(fig, use_container_width=True)
        st.caption("Age distribution is concentrated in adult and elderly groups within the selected range.")
    elif selected_plot == "Gender Distribution":
        gender_df = df_filtered["sex"].value_counts().rename_axis("Gender").reset_index(name="Count")
        fig = plot_bar(gender_df, "Gender", "Count", "Gender Distribution", PALETTE["secondary"])
        st.pyplot(fig, use_container_width=True)
        st.caption("Gender representation remains stable, supporting subgroup comparison.")
    elif selected_plot == "Drug Count Distribution":
        fig = plot_hist_kde(df_filtered["drug_count"], "Drug Count Distribution", "Drug Count", PALETTE["neutral"])
        st.pyplot(fig, use_container_width=True)
        st.caption("Most reports involve fewer medications, with a long tail of higher drug counts.")
    elif selected_plot == "Polypharmacy":
        poly_df = (
            df_filtered["polypharmacy"]
            .map({0: "No", 1: "Yes"})
            .value_counts()
            .rename_axis("Polypharmacy")
            .reset_index(name="Count")
        )
        fig = plot_bar(poly_df, "Polypharmacy", "Count", "Polypharmacy Distribution", PALETTE["accent"])
        st.pyplot(fig, use_container_width=True)
        st.caption("Polypharmacy cases represent a meaningful high-complexity subgroup.")
    else:
        severity_df = (
            df_filtered["severity_label"]
            .value_counts()
            .rename_axis("Severity")
            .reset_index(name="Count")
        )
        fig = plot_bar(severity_df, "Severity", "Count", "Severity Distribution", PALETTE["primary"])
        st.pyplot(fig, use_container_width=True)
        st.caption("Serious outcomes account for a substantial portion of filtered reports.")


def render_comparative_analysis(df_filtered: pd.DataFrame) -> None:
    """Render comparative analysis plots."""
    left_col, right_col = st.columns(2)
    
    with left_col:
        fig_age = plot_box(df_filtered, "severity_label", "age", "Age vs Severity", PALETTE["secondary"])
        st.pyplot(fig_age, use_container_width=True)
        avg_age_serious = df_filtered[df_filtered["serious"] == 1]["age"].mean()
        avg_age_non_serious = df_filtered[df_filtered["serious"] == 0]["age"].mean()
        st.caption(f"Serious cases are older on average ({avg_age_serious:.1f} vs {avg_age_non_serious:.1f} years).")
    
    with right_col:
        fig_drug = plot_box(df_filtered, "severity_label", "drug_count", "Drug Count vs Severity", PALETTE["accent"])
        st.pyplot(fig_drug, use_container_width=True)
        avg_drug_serious = df_filtered[df_filtered["serious"] == 1]["drug_count"].mean()
        avg_drug_non_serious = df_filtered[df_filtered["serious"] == 0]["drug_count"].mean()
        st.caption(f"Higher drug counts are linked to serious cases ({avg_drug_serious:.2f} vs {avg_drug_non_serious:.2f}).")

    st.markdown("---")
    fig_stack = plot_stacked_gender_severity(df_filtered)
    st.pyplot(fig_stack, use_container_width=True)
    serious_by_gender = df_filtered.groupby("sex")["serious"].mean().mul(100).round(1).to_dict()
    st.caption(f"Severity composition differs by gender; serious-rate by group: {serious_by_gender}.")


def render_correlation_outliers(df_filtered: pd.DataFrame) -> None:
    """Render correlation heatmap and outlier analysis."""
    corr_cols = [
        "age",
        "sex_code",
        "drug_count",
        "unique_drug_count",
        "reaction_count",
        "risk_score",
        "serious",
        "polypharmacy",
        "elderly",
    ]
    corr_cols = [col for col in corr_cols if col in df_filtered.columns]
    corr_data = df_filtered[corr_cols].rename(columns={"sex_code": "sex"})
    
    fig_heat = plot_correlation_heatmap(corr_data)
    st.pyplot(fig_heat, use_container_width=True)
    st.caption("Drug burden and polypharmacy show positive association with serious outcomes.")

    st.markdown("---")
    out_col1, out_col2 = st.columns(2)
    
    with out_col1:
        fig_age_out = plot_box(df_filtered, "severity_label", "age", "Outlier Analysis: Age", PALETTE["secondary"])
        st.pyplot(fig_age_out, use_container_width=True)
        st.caption("Age spread is broader in serious cases, highlighting clinically diverse risk groups.")
    
    with out_col2:
        fig_drug_out = plot_box(df_filtered, "severity_label", "drug_count", "Outlier Analysis: Drug Count", PALETTE["accent"])
        st.pyplot(fig_drug_out, use_container_width=True)
        st.caption("Drug-count outliers are concentrated among serious reports.")


def build_top_risk_segments(df_filtered: pd.DataFrame) -> pd.DataFrame:
    """Build top risk segments table."""
    risk_tbl = (
        df_filtered.groupby(["age_group", "sex", "polypharmacy"], observed=False)
        .agg(cases=("primaryid", "count"), serious_rate=("serious", "mean"))
        .reset_index()
    )
    risk_tbl["polypharmacy"] = risk_tbl["polypharmacy"].map({0: "No", 1: "Yes"})
    risk_tbl["serious_rate_pct"] = (risk_tbl["serious_rate"] * 100).round(1)
    risk_tbl = risk_tbl.sort_values(["serious_rate_pct", "cases"], ascending=[False, False]).head(10)
    return risk_tbl[["age_group", "sex", "polypharmacy", "cases", "serious_rate_pct"]]


def build_final_stats(df_filtered: pd.DataFrame) -> dict[str, float]:
    """Build final statistics summary."""
    elderly_rate = df_filtered.loc[df_filtered["elderly"] == 1, "serious"].mean() * 100
    poly_rate = df_filtered.loc[df_filtered["polypharmacy"] == 1, "serious"].mean() * 100
    overall_rate = df_filtered["serious"].mean() * 100
    median_age = df_filtered["age"].median()
    median_drug = df_filtered["drug_count"].median()
    return {
        "overall_rate": overall_rate,
        "elderly_rate": elderly_rate,
        "poly_rate": poly_rate,
        "median_age": median_age,
        "median_drug": median_drug,
    }


def safe_mean(series: pd.Series) -> float:
    """Safely calculate mean, handling empty series."""
    return 0.0 if series.empty else float(series.mean())


def render_executive_summary(df_filtered: pd.DataFrame) -> None:
    """Render executive summary section."""
    summary_left, summary_mid, summary_right = st.columns(3)
    
    with summary_left:
        st.markdown(
            f"- Serious outcomes in current cohort: **{safe_mean(df_filtered['serious']) * 100:.1f}%**\n"
            f"- Median patient age: **{df_filtered['age'].median():.1f} years**"
        )
    
    with summary_mid:
        st.markdown(
            f"- Median drug burden: **{df_filtered['drug_count'].median():.1f} drugs**\n"
            f"- Polypharmacy prevalence: **{safe_mean(df_filtered['polypharmacy']) * 100:.1f}%**"
        )
    
    with summary_right:
        st.markdown(
            f"- Elderly subgroup share: **{safe_mean(df_filtered['elderly']) * 100:.1f}%**\n"
            f"- Cases in filter: **{len(df_filtered):,}**"
        )
    
    st.caption("A concise summary of the current cohort for rapid review.")


def render_ai_decision_interface(model, scaler, feature_cols, ml_results) -> dict:
    """Render AI decision engine interface."""
    st.subheader("Patient Inputs")
    input_col1, input_col2 = st.columns(2)

    with input_col1:
        rt_age = st.number_input("Age", min_value=1, max_value=120, value=40, step=1, key="rt_age")
        rt_drug_count = st.number_input("Drug count", min_value=0, max_value=50, value=2, step=1, key="rt_drug_count")

    with input_col2:
        rt_sex = st.selectbox("Sex", ["F", "M"], index=0, key="rt_sex")
        rt_reaction_count = st.number_input("Reaction count", min_value=0, max_value=50, value=1, step=1, key="rt_reaction_count")

    st.caption("Risk assessment updates automatically from the current case profile.")
    with st.spinner("Analyzing patient…"):
        import time
        time.sleep(0.2)
        
        from ml_engine.ml_pipeline import predict_instance_light
        rt = predict_instance_light(
            model=model,
            scaler=scaler,
            feature_cols=feature_cols,
            age=float(rt_age),
            sex=str(rt_sex),
            drug_count=float(rt_drug_count),
            reaction_count=float(rt_reaction_count),
            importance_df=ml_results.get("model_explainability"),
            decision_threshold=float(ml_results.get("best_threshold", 0.5)),
        )

    return rt, rt_age, rt_sex, rt_drug_count, rt_reaction_count


def render_risk_meter(probability: float) -> None:
    """Render risk meter visualization."""
    gauge_left = min(max(probability * 100, 0.0), 100.0)
    st.caption("Risk Probability Score")
    st.markdown(
        f"""
        <div style="position: relative; height: 6px; border-radius: 999px; background: #2C2C2C; margin-top: 12px; margin-bottom: 24px;">
          <div style="position:absolute; left:0; top:0; bottom:0; width:{gauge_left}%; background: linear-gradient(90deg, #22C55E 0%, #EF4444 100%); border-radius: 999px; box-shadow: 0 0 10px rgba(239, 68, 68, 0.4);"></div>
          <div style="position:absolute; top:50%; left: calc({gauge_left}% - 6px); transform: translateY(-50%); width:12px; height:12px; border-radius:50%; background:#F5F5F5; border:1px solid #121212; box-shadow:0 0 8px rgba(0, 0, 0, 0.8);"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Predicted risk probability: **{probability * 100:.2f}%**")


def render_patient_profile(profile: dict) -> None:
    """Render patient profile card."""
    st.markdown(
        f"""
        <div style="background: #121212; border: 1px solid #2C2C2C; border-radius: 12px; padding: 12px 14px;">
          <div style="font-weight: 700; margin-bottom: 8px; color: #F5F5F5;">Patient Profile</div>
          <div class="kv"><div class="k">Age group</div><div class="v"><span class="badge {'bad' if profile['age_group']=='Elderly' else 'good'}"><span class="badge-dot"></span>{profile['age_group']}</span></div></div>
          <div class="kv"><div class="k">Drug burden</div><div class="v"><span class="badge {'warn' if profile['drug_burden']=='High' else 'good'}"><span class="badge-dot"></span>{profile['drug_burden']}</span></div></div>
          <div class="kv"><div class="k">Polypharmacy</div><div class="v"><span class="badge {'warn' if profile['polypharmacy']=='Yes' else 'good'}"><span class="badge-dot"></span>{profile['polypharmacy']}</span></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_what_if_simulation(
    model,
    scaler,
    feature_cols,
    current_risk: dict,
    rt_age: float,
    rt_sex: str,
    rt_drug_count: float,
    rt_reaction_count: float,
    importance_df: pd.DataFrame | None = None,
) -> None:
    """Render what-if simulation interface."""
    st.markdown("#### What-If Review")
    sim_left, sim_mid, sim_right = st.columns(3)
    
    with sim_left:
        sim_age = st.slider("Adjusted age", min_value=1, max_value=120, value=int(rt_age), step=1, key="sim_age")
    
    with sim_right:
        sim_drugs = st.slider("Adjusted drug count", min_value=0, max_value=50, value=int(rt_drug_count), step=1, key="sim_drugs")

    with sim_mid:
        sim_reactions = st.slider(
            "Adjusted reaction count",
            min_value=0,
            max_value=50,
            value=int(rt_reaction_count),
            step=1,
            key="sim_reactions",
        )

    from ml_engine.ml_pipeline import predict_instance_light
    sim = predict_instance_light(
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        age=float(sim_age),
        sex=str(rt_sex),
        drug_count=float(sim_drugs),
        reaction_count=float(sim_reactions),
        importance_df=importance_df,
        decision_threshold=float(current_risk.get("decision_threshold", 0.5)),
    )
    
    cur_risk = current_risk["risk"]
    new_risk = str(sim["risk"])
    sim_delta = (float(sim["prob"]) - float(current_risk["prob"])) * 100

    direction = "Increase" if sim_delta > 0.05 else ("Decrease" if sim_delta < -0.05 else "No material change")
    direction_class = "bad" if sim_delta > 0.05 else ("good" if sim_delta < -0.05 else "warn")

    s1, s2, s3 = st.columns([1, 1, 1.5])
    s1.metric("Current Risk", cur_risk)
    s2.metric("Adjusted Risk", new_risk)
    s3.metric("Probability Delta", f"{sim_delta:+.1f} pp")
    st.markdown(f'<div style="margin-top: 10px;"><span class="badge {direction_class}"><span class="badge-dot"></span>{direction}</span></div>', unsafe_allow_html=True)

    if cur_risk != new_risk:
        st.caption(f"Risk category changed from **{cur_risk}** to **{new_risk}** under the adjusted inputs.")
    else:
        st.caption("Risk category is unchanged under the adjusted inputs.")
