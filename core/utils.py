"""Utility functions and helpers for Drug Risk dashboard.

This module provides common helper functions, data validation,
and business logic utilities used across multiple components.

Key Functions:
- Risk Mapping: Converts probabilities to categorical risk levels
- Patient Profiles: Builds comprehensive patient information
- AI Interpretation: Generates natural language explanations
- Template Management: Creates and validates upload templates
- Data Validation: Ensures data quality and format compliance
- Format Helpers: Standardizes text and number formatting
- Metric Calculations: Computes risk deltas and statistics
- Export Utilities: Handles data download and formatting

Purpose:
- Reduces code duplication across modules
- Provides consistent business logic implementation
- Enables easy testing and maintenance
- Facilitates data quality and validation
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from core.config import ELDERLY_THRESHOLD, POLYPHARMACY_THRESHOLD


def format_ai_decision_text(rt: dict) -> str:
    """Format AI decision summary text."""
    label = str(rt["label"])
    prob = float(rt["prob"])
    risk = str(rt["risk"])
    conf_pct = prob * 100
    drivers = list(rt["drivers"])
    primary_drivers = ", ".join(drivers) if drivers else "No major risk drivers"
    
    decision_text = (
        f"**AI Decision Summary**  \n"
        f"Risk level: **{risk}** | Predicted label: **{label}** | Risk confidence: **{conf_pct:.2f}%**  \n"
        f"Key drivers: **{primary_drivers}**  \n"
        f"{rt['interpretation']}"
    )
    
    return decision_text


def create_ai_story_flow(age: float, drug_count: float, risk: str) -> tuple[str, str]:
    """Create AI story flow and because text."""
    flow_text = f"Age {int(age)} + Drug Count {int(drug_count)} → {risk} Risk"
    
    because_parts = []
    if float(drug_count) >= POLYPHARMACY_THRESHOLD:
        because_parts.append("high medication load")
    if float(age) >= ELDERLY_THRESHOLD:
        because_parts.append("age-related vulnerability")
    
    because_text = " + ".join(because_parts) if because_parts else "balanced age and medication profile"
    
    return flow_text, because_text


def get_ai_insight_text(risk: str, because_text: str) -> str:
    """Generate AI insight narrative."""
    return (
        f"**AI Insight:** Based on the patient's profile, the system detects **{risk.lower()} risk** "
        f"driven by {because_text}. Closer monitoring may be required if risk factors persist."
    )


def calculate_risk_deltas(df_filtered: pd.DataFrame) -> dict[str, float]:
    """Calculate risk deltas for insight metrics."""
    from frontend.ui_components import safe_mean
    
    elderly_serious = safe_mean(df_filtered.loc[df_filtered["elderly"] == 1, "serious"]) * 100
    non_elderly_serious = safe_mean(df_filtered.loc[df_filtered["elderly"] == 0, "serious"]) * 100
    poly_serious = safe_mean(df_filtered.loc[df_filtered["polypharmacy"] == 1, "serious"]) * 100
    non_poly_serious = safe_mean(df_filtered.loc[df_filtered["polypharmacy"] == 0, "serious"]) * 100
    avg_drug_delta = (
        safe_mean(df_filtered.loc[df_filtered["serious"] == 1, "drug_count"])
        - safe_mean(df_filtered.loc[df_filtered["serious"] == 0, "drug_count"])
    )
    
    return {
        "elderly_risk_delta": elderly_serious - non_elderly_serious,
        "polypharmacy_risk_delta": poly_serious - non_poly_serious,
        "drug_count_gap": avg_drug_delta,
    }


def render_insight_metrics(df_filtered: pd.DataFrame) -> None:
    """Render insight metrics strip."""
    deltas = calculate_risk_deltas(df_filtered)
    
    insight_col1, insight_col2, insight_col3, action_col = st.columns([1.1, 1.1, 1.1, 1.0])
    
    with insight_col1:
        st.metric("Elderly Risk Delta", f"{deltas['elderly_risk_delta']:.1f} pp")
    
    with insight_col2:
        st.metric("Polypharmacy Risk Delta", f"{deltas['polypharmacy_risk_delta']:.1f} pp")
    
    with insight_col3:
        st.metric("Drug Count Gap", f"{deltas['drug_count_gap']:.2f}")
    
    with action_col:
        export_csv = df_filtered.to_csv(index=False).encode("utf-8")
        st.markdown('<div class="filtered-download">', unsafe_allow_html=True)
        st.download_button(
            "Download Filtered CSV",
            data=export_csv,
            file_name="faers_filtered_analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


def render_key_metrics(df_filtered: pd.DataFrame) -> None:
    """Render key metrics section."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df_filtered):,}")
    col2.metric("Average Age", f"{df_filtered['age'].mean():.1f}")
    col3.metric("Average Drug Count", f"{df_filtered['drug_count'].mean():.2f}")
    col4.metric("Serious Cases (%)", f"{df_filtered['serious'].mean() * 100:.1f}%")


def render_final_stats_section(df_filtered: pd.DataFrame) -> None:
    """Render final stats section."""
    from frontend.ui_components import build_final_stats
    
    stats = build_final_stats(df_filtered)
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Overall Serious %", f"{stats['overall_rate']:.1f}%")
    s2.metric("Elderly Serious %", f"{stats['elderly_rate']:.1f}%")
    s3.metric("Polypharmacy Serious %", f"{stats['poly_rate']:.1f}%")
    s4.metric("Median Age", f"{stats['median_age']:.1f}")
    s5.metric("Median Drug Count", f"{stats['median_drug']:.1f}")


def check_shap_availability() -> bool:
    """Check if SHAP is available - deprecated function."""
    return False
