""""Data loading and preprocessing utilities for Drug Risk dashboard.

This module handles the complete data pipeline from raw FDA FAERS files
to analysis-ready structured data for machine learning.

Key Functions:
- load_data(): Loads raw DEMO, DRUG, OUTC, REAC files
- preprocess_data(): Cleans, merges, and engineers features
- resolve_column(): Handles column name variations across datasets

Features Created:
- reaction_count: number of reported reactions per case
- unique_drug_count: number of unique normalized drug names per case
- drug_repeat_flag: duplicate drug rows present for a case
- polypharmacy: 5+ medications (high risk indicator)
- elderly: 65+ years (age-based risk factor)
- age_group: Categorical age bins for analysis
- sex_code: Numeric encoding for ML models

Output: Clean DataFrame with primaryid as unique patient identifier
"""

from __future__ import annotations

import pandas as pd

from core.config import AGE_BINS, AGE_LABELS, DATA_FILES, ELDERLY_THRESHOLD, POLYPHARMACY_THRESHOLD
from core.pipeline_logging import get_logger


logger = get_logger(__name__)


def resolve_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Resolve a canonical column name from a list of case-insensitive candidates."""
    col_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in col_map:
            return col_map[cand.lower()]
    raise KeyError(f"Missing expected column. Tried: {candidates}")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load source FAERS-like files."""
    logger.info("Loading FAERS source files: %s", DATA_FILES)
    demo = pd.read_csv(DATA_FILES["demo"], sep="$", encoding="latin1", engine="python", on_bad_lines="skip")
    drug = pd.read_csv(DATA_FILES["drug"], sep="$", encoding="latin1", engine="python", on_bad_lines="skip")
    outc = pd.read_csv(DATA_FILES["outc"], sep="$", encoding="latin1", engine="python", on_bad_lines="skip")
    reac = pd.read_csv(DATA_FILES["reac"], sep="$", encoding="latin1", engine="python", on_bad_lines="skip")
    logger.info("Loaded source shapes demo=%s drug=%s outc=%s reac=%s", demo.shape, drug.shape, outc.shape, reac.shape)
    return demo, drug, outc, reac


def _build_risk_score(df: pd.DataFrame) -> pd.Series:
    """Build a leakage-safe composite score from pre-outcome patient factors."""
    age_component = (df["age"].clip(lower=0, upper=100) / 100) * 25
    drug_component = (df["drug_count"].clip(lower=0, upper=20) / 20) * 30
    unique_drug_component = (df["unique_drug_count"].clip(lower=0, upper=20) / 20) * 15
    reaction_component = (df["reaction_count"].clip(lower=0, upper=10) / 10) * 20
    flag_component = (df["polypharmacy"] * 7) + (df["drug_repeat_flag"] * 3)
    return (age_component + drug_component + unique_drug_component + reaction_component + flag_component).round(2)


def preprocess_data(
    demo: pd.DataFrame,
    drug: pd.DataFrame,
    outc: pd.DataFrame,
    reac: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Clean raw sources, engineer features, and return analysis-ready dataframe."""
    logger.info("Starting preprocessing demo=%s drug=%s outc=%s reac=%s", demo.shape, drug.shape, outc.shape, None if reac is None else reac.shape)
    demo_pid = resolve_column(demo, ["primaryid"])
    demo_age = resolve_column(demo, ["age"])
    demo_sex = resolve_column(demo, ["sex", "gender"])
    drug_pid = resolve_column(drug, ["primaryid"])
    drug_name = resolve_column(drug, ["drugname"])
    outc_pid = resolve_column(outc, ["primaryid"])
    reac_pid = resolve_column(reac, ["primaryid"]) if reac is not None else None

    demo = demo[[demo_pid, demo_age, demo_sex]].rename(columns={demo_pid: "primaryid", demo_age: "age", demo_sex: "sex"})
    demo["age"] = pd.to_numeric(demo["age"], errors="coerce")
    demo["sex"] = demo["sex"].astype(str).str.upper().str.strip()
    demo = demo[(demo["age"] > 0) & (demo["age"] < 120)]
    demo = demo[demo["sex"].isin(["M", "F"])].dropna(subset=["primaryid", "age", "sex"])

    drug = drug[[drug_pid, drug_name]].rename(columns={drug_pid: "primaryid", drug_name: "drugname"}).dropna(subset=["primaryid"])
    drug["drugname_norm"] = drug["drugname"].astype(str).str.upper().str.strip()
    drug["drugname_norm"] = drug["drugname_norm"].replace({"": pd.NA, "NAN": pd.NA, "UNK": pd.NA, "UNKNOWN": pd.NA})
    outc = outc[[outc_pid]].rename(columns={outc_pid: "primaryid"}).dropna(subset=["primaryid"])
    if reac is not None:
        reac = reac[[reac_pid]].rename(columns={reac_pid: "primaryid"}).dropna(subset=["primaryid"])

    drug_count = (
        drug.groupby("primaryid")
        .agg(
            drug_count=("primaryid", "size"),
            unique_drug_count=("drugname_norm", "nunique"),
        )
        .reset_index()
    )
    drug_count["drug_repeat_flag"] = (drug_count["drug_count"] > drug_count["unique_drug_count"]).astype(int)
    serious = outc.assign(serious=1).groupby("primaryid", as_index=False)["serious"].max()
    reaction_count = (
        reac.groupby("primaryid").size().reset_index(name="reaction_count")
        if reac is not None
        else pd.DataFrame(columns=["primaryid", "reaction_count"])
    )

    df = (
        demo.merge(drug_count, on="primaryid", how="left")
        .merge(serious, on="primaryid", how="left")
        .merge(reaction_count, on="primaryid", how="left")
    )
    df["drug_count"] = df["drug_count"].fillna(0).astype(int)
    df["unique_drug_count"] = df["unique_drug_count"].fillna(0).astype(int)
    df["drug_repeat_flag"] = df["drug_repeat_flag"].fillna(0).astype(int)
    df["reaction_count"] = df["reaction_count"].fillna(0).astype(int)
    df["serious"] = df["serious"].fillna(0).astype(int)
    df["polypharmacy"] = (df["drug_count"] >= POLYPHARMACY_THRESHOLD).astype(int)
    df["elderly"] = (df["age"] >= ELDERLY_THRESHOLD).astype(int)
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True)
    df["sex_code"] = df["sex"].map({"F": 0, "M": 1})
    df["risk_score"] = _build_risk_score(df)
    required = [
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
    ]
    df = df.dropna(subset=required).copy()
    df["sex_code"] = df["sex_code"].astype(int)
    df["severity_label"] = df["serious"].map({0: "Non-serious", 1: "Serious"})
    logger.info("Preprocessing complete final_shape=%s features=%s", df.shape, list(df.columns))
    return df
