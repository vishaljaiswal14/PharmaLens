import pytest
import pandas as pd
import numpy as np

# Absolute package imports
from core.config import ELDERLY_THRESHOLD, POLYPHARMACY_THRESHOLD, FEATURE_COLS
from core.utils import create_ai_story_flow, calculate_risk_deltas
from ml_engine.risk_logic import (
    risk_level_from_prob,
    probability_to_risk,
    validate_upload_data,
    prepare_external_features,
)

def test_risk_categorization():
    """Verify that probability thresholds map correctly to Low, Medium, High risk categories."""
    assert risk_level_from_prob(0.15) == "Low"
    assert risk_level_from_prob(0.45) == "Medium"
    assert risk_level_from_prob(0.75) == "High"
    
    # Test Series mapping
    probs = pd.Series([0.15, 0.45, 0.75])
    risks = probability_to_risk(probs)
    assert list(risks) == ["Low", "Medium", "High"]

def test_ai_story_flow():
    """Verify the generation of clinical logic stories and rationale strings."""
    # Scenario 1: Elderly + Polypharmacy
    flow, reason = create_ai_story_flow(age=70, drug_count=8, risk="High")
    assert "70" in flow
    assert "8" in flow
    assert "high medication load" in reason
    assert "age-related vulnerability" in reason

    # Scenario 2: Young + Safe Medications count
    flow2, reason2 = create_ai_story_flow(age=30, drug_count=2, risk="Low")
    assert "30" in flow2
    assert "2" in flow2
    assert "balanced age and medication profile" in reason2

def test_data_validation():
    """Verify that file uploader validation correctly flags valid and invalid data schemas."""
    # Valid columns schema
    valid_data = pd.DataFrame({
        "age": [45, 67],
        "sex": ["M", "F"],
        "drug_count": [3, 6],
        "reaction_count": [1, 4]
    })
    is_valid, error = validate_upload_data(valid_data)
    assert is_valid is True
    assert error == "Data validation passed."

    # Invalid columns schema (missing drug_count)
    invalid_data = pd.DataFrame({
        "age": [45, 67],
        "sex": ["M", "F"],
        "reaction_count": [1, 4]
    })
    is_valid2, error2 = validate_upload_data(invalid_data)
    assert is_valid2 is False
    assert "drug_count" in error2

def test_feature_engineering_prep():
    """Test feature scaling preparation and validation for external inputs."""
    upload_df = pd.DataFrame({
        "age": [50.0, 75.0],
        "sex": ["M", "F"],
        "drug_count": [6.0, 3.0],
        "reaction_count": [2.0, 4.0]
    })
    
    # Run the feature preparation
    features = prepare_external_features(upload_df, FEATURE_COLS)
    
    assert len(features) == 2
    assert "sex_code" in features.columns
    assert "polypharmacy" in features.columns
    assert "elderly" in features.columns
    assert "risk_score" in features.columns

def test_calculate_risk_deltas():
    """Verify risk delta computations over cohort tables."""
    cohort = pd.DataFrame({
        "age": [70, 75, 30, 25],
        "elderly": [1, 1, 0, 0],
        "polypharmacy": [1, 0, 1, 0],
        "drug_count": [8, 4, 6, 2],
        "serious": [1, 1, 0, 0]
    })
    
    deltas = calculate_risk_deltas(cohort)
    assert "elderly_risk_delta" in deltas
    assert "polypharmacy_risk_delta" in deltas
    assert "drug_count_gap" in deltas
    
    # Assert values
    assert deltas["elderly_risk_delta"] == 100.0
    assert deltas["polypharmacy_risk_delta"] == 0.0
    assert deltas["drug_count_gap"] == 2.0  # (8+4)/2 - (6+2)/2 = 6 - 4 = 2.0
