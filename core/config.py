"""Configuration and constants for PharmaLens (Machine Learning-Powered Pharmacovigilance Intelligence Platform).

Key Sections:
- Color Palette: Professional color scheme for all visualizations
- Risk Thresholds: Probability-to-risk level mappings (Low/Medium/High)
- Age Groups: Categorical age bins for demographic analysis
- Drug Categories: Medication count classifications
- ML Parameters: Model hyperparameters and training settings
- Feature Definitions: Core ML feature columns and thresholds

Purpose:
- Ensures consistency across entire application
- Facilitates easy parameter tuning
- Provides single source of truth for constants
"""

from __future__ import annotations

# Color palette for visualizations
PALETTE = {
    "primary": "#4CC9F0",
    "secondary": "#90BE6D",
    "accent": "#F72585",
    "neutral": "#F9C74F",
    "purple": "#8B5CF6",
    "cyan": "#22D3EE",
    "orange": "#FB923C",
}

# Risk level definitions
RISK_THRESHOLDS = {
    "low": 0.40,
    "medium": 0.70,
    "high": 1.0,
}

# Age group definitions
AGE_BINS = [0, 12, 18, 35, 50, 65, 80, 120]
AGE_LABELS = ["Child", "Teen", "Young Adult", "Adult", "Older Adult", "Senior", "Elderly"]

# Drug count categories
DRUG_COUNT_BINS = [0, 2, 5, 10, 50]
DRUG_COUNT_LABELS = ["Low", "Moderate", "High", "Extreme"]

# ML model parameters
ML_CONFIG = {
    "max_train_rows": 70000,
    "test_size": 0.3,
    "validation_split": 0.5,
    "random_state": 42,
    "n_clusters": 3,
    "threshold_grid_start": 0.2,
    "threshold_grid_end": 0.81,
    "threshold_grid_step": 0.02,
    "use_class_weight": True,
    "use_smote": False,
    "permutation_importance_max_rows": 5000,
}

# Model hyperparameters
MODEL_PARAMS = {
    "RandomForest": {
        "n_estimators": 120,
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced",
    },
    "DecisionTree": {
        "max_depth": 8,
        "random_state": 42,
        "class_weight": "balanced",
    },
    "LogisticRegression": {
        "max_iter": 1200,
        "random_state": 42,
        "class_weight": "balanced",
    },
    "MLPClassifier": {
        "hidden_layer_sizes": (96, 48, 24),
        "activation": "relu",
        "solver": "adam",
        "random_state": 42,
        "max_iter": 120,
        "early_stopping": True,
        "n_iter_no_change": 8,
    },
}

# Feature columns for ML
FEATURE_COLS = [
    "age",
    "sex_code",
    "drug_count",
    "unique_drug_count",
    "drug_repeat_flag",
    "reaction_count",
    "polypharmacy",
    "elderly",
    "risk_score",
]

# Polypharmacy threshold
POLYPHARMACY_THRESHOLD = 5

# Elderly threshold
ELDERLY_THRESHOLD = 65

# Source data files
DATA_FILES = {
    "demo": "DEMO25Q4.txt",
    "drug": "DRUG25Q4.txt",
    "outc": "OUTC25Q4.txt",
    "reac": "REAC25Q4.txt",
}
