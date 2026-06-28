"""Machine learning pipeline and prediction utilities for Drug Risk dashboard.

This module implements the complete ML workflow from training to prediction.
Handles multiple algorithms, model evaluation, and real-time inference.

Key Components:
- Model Training: RF, NN, LR, DT with hyperparameter optimization
- Probability Mapping: Binary classification to risk levels (Low/Medium/High)
- Real-time Prediction: Single instance inference with confidence scores
- Model Evaluation: Accuracy, calibration, feature importance analysis
- Risk Stratification: Evidence-based threshold optimization

ML Algorithms:
- RandomForest: Ensemble method with 120 estimators
- Neural Network: Multi-layer perceptron with ReLU activation
- Logistic Regression: Linear baseline model
- Decision Tree: Interpretable rule-based classifier

Output:
- Risk levels with probability confidence
- Feature importance for model explainability
- Calibration curves for reliability assessment
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway, pearsonr, spearmanr, ttest_ind
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from core.config import FEATURE_COLS, ML_CONFIG, MODEL_PARAMS
from ml_engine.risk_logic import (
    action_from_risk,
    build_ai_interpretation,
    build_feature_frame,
    build_patient_profile,
    confidence_from_probability,
    explain_prediction,
    prepare_external_features,
    probability_to_risk,
    risk_level_from_prob,
)
from core.pipeline_logging import get_logger


logger = get_logger(__name__)


def _threshold_grid() -> np.ndarray:
    """Return the configured threshold search grid."""
    return np.arange(
        ML_CONFIG["threshold_grid_start"],
        ML_CONFIG["threshold_grid_end"],
        ML_CONFIG["threshold_grid_step"],
    )


def _model_params(model_key: str, *, balanced: bool) -> dict:
    """Return model params, optionally removing class weights for baseline comparison."""
    params = MODEL_PARAMS[model_key].copy()
    if not balanced:
        params.pop("class_weight", None)
    return params


def build_model_candidates(*, balanced: bool = True) -> dict:
    """Create fresh model instances for each training pass."""
    return {
        "ANN (MLPClassifier)": Pipeline(
            [
                (
                    "model",
                    MLPClassifier(**MODEL_PARAMS["MLPClassifier"]),
                ),
            ]
        ),
        "Logistic Regression": LogisticRegression(**_model_params("LogisticRegression", balanced=balanced)),
        "Decision Tree": DecisionTreeClassifier(**_model_params("DecisionTree", balanced=balanced)),
        "Random Forest": RandomForestClassifier(**_model_params("RandomForest", balanced=balanced)),
    }


def evaluate_thresholds(y_true, y_prob, model_name: str) -> pd.DataFrame:
    """Evaluate precision, recall, and F1 across configured thresholds."""
    rows = []
    for thr in _threshold_grid():
        pred = (y_prob >= thr).astype(int)
        rows.append(
            {
                "model": model_name,
                "threshold": float(thr),
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def _roc_curve_frame(y_true, y_prob, model_name: str) -> pd.DataFrame:
    """Build a dataframe of ROC curve coordinates for a model."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return pd.DataFrame({"model": model_name, "fpr": fpr, "tpr": tpr, "threshold": thresholds})


def _pr_curve_frame(y_true, y_prob, model_name: str) -> pd.DataFrame:
    """Build a dataframe of PR curve coordinates for a model."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns thresholds len=N-1
    thresh = np.append(thresholds, 1.0)
    return pd.DataFrame({"model": model_name, "precision": precision, "recall": recall, "threshold": thresh})


def _tree_importance_frame(model, feature_cols: list[str], model_name: str) -> pd.DataFrame | None:
    """Return native tree feature importance where supported."""
    tree_model = model
    if hasattr(model, "named_steps"):
        tree_model = model.named_steps.get("model", model)
    if not hasattr(tree_model, "feature_importances_"):
        return None
    return (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": tree_model.feature_importances_,
                "source": f"{model_name} native importance",
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _best_importance_frame(ml_results: dict) -> pd.DataFrame | None:
    """Choose the most relevant available importance table."""
    importance_df = ml_results.get("model_explainability")
    if isinstance(importance_df, pd.DataFrame) and not importance_df.empty:
        return importance_df
    importance_df = ml_results.get("tree_feature_importance")
    if isinstance(importance_df, pd.DataFrame) and not importance_df.empty:
        return importance_df
    return None


def predict_instance(
    ml_results: dict,
    *,
    age: float,
    sex: str,
    drug_count: float,
    reaction_count: float = 0,
    unique_drug_count: float | None = None,
    drug_repeat_flag: int | None = None,
) -> dict[str, object]:
    """Prediction helper using pipeline result dictionary objects."""
    X = build_feature_frame(
        age=float(age),
        sex=str(sex),
        drug_count=float(drug_count),
        reaction_count=float(reaction_count),
        unique_drug_count=unique_drug_count,
        drug_repeat_flag=drug_repeat_flag,
        feature_cols=ml_results["feature_cols"],
    )

    X_scaled = ml_results["scaler"].transform(X)
    model = ml_results["best_model"]
    pred = int(model.predict(X_scaled)[0])
    prob = float(model.predict_proba(X_scaled)[0][1])
    risk = risk_level_from_prob(prob)
    label = "Serious" if pred == 1 else "Non-serious"
    profile = build_patient_profile(float(age), float(drug_count))
    interp, drivers = build_ai_interpretation(float(age), float(drug_count))
    explanation = explain_prediction(X, risk, _best_importance_frame(ml_results))
    action = action_from_risk(risk)
    confidence = confidence_from_probability(prob, float(ml_results.get("best_threshold", 0.5)))
    logger.info("Prediction made risk=%s probability=%.4f label=%s", risk, prob, label)

    return {
        "features_df": X,
        "features_scaled": X_scaled,
        "polypharmacy": int(X.iloc[0]["polypharmacy"]),
        "elderly": int(X.iloc[0]["elderly"]),
        "sex_code": int(X.iloc[0]["sex_code"]),
        "reaction_count": float(X.iloc[0]["reaction_count"]) if "reaction_count" in X.columns else 0.0,
        "unique_drug_count": float(X.iloc[0]["unique_drug_count"]) if "unique_drug_count" in X.columns else float(drug_count),
        "drug_repeat_flag": int(X.iloc[0]["drug_repeat_flag"]) if "drug_repeat_flag" in X.columns else 0,
        "risk_score": float(X.iloc[0]["risk_score"]) if "risk_score" in X.columns else 0.0,
        "decision_threshold": float(ml_results.get("best_threshold", 0.5)),
        "pred": pred,
        "prob": prob,
        "risk": risk,
        "label": label,
        "profile": profile,
        "interpretation": interp,
        "drivers": drivers,
        "explanation": explanation,
        "action": action,
        "confidence": confidence,
    }


def predict_instance_light(
    *,
    model,
    scaler,
    feature_cols: list[str],
    age: float,
    sex: str,
    drug_count: float,
    reaction_count: float = 0,
    unique_drug_count: float | None = None,
    drug_repeat_flag: int | None = None,
    importance_df: pd.DataFrame | None = None,
    decision_threshold: float = 0.5,
) -> dict[str, object]:
    """Lightweight real-time prediction path (no retraining)."""
    X = build_feature_frame(
        age=float(age),
        sex=str(sex),
        drug_count=float(drug_count),
        reaction_count=float(reaction_count),
        unique_drug_count=unique_drug_count,
        drug_repeat_flag=drug_repeat_flag,
        feature_cols=feature_cols,
    )
    X_scaled = scaler.transform(X)
    pred = int(model.predict(X_scaled)[0])
    prob = float(model.predict_proba(X_scaled)[0][1])
    risk = risk_level_from_prob(prob)
    label = "Serious" if pred == 1 else "Non-serious"
    profile = build_patient_profile(float(age), float(drug_count))
    interp, drivers = build_ai_interpretation(float(age), float(drug_count))
    explanation = explain_prediction(X, risk, importance_df)
    action = action_from_risk(risk)
    confidence = confidence_from_probability(prob, decision_threshold)
    logger.info("Prediction made risk=%s probability=%.4f label=%s", risk, prob, label)

    return {
        "features_df": X,
        "features_scaled": X_scaled,
        "polypharmacy": int(X.iloc[0]["polypharmacy"]),
        "elderly": int(X.iloc[0]["elderly"]),
        "sex_code": int(X.iloc[0]["sex_code"]),
        "reaction_count": float(X.iloc[0]["reaction_count"]) if "reaction_count" in X.columns else 0.0,
        "unique_drug_count": float(X.iloc[0]["unique_drug_count"]) if "unique_drug_count" in X.columns else float(drug_count),
        "drug_repeat_flag": int(X.iloc[0]["drug_repeat_flag"]) if "drug_repeat_flag" in X.columns else 0,
        "risk_score": float(X.iloc[0]["risk_score"]) if "risk_score" in X.columns else 0.0,
        "decision_threshold": float(decision_threshold),
        "pred": pred,
        "prob": prob,
        "risk": risk,
        "label": label,
        "profile": profile,
        "interpretation": interp,
        "drivers": drivers,
        "explanation": explanation,
        "action": action,
        "confidence": confidence,
    }

def run_ml_pipeline(df_model: pd.DataFrame) -> dict:
    """Run full ML analytics pipeline and return all artifacts for UI."""
    logger.info("Starting ML pipeline input_shape=%s", df_model.shape)
    feature_cols = FEATURE_COLS
    ml_df = df_model[feature_cols + ["age_group", "serious"]].dropna().copy()
    if ml_df["serious"].nunique() < 2:
        raise ValueError("Need at least two target classes in the filtered data for ML training.")

    max_train_rows = ML_CONFIG["max_train_rows"]
    if len(ml_df) > max_train_rows:
        sampled_groups = []
        class_counts = ml_df["serious"].value_counts()
        for target_value, class_df in ml_df.groupby("serious"):
            frac = class_counts[target_value] / len(ml_df)
            sample_size = min(len(class_df), max(1, int(max_train_rows * frac)))
            sampled_groups.append(class_df.sample(n=sample_size, random_state=ML_CONFIG["random_state"]))
        ml_df = pd.concat(sampled_groups, axis=0).sample(frac=1, random_state=ML_CONFIG["random_state"]).reset_index(drop=True)

    X_raw = ml_df[feature_cols].copy()
    y = ml_df["serious"].astype(int)

    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X_raw, y, test_size=ML_CONFIG["test_size"], random_state=ML_CONFIG["random_state"], stratify=y
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw,
        y_temp,
        test_size=ML_CONFIG["validation_split"],
        random_state=ML_CONFIG["random_state"],
        stratify=y_temp,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)
    X_train_val = scaler.transform(pd.concat([X_train_raw, X_val_raw], axis=0))
    y_train_val = pd.concat([y_train, y_val], axis=0)

    X_test_df = X_test_raw.reset_index(drop=True).copy()

    selector = SelectKBest(score_func=mutual_info_classif, k=min(5, len(feature_cols)))
    selector.fit(X_train, y_train)
    feature_scores = (
        pd.DataFrame({"feature": feature_cols, "score": selector.scores_})
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    pca = PCA(n_components=min(3, len(feature_cols)), random_state=42)
    pca.fit(X_train)
    pca_df = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )

    eval_rows = []
    fit_models = {}
    prob_store = {}
    threshold_store = {}
    threshold_frames = []
    roc_frames = []
    pr_frames = []
    imbalance_rows = []
    use_balanced = bool(ML_CONFIG.get("use_class_weight", True))
    models = build_model_candidates(balanced=use_balanced)

    for model_name, model_obj in models.items():
        logger.info("Training model=%s", model_name)
        model_obj.fit(X_train, y_train)
        val_prob = model_obj.predict_proba(X_val)[:, 1]
        best_thr = 0.5
        best_f1 = -1.0
        val_threshold_df = evaluate_thresholds(y_val, val_prob, model_name)
        for row in val_threshold_df.itertuples(index=False):
            score = float(row.f1)
            thr = float(row.threshold)
            val_pred_thr = (val_prob >= thr).astype(int)
            if score > best_f1:
                best_f1 = score
                best_thr = float(thr)

        threshold_store[model_name] = best_thr
        model_refit = build_model_candidates(balanced=use_balanced)[model_name]
        model_refit.fit(X_train_val, y_train_val)
        y_prob = model_refit.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= best_thr).astype(int)
        fit_models[model_name] = model_refit
        prob_store[model_name] = y_prob
        threshold_frames.append(evaluate_thresholds(y_test, y_prob, model_name))
        roc_frames.append(_roc_curve_frame(y_test, y_prob, model_name))
        pr_frames.append(_pr_curve_frame(y_test, y_prob, model_name))
        roc_auc = roc_auc_score(y_test, y_prob) if pd.Series(y_test).nunique() > 1 else np.nan

        eval_rows.append(
            {
                "model": model_name,
                "class_weight": "balanced" if use_balanced and model_name != "ANN (MLPClassifier)" else "none",
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc,
                "threshold": best_thr,
                "brier_score": brier_score_loss(y_test, y_prob),
            }
        )

        if model_name != "ANN (MLPClassifier)" and use_balanced:
            baseline_model = build_model_candidates(balanced=False)[model_name]
            baseline_model.fit(X_train_val, y_train_val)
            baseline_prob = baseline_model.predict_proba(X_test)[:, 1]
            baseline_pred = (baseline_prob >= best_thr).astype(int)
            imbalance_rows.extend(
                [
                    {
                        "model": model_name,
                        "setting": "unweighted",
                        "threshold": best_thr,
                        "precision": precision_score(y_test, baseline_pred, zero_division=0),
                        "recall": recall_score(y_test, baseline_pred, zero_division=0),
                        "f1": f1_score(y_test, baseline_pred, zero_division=0),
                        "roc_auc": roc_auc_score(y_test, baseline_prob) if pd.Series(y_test).nunique() > 1 else np.nan,
                    },
                    {
                        "model": model_name,
                        "setting": "class_weight=balanced",
                        "threshold": best_thr,
                        "precision": precision_score(y_test, y_pred, zero_division=0),
                        "recall": recall_score(y_test, y_pred, zero_division=0),
                        "f1": f1_score(y_test, y_pred, zero_division=0),
                        "roc_auc": roc_auc,
                    },
                ]
            )

    eval_df = pd.DataFrame(eval_rows).sort_values("f1", ascending=False).reset_index(drop=True)
    threshold_analysis = pd.concat(threshold_frames, axis=0, ignore_index=True)
    roc_curve_df = pd.concat(roc_frames, axis=0, ignore_index=True)
    pr_curve_df = pd.concat(pr_frames, axis=0, ignore_index=True)
    imbalance_comparison = pd.DataFrame(imbalance_rows)
    best_model_name = eval_df.iloc[0]["model"]
    best_model = fit_models[best_model_name]
    best_threshold = float(threshold_store[best_model_name])
    best_prob = prob_store[best_model_name]
    best_pred = (best_prob >= best_threshold).astype(int)
    prob_risk = probability_to_risk(pd.Series(best_prob))

    if pd.Series(y_test).nunique() > 1:
        calib_true, calib_pred = calibration_curve(y_test, best_prob, n_bins=10, strategy="uniform")
    else:
        calib_true, calib_pred = [0, 1], [0, 1]

    tree_feature_importance = _tree_importance_frame(best_model, feature_cols, best_model_name)
    if tree_feature_importance is None and fit_models.get("Random Forest") is not None:
        tree_feature_importance = _tree_importance_frame(fit_models["Random Forest"], feature_cols, "Random Forest")

    model_explainability = None
    try:
        importance_model = best_model if best_model_name in {"Decision Tree", "Random Forest"} else fit_models["Random Forest"]
        perm_rows = min(len(X_test), int(ML_CONFIG.get("permutation_importance_max_rows", 5000)))
        X_perm = X_test[:perm_rows]
        y_perm = y_test.iloc[:perm_rows] if hasattr(y_test, "iloc") else y_test[:perm_rows]
        perm = permutation_importance(
            importance_model,
            X_perm,
            y_perm,
            n_repeats=5,
            random_state=ML_CONFIG["random_state"],
            n_jobs=1,
            scoring="f1",
        )
        model_explainability = (
            pd.DataFrame(
                {
                    "feature": feature_cols,
                    "importance": perm.importances_mean,
                    "importance_std": perm.importances_std,
                    "source": f"{best_model_name} permutation importance",
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception:
        model_explainability = None

    cluster_scaler = StandardScaler()
    X_cluster = cluster_scaler.fit_transform(X_raw)
    kmeans = KMeans(
        n_clusters=ML_CONFIG["n_clusters"],
        random_state=ML_CONFIG["random_state"],
        n_init=10,
    )
    cluster_labels = kmeans.fit_predict(X_cluster)
    cluster_df = ml_df.copy()
    cluster_df["cluster"] = cluster_labels
    X_pca = pca.transform(X_cluster)
    cluster_df["PC1"] = X_pca[:, 0]
    cluster_df["PC2"] = X_pca[:, 1] if X_pca.shape[1] > 1 else 0
    cluster_viz_df = cluster_df[["serious", "cluster", "PC1", "PC2", "age", "drug_count"]].copy()
    
    cluster_summary = (
        cluster_df.groupby("cluster")
        .agg(
            cases=("serious", "size"),
            serious_rate=("serious", "mean"),
            avg_age=("age", "mean"),
            avg_drug_count=("drug_count", "mean"),
            polypharmacy_rate=("polypharmacy", "mean"),
        )
        .reset_index()
    )
    sil_score = silhouette_score(X_cluster, cluster_labels) if len(np.unique(cluster_labels)) > 1 else 0.0

    serious_grp = ml_df[ml_df["serious"] == 1]
    non_serious_grp = ml_df[ml_df["serious"] == 0]
    t_stat, t_p = ttest_ind(serious_grp["age"], non_serious_grp["age"], equal_var=False, nan_policy="omit")
    chi_sex = pd.crosstab(ml_df["sex_code"], ml_df["serious"])
    chi_poly = pd.crosstab(ml_df["polypharmacy"], ml_df["serious"])
    chi_sex_stat, chi_sex_p, _, _ = chi2_contingency(chi_sex)
    chi_poly_stat, chi_poly_p, _, _ = chi2_contingency(chi_poly)
    anova_groups = [grp["drug_count"] for _, grp in ml_df.groupby("age_group", observed=True)]
    anova_groups = [grp for grp in anova_groups if len(grp) > 1]
    if len(anova_groups) >= 2:
        f_stat, anova_p = f_oneway(*anova_groups)
    else:
        f_stat, anova_p = np.nan, np.nan
    pearson_r, pearson_p = pearsonr(ml_df["drug_count"], ml_df["serious"])
    spearman_r, spearman_p = spearmanr(ml_df["drug_count"], ml_df["serious"])

    stats_tests = pd.DataFrame(
        [
            {"test": "t-test (age: serious vs non-serious)", "statistic": t_stat, "p_value": t_p},
            {"test": "chi-square (sex vs serious)", "statistic": chi_sex_stat, "p_value": chi_sex_p},
            {
                "test": "chi-square (polypharmacy vs serious)",
                "statistic": chi_poly_stat,
                "p_value": chi_poly_p,
            },
            {"test": "ANOVA (drug_count across age groups)", "statistic": f_stat, "p_value": anova_p},
            {"test": "Pearson (drug_count vs serious)", "statistic": pearson_r, "p_value": pearson_p},
            {"test": "Spearman (drug_count vs serious)", "statistic": spearman_r, "p_value": spearman_p},
        ]
    )

    return {
        "eval_df": eval_df,
        "threshold_analysis": threshold_analysis,
        "roc_curve_df": roc_curve_df,
        "imbalance_comparison": imbalance_comparison,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "best_threshold": best_threshold,
        "random_forest_model": fit_models.get("Random Forest"),
        "random_forest_threshold": float(threshold_store.get("Random Forest", 0.5)),
        "scaler": scaler,
        "feature_cols": feature_cols,
        "y_test": y_test.reset_index(drop=True),
        "X_test": X_test_df,
        "best_pred": pd.Series(best_pred),
        "best_prob": pd.Series(best_prob).reset_index(drop=True),
        "prob_risk": prob_risk.reset_index(drop=True),
        "calib_true": calib_true,
        "calib_pred": calib_pred,
        "model_explainability": model_explainability,
        "tree_feature_importance": tree_feature_importance,
        "feature_scores": feature_scores,
        "pca_df": pca_df,
        "cluster_summary": cluster_summary,
        "cluster_viz_df": cluster_viz_df,
        "silhouette_score": sil_score,
        "stats_tests": stats_tests,
        "pr_curve_df": pr_curve_df,
    }
