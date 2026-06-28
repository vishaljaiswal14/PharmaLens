"""Visualization utilities for Drug Risk Analysis Dashboard.

This module provides all plotting functions with consistent theming
and professional styling for data exploration and model evaluation.

Chart Types:
- Distribution Plots: Age, gender, drug count with KDE overlays
- Comparative Charts: Box plots and bar charts for group analysis
- Correlation Analysis: Heatmaps and relationship matrices
- Model Evaluation: Confusion matrices and calibration curves
- Feature Importance: SHAP-style visualizations and ranking charts

Styling Features:
- Professional dark theme with consistent color palette
- High-resolution figures with proper labeling
- Responsive sizing for different screen dimensions
- Accessibility features with clear legends and titles

Purpose:
- Separates visualization logic from business logic
- Ensures consistent styling across all charts
- Enables easy maintenance and updates
- Provides reusable components for different analyses
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from core.config import PALETTE


FIG_BG = "#121212"
AX_BG = "#121212"
TEXT = "#F5F5F5"
MUTED = "#A3A3A3"
GRID = "#2C2C2C"


def _style_axis(ax, title: str, xlabel: str | None = None, ylabel: str | None = None) -> None:
    """Apply a neutral, product-style chart treatment."""
    ax.set_title(title, color=TEXT, loc="left", pad=12, fontsize=12, fontweight="semibold")
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=MUTED)
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color("#D9E2EC")
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def plot_hist_kde(series: pd.Series, title: str, xlabel: str, color: str) -> plt.Figure:
    """Create histogram with KDE overlay."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    sns.histplot(series, bins=30, kde=True, color=color, ax=ax)
    _style_axis(ax, title, xlabel, "Count")
    return fig


def plot_box(data: pd.DataFrame, x: str, y: str, title: str, color: str) -> plt.Figure:
    """Create box plot for categorical vs numerical analysis."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    sns.boxplot(data=data, x=x, y=y, color=color, ax=ax)
    _style_axis(ax, title, ax.get_xlabel(), ax.get_ylabel())
    return fig


def plot_bar(data: pd.DataFrame, x: str, y: str, title: str, color: str) -> plt.Figure:
    """Create bar plot for categorical analysis."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    sns.barplot(data=data, x=x, y=y, color=color, ax=ax)
    _style_axis(ax, title, ax.get_xlabel(), ax.get_ylabel())
    return fig


def plot_correlation_heatmap(data: pd.DataFrame) -> plt.Figure:
    """Create correlation heatmap."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    sns.heatmap(
        data.corr(numeric_only=True),
        annot=True,
        fmt=".2f",
        cmap="mako",
        linewidths=0.5,
        linecolor=FIG_BG,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap", color=TEXT, loc="left", pad=12, fontsize=12, fontweight="semibold")
    ax.tick_params(colors=MUTED)
    return fig


def plot_stacked_gender_severity(data: pd.DataFrame) -> plt.Figure:
    """Create stacked bar chart for gender vs severity."""
    gender_stacked = data.groupby(["sex", "severity_label"]).size().reset_index(name="count")
    pivot_stacked = gender_stacked.pivot(index="sex", columns="severity_label", values="count").fillna(0)
    
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    pivot_stacked.plot(kind="bar", stacked=True, ax=ax, color=[PALETTE["secondary"], PALETTE["primary"]])
    _style_axis(ax, "Gender vs Severity", "Gender", "Count")
    ax.legend(facecolor=FIG_BG, edgecolor="#D9E2EC", labelcolor=TEXT)
    return fig


def plot_calibration_curve(true_values, predicted_probs) -> plt.Figure:
    """Create calibration curve plot."""
    from sklearn.calibration import calibration_curve
    
    calib_true, calib_pred = calibration_curve(true_values, predicted_probs, n_bins=10, strategy="uniform")
    
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    ax.plot(calib_pred, calib_true, marker='o', color=PALETTE["primary"], linewidth=2, label='Model')
    ax.plot([0, 1], [0, 1], '--', color=PALETTE["neutral"], label='Perfect')
    _style_axis(ax, "Calibration Curve", "Mean Predicted Probability", "Fraction of Positives")
    ax.legend(facecolor=FIG_BG, edgecolor="#D9E2EC", labelcolor=TEXT)
    return fig


def plot_roc_curves(roc_df: pd.DataFrame, auc_df: pd.DataFrame | None = None) -> plt.Figure:
    """Create ROC curves for all evaluated models."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    auc_lookup = {}
    if auc_df is not None and {"model", "roc_auc"}.issubset(auc_df.columns):
        auc_lookup = auc_df.set_index("model")["roc_auc"].to_dict()

    for model_name, model_df in roc_df.groupby("model"):
        auc_value = auc_lookup.get(model_name)
        label = f"{model_name} (AUC={auc_value:.3f})" if pd.notna(auc_value) else str(model_name)
        ax.plot(model_df["fpr"], model_df["tpr"], linewidth=2, label=label)

    ax.plot([0, 1], [0, 1], "--", color=PALETTE["neutral"], label="Random")
    _style_axis(ax, "ROC Curves", "False Positive Rate", "True Positive Rate")
    ax.legend(facecolor=FIG_BG, edgecolor="#D9E2EC", labelcolor=TEXT, fontsize=8)
    return fig


def plot_threshold_metrics(threshold_df: pd.DataFrame, model_name: str) -> plt.Figure:
    """Plot precision, recall, and F1 across classification thresholds."""
    model_df = threshold_df[threshold_df["model"] == model_name].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    for metric, color in [
        ("precision", PALETTE["primary"]),
        ("recall", PALETTE["secondary"]),
        ("f1", PALETTE["accent"]),
    ]:
        ax.plot(model_df["threshold"], model_df[metric], marker="o", markersize=3, linewidth=2, label=metric.title(), color=color)

    ax.set_ylim(0, 1.05)
    _style_axis(ax, f"Threshold Analysis: {model_name}", "Threshold", "Score")
    ax.legend(facecolor=FIG_BG, edgecolor="#D9E2EC", labelcolor=TEXT)
    return fig


def plot_confusion_matrix(y_true, y_pred) -> plt.Figure:
    """Create confusion matrix plot."""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='mako',
        linewidths=0.5,
        linecolor=FIG_BG,
        ax=ax,
        xticklabels=['Non-serious', 'Serious'],
        yticklabels=['Non-serious', 'Serious'],
    )
    ax.set_title("Confusion Matrix", color=TEXT, loc="left", pad=12, fontsize=12, fontweight="semibold")
    ax.set_xlabel("Predicted", color=MUTED)
    ax.set_ylabel("Actual", color=MUTED)
    ax.tick_params(colors=MUTED)
    return fig


def plot_feature_importance(importance_df: pd.DataFrame) -> plt.Figure:
    """Create feature importance bar plot."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    sns.barplot(data=importance_df, x="importance", y="feature", ax=ax, color=PALETTE["accent"])
    _style_axis(ax, "Feature Importance", "Importance", "Features")
    return fig


def plot_risk_segment_distribution(segment_df: pd.DataFrame) -> plt.Figure:
    """Plot patient counts by predicted risk segment."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    order = ["Low", "Medium", "High"]
    plot_df = segment_df.copy()
    plot_df["risk_segment"] = pd.Categorical(plot_df["risk_segment"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("risk_segment")
    colors = [PALETTE["secondary"], PALETTE["neutral"], PALETTE["accent"]]
    sns.barplot(data=plot_df, x="risk_segment", y="patients", ax=ax, palette=colors, hue="risk_segment", legend=False)
    _style_axis(ax, "Predicted Risk Segment Distribution", "Risk Segment", "Patients")
    return fig


def plot_precision_recall_curves(pr_df: pd.DataFrame) -> plt.Figure:
    """Create Precision-Recall curves for evaluated models."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    
    for model_name, model_df in pr_df.groupby("model"):
        ax.plot(model_df["recall"], model_df["precision"], linewidth=2, label=model_name)

    _style_axis(ax, "Precision-Recall Curves", "Recall", "Precision")
    ax.legend(facecolor=FIG_BG, edgecolor="#D9E2EC", labelcolor=TEXT, fontsize=8)
    return fig


def plot_pca_clusters(pca_viz_df: pd.DataFrame, color_col: str, title: str) -> plt.Figure:
    """Create a 2D scatter plot of PCA components."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    
    unique_vals = pca_viz_df[color_col].nunique()
    palette = sns.color_palette("deep", unique_vals) if unique_vals > 2 else [PALETTE["secondary"], PALETTE["primary"]]
    
    sns.scatterplot(
        data=pca_viz_df,
        x="PC1",
        y="PC2",
        hue=color_col,
        palette=palette,
        alpha=0.6,
        s=40,
        ax=ax,
        edgecolor=None
    )
    
    _style_axis(ax, title, "Principal Component 1", "Principal Component 2")
    ax.legend(facecolor=FIG_BG, edgecolor="#D9E2EC", labelcolor=TEXT, fontsize=8, title=color_col.capitalize())
    return fig


def plot_model_comparison(eval_df: pd.DataFrame) -> plt.Figure:
    """Plot F1-Score and ROC-AUC for all models."""
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    
    plot_df = eval_df.set_index("model")[["f1", "roc_auc"]].copy()
    plot_df.plot(kind="bar", ax=ax, color=[PALETTE["accent"], PALETTE["primary"]], width=0.6)
    
    _style_axis(ax, "Model Comparison", "", "Score")
    ax.set_ylim(0, 1.05)
    ax.legend(["F1 Score", "ROC AUC"], facecolor=FIG_BG, edgecolor="#D9E2EC", labelcolor=TEXT, fontsize=8, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.4))
    
    # Improve label readability
    plt.xticks(rotation=15, ha='right', color=MUTED, fontsize=9)
    fig.subplots_adjust(bottom=0.3)
    
    return fig
