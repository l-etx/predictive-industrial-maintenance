# Predictive Maintenance for Industrial Equipment

An end-to-end machine learning project that predicts industrial machine failures from operational sensor data. The goal is to shift from reactive to **proactive maintenance** — catching failures before they happen, reducing unplanned downtime and its associated costs.

Motivated by hands-on experience with high-precision industrial equipment, where unexpected failures have significant operational and financial consequences.

---

## Problem Statement

Industrial machines continuously generate sensor readings during operation. Failures are rare but costly: they cause production downtime, emergency maintenance, and potential damage to downstream processes. Traditional scheduled maintenance is inefficient — it either intervenes too early (waste) or too late (failure).

This project frames failure prediction as a **binary classification problem**: given a set of sensor readings from a machine cycle, predict whether a failure will occur.

---

## Dataset

**AI4I 2020 Predictive Maintenance Dataset** — a synthetic but realistic dataset from the UCI Machine Learning Repository, designed to reflect real-world predictive maintenance scenarios.

- ~10,000 samples, one per machine cycle
- **Target variable:** `Machine failure` (binary: 0 = no failure, 1 = failure)
- Class imbalance: failures represent ~3.4% of samples

| Feature | Description |
|---------|-------------|
| `Air temperature [K]` | Ambient temperature |
| `Process temperature [K]` | Temperature of the active process |
| `Rotational speed [rpm]` | Tool rotational speed |
| `Torque [Nm]` | Rotational force applied |
| `Tool wear [min]` | Cumulative tool usage time |

Failure sub-types (tool wear, heat dissipation, power, overstrain, random) are available but the primary task targets the binary failure label.

---

## Pipeline

```
Data → EDA → Feature Engineering → Modeling → Evaluation → Explainability
```

1. **Data understanding & cleaning** — audit nulls, types, outliers, and class distribution
2. **Exploratory data analysis** — sensor distributions, feature correlations, failure pattern visualization
3. **Feature engineering** — derived features such as `power = torque × rotational speed` and `ΔT = process temp − air temp`
4. **Modeling** — baseline models (logistic regression, decision tree) followed by advanced models (random forest, XGBoost); class imbalance handled via class weights and threshold tuning
5. **Evaluation** — focus on **recall and ROC-AUC** over accuracy, given the imbalanced dataset; a missed failure is more costly than a false alarm
6. **Explainability** — SHAP (SHapley Additive exPlanations) used to explain individual predictions and identify the most influential sensor features

---

## Project Structure

```
predictive-industrial-maintenance/
├── data/
│   └── raw/               # Original AI4I 2020 dataset
├── notebooks/             # Jupyter notebooks (EDA, modeling, evaluation)
├── src/                   # Reusable Python modules
├── models/                # Saved trained models
├── reports/
│   └── figures/           # Plots: ROC curves, SHAP summaries, confusion matrices
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/l-etx/predictive-industrial-maintenance.git
cd predictive-industrial-maintenance
pip install -r requirements.txt
```

Run notebooks in order from the `notebooks/` directory using JupyterLab:

```bash
jupyter lab
```

---

## Tools & Libraries

| Library | Purpose |
|---------|---------|
| `pandas` / `numpy` | Data manipulation and numerical computation |
| `scikit-learn` | Preprocessing, baseline models, cross-validation, metrics |
| `xgboost` | Gradient boosting — primary model for tabular classification |
| `matplotlib` | EDA and evaluation plots |
| `shap` | Model interpretability — feature-level explanation of predictions |

---

## Key Design Decisions

**Why recall over accuracy?** With only ~3.4% failures, a naive classifier that always predicts "no failure" achieves 96.6% accuracy — but is completely useless. Evaluation focuses on recall (catching real failures) and ROC-AUC (ranking quality across thresholds).

**Why SHAP?** In industrial settings, operators need to understand *why* a model flags a failure, not just that it did. SHAP provides per-prediction, feature-level explanations that make the model actionable and trustworthy.

**Why XGBoost?** Gradient boosting consistently outperforms other algorithms on imbalanced tabular data and provides native feature importance, complementing SHAP analysis.

---

## Status

Work in progress. Completed stages will be checked off as the project advances.

- [ ] Data cleaning and validation
- [ ] Exploratory data analysis
- [ ] Feature engineering
- [ ] Baseline modeling
- [ ] XGBoost modeling
- [ ] Evaluation report
- [ ] SHAP explainability analysis
- [ ] Results summary

---

## Dataset Reference

Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset*. UCI Machine Learning Repository.
