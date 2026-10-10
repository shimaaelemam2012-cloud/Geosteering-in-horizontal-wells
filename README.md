# Geosteering-in-horizontal-wells
# Hybrid Machine Learning for Horizontal-Well Geosteering

A hybrid machine-learning workflow for predicting formation boundaries and supporting horizontal-well geosteering using the Eagle Ford benchmark dataset.

The project combines **Dynamic Time Warping (DTW)** for log alignment with **LightGBM** and **CatBoost** regression models. Their predictions are combined through a weighted ensemble to improve formation-boundary prediction accuracy.

> **Decision-support scope:** This project is a research workflow evaluated on historical well data. Predictions require geosteering-team review and do not constitute autonomous drilling instructions. Real-time operational benefits require field validation.

---

## 1. Project Overview

Horizontal-well geosteering requires interpreting measurements relative to formation boundaries and maintaining the well trajectory within the intended target interval.

This project investigates a workflow that:

- Checks and prepares well-log data.
- Aligns gamma-ray measurements with offset/reference information.
- Builds machine-learning predictors.
- Estimates formation top and base true vertical depth (TVD).
- Combines predictions from two regression models.
- Visualizes the measured trajectory and predicted formation boundaries.
- Supports human review before operational decisions.

### Main components

| Component | Purpose |
|---|---|
| Dynamic Time Warping | Align well-log signatures with offset/reference information |
| LightGBM | Predict formation-boundary TVD |
| CatBoost | Provide complementary formation-boundary predictions |
| Weighted ensemble | Combine the two model outputs |
| Optuna | Tune model hyperparameters using development data |
| Well-level GroupKFold | Evaluate generalization across held-out wells |

---

## 2. Dataset

The project uses an **Eagle Ford benchmark dataset containing 773 wells**.

An independent **130-well blind-test set** is used for the reported final evaluation.

Blind-test wells are excluded from hyperparameter tuning and ensemble-weight selection.

**Dataset source:** Kaggle Eagle Ford dataset.

---

## 3. Seven-Step Workflow

### Step 1 — Load Well Data

Load the well data and associated offset/reference information.

Verify that the required measurements, station identifiers, and prediction inputs are available.

### Step 2 — Perform Data Quality Control

Check for:

- Missing gamma-ray measurements.
- Duplicate or incorrectly ordered stations.
- Invalid numerical values.
- Inconsistent units.
- Missing required inputs.

For causal prediction, short gamma-ray gaps are forward-filled. Interpolation is reserved for retrospective visualization and is not used to introduce future information into causal predictors.

### Step 3 — Generate Predictors

Construct the **38 predictors** used by the models.

Feature engineering includes causal transformations such as:

- Forward-filled gamma-ray measurements.
- A trailing 25-sample gamma-ray mean.
- Additional well and alignment-derived predictors defined in the project implementation.

Prediction-time feature definitions and column ordering must match those used during training.

### Step 4 — Align Logs Using DTW

Use Dynamic Time Warping to align available well measurements with offset/reference information.

For causal use, alignment uses only:

- Available offset/reference data.
- Well measurements available up to the current station.

Future measurements from the evaluated well must not be used to predict the current station.

### Step 5 — Predict Formation Boundaries

Generate formation top and base TVD predictions using:

1. LightGBM.
2. CatBoost.
3. A weighted hybrid ensemble.

The ensemble is defined as:

**Hybrid prediction = w × CatBoost prediction + (1 − w) × LightGBM prediction**

Here, `w` is the CatBoost weight.

The ensemble weight is selected using development-set validation only. Blind-test results are not used to choose the weight.

### Step 6 — Visualize and Evaluate

Compare predictions with reference values where labels are available.

Main outputs include:

- Measured well trajectory.
- Predicted formation top and base.
- Predicted target-zone status.
- Residual distributions.
- RMSE and absolute-error summaries.
- Formation-specific performance.

For unlabeled uploaded wells, accuracy metrics cannot be calculated without reference boundaries.

### Step 7 — Human Review and Export

Use predictions as inputs to a geosteering-team review process:

**Review → QC → Recalibrate if needed → Validate**

The team remains responsible for interpreting results and deciding whether additional data checks or recalibration are needed.

An interactive upload, approval, and export interface is a planned extension of the research workflow.

---

## 4. Validation Strategy

Model development uses:

- **5-fold well-level GroupKFold cross-validation**.
- **Optuna hyperparameter tuning**.
- Development-set validation for ensemble-weight selection.
- An independent 130-well blind test for final evaluation.

Well-level grouping keeps observations from the same well within one fold, reducing leakage between training and validation wells.

Cross-validation and blind-test results are reported separately because they represent different evaluation settings.

---

## 5. Model Performance

### Independent Blind-Test Results

**Evaluation set: 130 wells**

| Model | RMSE (ft) |
|---|---:|
| LightGBM | 10.40 |
| CatBoost | 14.23 |
| **Hybrid Ensemble** | **9.70** |

The hybrid ensemble reduces blind-test RMSE by approximately **6.7%** compared with standalone LightGBM.

### High-Error Tail Comparison

| Model | 95th-percentile absolute error (ft) |
|---|---:|
| LightGBM | 22.5 |
| **Hybrid Ensemble** | **19.8** |

The ensemble reduces the 95th-percentile absolute error by **12%** relative to LightGBM.

This indicates an improvement in the evaluated high-error tail, but does not guarantee that every well or station improves.

### Cross-Validation Results

**Evaluation: 5-fold well-level GroupKFold**

| Model | Mean fold RMSE (ft) |
|---|---:|
| LightGBM | 11.05 |
| CatBoost | 14.26 |

These values are mean cross-validation fold RMSEs, not blind-test RMSEs.

A pooled out-of-fold RMSE and a mean fold RMSE are different summaries and should not be labeled interchangeably.

---

## 6. Example-Well Results

A selected example well contains **6,747 evaluated stations**.

| Evaluation scope | Hybrid Ensemble RMSE (ft) |
|---|---:|
| Overall example well | 4.47 |
| Lower Eagle Ford target zone | 4.35 |

### Formation-Specific Results

| Formation | Hybrid Ensemble RMSE (ft) |
|---|---:|
| BUDA | 1.98 |
| EGFDL — Lower Eagle Ford | 4.35 |
| EGFDU — Upper Eagle Ford | 3.64 |
| ASTNL | 5.65 |
| ANCU | 8.81 |
| ANCC | 8.46 |

These results describe the selected example well and should not be interpreted as dataset-wide performance.

### Target-Zone Classification

The reported ML-guided target-zone classification accuracy is **88.5%** under the evaluated setup.

This is a classification metric and is separate from formation-boundary TVD RMSE.

---

## 7. DTW Alignment Results

The reported alignment comparison shows:

| Alignment method | Pearson correlation coefficient (r) |
|---|---:|
| Baseline | 0.380 |
| DTW with Itakura constraint, maximum slope 5.1 | 0.776 |

The higher correlation indicates improved similarity between the evaluated aligned log signals.

This alignment metric is separate from formation-boundary prediction accuracy.

---

## 8. Metrics and Residual Visualization

### Root Mean Square Error

RMSE is calculated as:

**RMSE = √[(1 / N) × Σ(actual TVD − predicted TVD)²]**

For the reported boundary-prediction evaluation, squared residuals from the evaluated top and base TVD predictions are pooled.

`N` therefore represents the number of evaluated target values, which is not necessarily the number of unique stations.

### Residual Definition

**Residual = Actual TVD − Predicted TVD**

- Positive residual: actual TVD is greater than predicted TVD.
- Negative residual: actual TVD is smaller than predicted TVD.
- Zero residual: exact agreement.

### Residual Plot Conventions

Use a combined **Out-of-Fold Residual Distribution** plot with:

- Common histogram bin edges across all models.
- LightGBM shown in blue/cyan.
- CatBoost shown in orange.
- Hybrid Ensemble shown in green.
- A dashed reference line at zero error.
- Density units of **ft⁻¹**.

The area under each complete density distribution equals 1.

For readability, plots may display only **−40 to +40 ft**, but density normalization and RMSE must include all evaluated residuals, including those outside the displayed range.

---

## 9. Planned Interactive Pipeline

A proposed application extension uses:

- **FastAPI** for backend processing and prediction.
- **Streamlit** for an interactive dashboard.

### Proposed Application Flow

**CSV upload → QC → Feature generation → DTW alignment → Prediction → Visualization → Human review and export**

The dashboard would allow users to:

- Upload a well CSV.
- Inspect data-quality flags.
- Run the saved models.
- View GR alignment and predicted boundaries.
- Compare predictions with the measured trajectory.
- Record reviewer comments.
- Flag results for correction or recalibration.
- Export predictions and a review report.

Saved model artifacts should be packaged with the preprocessing rules, feature order, DTW configuration, model versions, and validated ensemble weight.

> This interface is a planned prototype extension. Historical station-by-station demonstrations should be labeled as replay or simulated streaming, not live field deployment.

---

## 10. Limitations

- Results reflect the evaluated Eagle Ford dataset and validation setup.
- Performance may differ for other formations, fields, logging conditions, or data distributions.
- The selected example well does not represent performance across all wells.
- DTW alignment quality alone does not establish geological correctness.
- Low RMSE does not guarantee reliable predictions at every station.
- Data-quality issues and changing geological conditions may require review and recalibration.
- Human approval remains necessary before predictions inform operational decisions.
- Field validation is required before claiming real-time operational effectiveness.

---

## 11. References

- **Dataset:** Kaggle Eagle Ford benchmark dataset.
- **Project repository:** https://github.com/shimaaelemam2012-cloud/Geosteering-in-horizontal-wells
- Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.*
- Prokhorenkova, L., et al. (2018). *CatBoost: Unbiased Boosting with Categorical Features.*
- Akiba, T., et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework.*

---

## Summary

The hybrid LightGBM–CatBoost ensemble achieves a **9.70 ft blind-test RMSE**, compared with **10.40 ft for LightGBM** and **14.23 ft for CatBoost**.

The workflow combines log alignment, causal feature preparation, machine-learning prediction, and human review to support formation-boundary interpretation in horizontal-well geosteering.

