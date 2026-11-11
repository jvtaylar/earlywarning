# Engineering Mathematics Early Warning System – Research Prototype

## Contents
- `app.py`: Streamlit instructor dashboard.
- `models/`: fitted Logistic Regression, Random Forest, and XGBoost pipelines.
- `model_metrics.csv`: holdout evaluation metrics.
- `test_predictions.csv`: test-set predictions and predicted probabilities.
- `instructor_risk_register.csv`: Week 8 risk register and intervention suggestions.
- `feature_importance.csv`: absolute standardized LR coefficients and tree feature importance.
- `shap_global_importance.csv`: global mean absolute SHAP values for Random Forest and XGBoost.
- `shap_values_*.csv`: SHAP values for the explained test sample.
- `figures/`: model comparison, ROC, feature importance, and SHAP summary plots.
- `train_pipeline.py`: reproducible training, evaluation, and SHAP analysis.
- `requirements.txt`: Python dependencies.

## Experiment design
- Snapshot: Week 8 (predictors available by the end of Week 8).
- Training: AY 2024-2025.
- Testing: AY 2025-2026.
- Target: `At_Risk` (1 = synthetic failed outcome, 0 = synthetic passed outcome).
- Features: academic, LMS, attendance, and derived indicators available by Week 8.
- Excluded: student identifier, observation date, final grade, final status, risk level, target leakage fields, and the synthetic latent engagement index.
- Classification threshold for reported metrics: 0.50.
- Risk bands for the dashboard: Low < 0.30; Moderate 0.30–<0.60; High >= 0.60. These are illustrative operational thresholds, not validated cutoffs.

## Run the dashboard
1. Install Python 3.10 or later.
2. Install dependencies: `pip install -r requirements.txt`
3. Put `engineering_math_early_warning_dummy.csv` in this folder.
4. Run: `streamlit run app.py`

The dashboard uses the generated `instructor_risk_register.csv`, metrics, feature-importance and SHAP summaries. It is a demonstration interface; it does not connect to a live LMS.

## Re-run model training
Place `engineering_math_early_warning_dummy.csv` in the project folder, then run:
`python train_pipeline.py --data engineering_math_early_warning_dummy.csv`

## Important research limitations
1. All records are synthetic. Performance values are not evidence of real institutional performance.
2. The target labels were artificially generated using related academic and engagement variables. This can create strong feature-target relationships.
3. The chronological holdout is by academic year, but a student may occur in more than one year. For generalization to unseen students, add a student-grouped holdout. For deployment forecasting, retain a strictly chronological split.
4. Risk categories and intervention rules are provisional and must be calibrated with institutional policy, validation data, and instructor input.
5. SHAP explains model associations, not causal effects.
6. Obtain institutional approval, follow privacy requirements, minimize identifiable information, and ensure instructors review alerts before acting.

## Recommended next validation
- Re-evaluate with real records from multiple cohorts.
- Compare chronological and student-grouped validation.
- Report confidence intervals and calibration (Brier score / calibration curve).
- Consider precision-recall AUC and threshold selection based on the cost of false negatives.
- Audit performance by course and relevant student groups, with privacy safeguards.
- Conduct instructor usability and intervention-utility assessment.
