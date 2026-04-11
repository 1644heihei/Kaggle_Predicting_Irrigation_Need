# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

This is a Kaggle competition project for **Predicting Irrigation Need**.
- **Target**: Binary classification for irrigation requirement
- **Data**: Structured tabular data with soil, weather, crop, and irrigation features
- **Evaluation**: Likely AUC-ROC (check submission page)
- **Approach**: Gradient boosting models, feature engineering, ensemble methods

## Directory Structure

```
Kaggle_Predicting_Irrigation_Need/
├── EXP/
│   ├── EXP001/
│   │   ├── train.py                (training script)
│   │   ├── infer.py                (inference script)
│   │   ├── config/
│   │   │   └── child-exp{N}.yaml   (experiment configs)
│   │   └── outputs/
│   │       └── child-exp{N}/       (results)
│   └── EXP_SUMMARY.md              (experiment log)
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── outputs/
│   └── CV_LB/
├── docs/
│   └── Idea_Research/
├── execute_train.ipynb             (main training notebook)
└── _CLAUDE.md                       (this file)
```

## Workflow

1. **Create child-exp config**: Add `config/child-exp{N}.yaml` under the appropriate EXP directory
2. **Run training**: Execute `execute_train.ipynb` with the config
3. **Check results**: Results saved to `outputs/child-exp{N}/`
4. **Log results**: Update `EXP_SUMMARY.md` with CV/LB scores
5. **Iterate**: Based on results, decide next experiment

## Config Structure (YAML)

```yaml
# child-exp000.yaml
exp_name: "EXP001"
child_exp_name: "child-exp000"
description: "Baseline LightGBM"

model_type: "lgbm"  # Options: lgbm, xgb, catboost, ensemble
model_params:
  num_leaves: 31
  learning_rate: 0.05
  n_estimators: 1000
  random_state: 42

feature_params:
  drop_features: []
  use_feature_engineering: false

cv_params:
  n_splits: 5
  random_state: 42
  stratify: true
```

## Important Rules

### MUST DO

1. **Use correct train/test split**: Always load from `data/train.csv` and `data/test.csv`
2. **Create output directory**: Before running, ensure `EXP/{EXP_NAME}/outputs/` exists
3. **Save predictions**: Always save OOF predictions to `outputs/child-exp{N}/oof_predictions.csv`
4. **Save scores**: Always save metrics to `outputs/child-exp{N}/results.json`
5. **Preserve config**: Copy `config/child-exp{N}.yaml` to `outputs/child-exp{N}/config.yaml`
6. **Use stratified k-fold**: For tabular data, always use StratifiedKFold for binary classification

### MUST NOT

1. **Do NOT hardcode paths**: Always use relative paths from workspace root
2. **Do NOT modify original data files** in `data/` directory
3. **Do NOT run without config**: Always pass config file as argument
4. **Do NOT mix CV and test predictions**: Keep them separate
5. **Do NOT forget random seeds**: Set seed for reproducibility

## Common Gotchas

1. **Memory issues**: If working with large datasets, use dtype optimization (int8/int16 for categories)
2. **Data leakage**: Never use test statistics for training preprocessing
3. **Class imbalance**: Check balance; may need `scale_pos_weight` or `class_weight`
4. **Feature types**: Distinguish between numerical, categorical, ordinal features
5. **Inference mismatch**: Ensure inference pipeline matches training pipeline exactly

## Feature Engineering Ideas

1. **Domain-aware features**:
   - Soil-weather interaction (pH × Temperature)
   - Seasonal irrigation patterns
   - Region-based soil characteristics

2. **Statistical features**:
   - Rolling statistics from time-series columns
   - Drought severity indices

3. **Tree-based feature importance**: Use permutation importance to select top features

## Model Selection Strategy

1. **Start**: Baseline with default LightGBM
2. **Then**: Try XGBoost and CatBoost separately
3. **Next**: Ensemble top 2-3 models
4. **Finally**: Hyperparameter tuning on best ensemble

## Evaluation & Metrics

- **Metric**: Check Kaggle submission for exact metric (likely AUC-ROC or F1)
- **CV strategy**: StratifiedKFold with 5 splits
- **Early stopping**: Monitor OOF score to detect overfitting
- **Public/Private split**: Reserve some data understanding for final submission

## When Adding New Experiments

- If major architecture change: Create new `EXP{N}` directory
- If just config/parameter change: Create new `child-exp{N}.yaml` in same `EXP{N}`
- Always increment experiment numbers sequentially

## Questions to Monitor

- Are CV and LB scores tracking similarly? (gap indicates overfitting)
- Is the model using all available features effectively?
- Which features contribute most to predictions?
- Is class balance maintained across folds?
