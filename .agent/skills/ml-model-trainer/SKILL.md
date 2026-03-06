---
name: ml-model-trainer
description: Expert skill for hyperparameter optimization and model selection. Use for Optuna studies, Ray Tune, and cross-validation setups.
risk: high
source: gemini
version: 1.0.0
date_added: '2026-03-05'
---

# Model Tuning Protocol

## Instructions
1. **Search Space Definition**: Define explicit ranges for hyperparameters (e.g., `trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)`).
2. **Pruning**: Always implement an Optuna pruner (e.g., `MedianPruner`) to kill underperforming trials early and save compute.
3. **Reproducibility**: Fix the `seed` for both the Sampler and the Model.
4. **Logging**: Integrate `mlflow` or a local `trials.csv` logger to track every iteration’s parameters and metrics.
5. **Evaluation**: Use Stratified K-Fold cross-validation for imbalanced datasets.

## Constraints
- Never run more than 100 trials without checking in with the user on "Time vs. Accuracy" trade-offs.
- Do not suggest `GridSearchCV` for spaces with >3 parameters; insist on `@ml-model-trainer` (Bayesian) methods.

## Example Interaction
"Apply @ml-model-trainer to the XGBoost model in `train.py`. Aim for 50 trials using the 'f1-macro' metric."