# Data Science & ML Implementation Playbook (Gold Standard)

This document defines the mandatory engineering standards for all Python, Machine Learning, and Data Science tasks within this workspace.

name: implementation-playbook
source: gemini
version: 1.0.0
Date: 2026-03-05
risk: medium
---

## 1. Python Engineering Standards
### Structure & Style
- **PEP 8**: All code must strictly adhere to PEP 8.
- **Type Hinting**: Use Python type hints (`typing` module) for all function signatures.
- **Docstrings**: Use Google-style docstrings for all modules, classes, and functions.
- **Modularity**: Logic must be separated into `src/data`, `src/models`, and `src/features`. No "God Scripts."

### Error Handling
- Never use bare `except:`.
- Use custom exception classes for domain-specific errors (e.g., `DataValidationError`).
- Implement logging using the `logging` library; avoid `print()` in production-ready scripts.

---

## 2. Data Science Workflow
### Data Integrity
- **Immutability**: Never overwrite raw data. Always version processed datasets (e.g., `data/processed/v1.parquet`).
- **Validation**: Use `pandera` or basic `assert` statements to verify schema and data types after every transformation.
- **Efficiency**: Prefer `polars` or `vectorized pandas` over manual loops.

### Exploratory Data Analysis (EDA)
- Every EDA must produce a summary artifact containing:
  - Missing value heatmaps.
  - Correlation matrices.
  - Distribution plots for target variables.

---

## 3. Machine Learning Excellence
### Experiment Tracking
- All training runs must be logged. If no remote server exists, log to a local `experiments.csv`.
- Mandatory metadata: Model version, Hyperparameters, Git hash, and Hardware used.

### Model Evaluation
- **Baselines**: A "Dummy Regressor/Classifier" baseline must be established before training complex models.
- **Metrics**: 
  - Classification: Precision-Recall curves are preferred over ROC for imbalanced data.
  - Regression: Include MAE and Max Error alongside RMSE.
- **Interpretability**: For every final model, generate a SHAP summary plot to justify feature importance.

---

## 4. ML Ops & Deployment
### Reproducibility
- **Environment**: All projects must include a `requirements.txt` or `pyproject.toml`.
- **Seeding**: Global seeds (`numpy`, `torch`, `random`) must be set to `42` unless otherwise specified.

### Serialization
- Use `joblib` for Scikit-Learn.
- Use `safetensors` for Deep Learning weights.
- Include a `metadata.json` alongside every saved model containing versioning info.

---

## 5. Verification Checklist (The "Final Gate")
Before declaring a task "Complete," the agent must verify:
1. [ ] Code passes `flake8` or `black` formatting.
2. [ ] No data leakage (training data does not contain info from the future/test set).
3. [ ] Model performance beats the established baseline.
4. [ ] Documentation includes a "How to Run" section.