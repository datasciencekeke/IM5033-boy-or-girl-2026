---
description: End-to-End ML Pipeline
---

# Workflow: End-to-End ML Pipeline
## Steps
1. **Audit**: Use `@ds-data-cleaner` to check `data/raw/` for outliers and missing values.
2. **Preprocess**: Generate a cleaning script based on the audit.
3. **Baseline**: Use `@data-scientist` to establish a simple Logistic Regression baseline.
4. **Tune**: Use `@ml-model-trainer` to run 50 trials of XGBoost.
5. **Verify**: Use `@implementation-playbook` to ensure the final model is serialized correctly.