# Project: Boy or Girl - Data Science Mission
- version: 1.0.0
- date: 2026-03-05

## Core Architecture
- **Orchestrator**: Use `@data-scientist` for all high-level planning and experimental design.
- **Data Engineering**: Trigger `@ds-data-cleaner` for ingestion, null-handling, and schema validation.
- **Model Optimization**: Trigger `@ml-model-trainer` for hyperparameter search using Optuna.

## Active Standards
- All code must pass the `@implementation-playbook`.
- Performance must be validated against `@baseline-metrics.json`.

## Current Objective
Building a boy or girl prediction model for boy or girl data.