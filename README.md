# readmission-risk-service

Predicts 30-day hospital readmission from public encounter data, then serves the model with calibrated probabilities, drift monitoring, and CI/CD.

**Status:** week 1 scaffold. Ingest, leakage guard, baseline model, CI.

## Data

[UCI Diabetes 130-US Hospitals (1999-2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008). Patient-level encounters, public, no PHI.

## Baseline

Logistic regression, class-balanced, median/mode imputation, one-hot encoding. Split by `encounter_id` order as a time proxy. `discharge_disposition_id` is dropped because it leaks the outcome: expired and hospice discharges cannot be readmitted.

| Metric | Value |
|---|---|
| AUROC | pending |
| AUPRC | pending |
| Brier | pending |
| Prevalence | pending |

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
make install
make train
make test
```

## Layout

```
src/readmission/   data loading, features, model, metrics
scripts/           training entrypoints
tests/             unit tests
```

## Roadmap

- Week 2: XGBoost, calibration, ablation, model card
- Week 3: FastAPI service, tests, Docker
- Week 4: Cloud Run deploy, BigQuery logging, CI/CD, load test
- Week 5: drift monitoring, data-quality checks, retraining
- Week 6: README polish, architecture diagram, demo
