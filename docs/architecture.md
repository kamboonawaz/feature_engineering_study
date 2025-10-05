## Architecture & Design

### Goal
Quantify the impact of targeted feature engineering on Ames Housing sale price prediction versus a minimal baseline.

### Data Flow
1. Download raw dataset from OpenML (Ames Housing) via `data_download.py`.
2. Persist raw CSV under `data/raw/` and a light cleaned version under `data/processed/`.
3. Training script loads processed data, splits (80/20), builds a feature pipeline based on config.
4. Pipeline transforms features -> model trains -> metrics appended to `reports/metrics.json`.
5. Evaluation script assembles comparison table and computes % improvement.

### Components
| Component | File | Responsibility |
|----------|------|----------------|
| Data Ingestion | `src/data_download.py` | Fetch & store raw + cleaned data |
| Feature Pipelines | `src/features.py` | Baseline & engineered transformations |
| Training Orchestration | `src/train.py` | Config-driven training & metric logging |
| Evaluation | `src/evaluate.py` | Aggregate metrics & compute improvement |
| Configurations | `configs/*.yaml` | Experiment parameters & pipeline selection |

### Feature Engineering (Engineered Variant)
Added derived features:
- `TotalBath = FullBath + 0.5 * HalfBath`
- `TotalSF = 1stFlrSF + 2ndFlrSF + TotalBsmtSF`
- `GarageQualityRatio = GarageArea / (GarageCars + 1)`
Plus safe log1p transform for strictly positive numeric variables and ordinal encoding for categoricals.

### Modeling Choices
- Baseline: RandomForestRegressor (robust, minimal tuning needed)
- Engineered: GradientBoostingRegressor (sensitive to feature quality; showcases benefit)

### Metrics
- Root Mean Squared Error (RMSE)
- R² Score
- Relative RMSE improvement computed as (Baseline - Engineered)/Baseline * 100

### Current Results (See `reports/summary.txt` after running)
Example run produced ~3% RMSE improvement with engineered features.

### Reproducibility
- Pinned dependencies in `requirements.txt`
- Config-driven experiments (YAML)
- Random seeds set where applicable

### Possible Extensions
- Cross-validation (KFold) for more stable estimates
- Add LightGBM & XGBoost configs and compare vs GradientBoosting
- SHAP / permutation importance analysis (`src/interpret.py` future)
- Hyperparameter sweep script producing leaderboard CSV
- Save feature matrix schema JSON for downstream consumption

### Risks / Limitations
- Single random split may not generalize -> address with CV
- Engineered features are illustrative, not exhaustive
- No hyperparameter tuning yet; improvement margin modest

### Roadmap
| Priority | Task | Rationale |
|----------|------|-----------|
| P1 | Add cross-validation training mode | Stabilize metrics |
| P1 | SHAP importance report | Interpretability for portfolio |
| P2 | LightGBM + tuned params | Potential larger gain |
| P2 | Add CI workflow (lint + tests) | Engineering maturity |
| P3 | Feature selection experiment | Quantify redundancy |

---
Generated: Initial version (auto populated).
# Architecture

Simple pipeline:

1. Download & cache dataset (`data_download.py`).
2. Train baseline and engineered variants (`train.py`).
3. Persist models (`models/`).
4. Aggregate metrics (`reports/metrics.json`).
5. Compare & summarize (`evaluate.py`).

Future extensions:
- Add SHAP analysis
- Add hyperparameter tuning loop
- Add model registry promotion script
