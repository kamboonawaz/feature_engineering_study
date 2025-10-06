# Feature Engineering Impact Study

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Repository: https://github.com/kamboonawaz/feature_engineering_study (create this repo then push).

Compare baseline vs engineered features on a regression dataset (Ames Housing) to quantify impact on model performance and interpretability.

## Objectives
- Baseline model using minimal preprocessing
- Enhanced feature engineering pipeline
- Quantify improvement (RMSE, R^2) and feature importance shifts
- Reproducible training & evaluation scripts

## Quick Start

Install deps & download data (if `make` is available):

```
make setup
make data
```

Run baseline & engineered experiments:

```
make baseline
make engineered
```

Outputs:
- Models: `models/{baseline,engineered}.joblib`
- Metrics log: `reports/metrics.json`

## Structure
```
src/
	data_download.py
	features.py
	train.py
configs/
	baseline.yaml
	engineered.yaml
notebooks/
	feature_engineering_showcase.ipynb
```

## Next Ideas
- Add SHAP analysis for feature importance shifts
- Add LightGBM tuned variant
- Add cross-validation pipeline

## Reproducibility
Set random_state in configs; pipeline deterministic aside from inherent multi-thread scheduling in random forests.

### If `make` is NOT installed (Windows default)
Run the steps manually:
```
python -m pip install -r requirements.txt
python src/data_download.py
python src/train.py --config configs/baseline.yaml
python src/train.py --config configs/engineered.yaml
```
