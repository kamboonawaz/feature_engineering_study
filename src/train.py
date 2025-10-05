import argparse
import json
from pathlib import Path
import sys
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib

# Ensure src/ is on path when executing as a script from project root
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from features import get_feature_pipeline  # noqa: E402

MODEL_MAP = {
    'RandomForestRegressor': RandomForestRegressor,
    'GradientBoostingRegressor': GradientBoostingRegressor,
}


def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def train(config_path: str):
    cfg = load_config(config_path)
    df = pd.read_csv('data/processed/ames_clean.csv')
    y = df['SalePrice']
    X = df.drop(columns=['SalePrice'])

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=cfg.get('random_state', 42)
    )

    pipe = get_feature_pipeline(cfg['feature_pipeline'], df)

    ModelCls = MODEL_MAP[cfg['model']['type']]
    model = ModelCls(**cfg['model'].get('params', {}))

    # Fit feature pipeline
    X_train_t = pipe.fit_transform(X_train, y_train)
    X_valid_t = pipe.transform(X_valid)

    model.fit(X_train_t, y_train)

    preds = model.predict(X_valid_t)
    rmse = mean_squared_error(y_valid, preds, squared=False)
    r2 = r2_score(y_valid, preds)

    metrics = {
        'experiment': cfg['experiment_name'],
        'rmse': rmse,
        'r2': r2,
        'model_type': cfg['model']['type']
    }
    print(metrics)

    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    joblib.dump({'model': model, 'pipeline': pipe}, models_dir / f"{cfg['experiment_name']}.joblib")

    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)

    metrics_path = reports_dir / 'metrics.json'
    all_metrics = []
    if metrics_path.exists():
        try:
            all_metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            pass
    all_metrics.append(metrics)
    metrics_path.write_text(json.dumps(all_metrics, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    train(args.config)
