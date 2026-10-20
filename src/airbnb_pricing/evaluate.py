import copy
import json

import numpy as np
import pandas as pd

from config import FEATURES_PATH, RESULTS_DIR, SPLIT_PATH
from src.airbnb_pricing.baselines import CATEGORICAL, as_model_input, constant_baseline, linear_baseline
from src.airbnb_pricing.models import catboost, lightgbm, random_forest, xgboost
from src.airbnb_pricing.split import cv_folds


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """y in THB. Models predict log(price); callers exp() the output first (conditional median)."""
    err = y_true - y_pred
    ape = np.abs(err) / y_true
    log_t, log_p = np.log(y_true), np.log(y_pred)
    return {
        'MAE_THB': float(np.mean(np.abs(err))),
        'RMSE_THB': float(np.sqrt(np.mean(err ** 2))),
        'MAPE': float(np.mean(ape)),
        'MdAPE': float(np.median(ape)),
        'within20': float(np.mean(ape <= 0.2)),
        'R2_log': float(1 - np.sum((log_t - log_p) ** 2) / np.sum((log_t - log_t.mean()) ** 2)),
    }


def fit_predict(model, X_tr, y_tr, X_te) -> np.ndarray:
    return np.exp(copy.deepcopy(model).fit(X_tr, np.log(y_tr)).predict(X_te))


def evaluate(model, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Grouped 5-fold CV on train (mean/sd per metric) plus one fit-on-train score on test."""
    X_tr, y_tr = as_model_input(train), train['price'].to_numpy()
    folds = [metrics(y_tr[va], fit_predict(model, X_tr.iloc[tr], y_tr[tr], X_tr.iloc[va]))
             for tr, va in cv_folds(train)]
    cv = {k: {'mean': float(np.mean([f[k] for f in folds])), 'sd': float(np.std([f[k] for f in folds]))}
          for k in folds[0]}
    y_te = test['price'].to_numpy()
    return {'cv': cv, 'test': metrics(y_te, fit_predict(model, X_tr, y_tr, as_model_input(test)))}


if __name__ == '__main__':
    df = pd.read_parquet(FEATURES_PATH).merge(pd.read_parquet(SPLIT_PATH), on='id')
    df[CATEGORICAL] = df[CATEGORICAL].astype('category')  # vocabulary only, so folds/test share dtype (LightGBM)
    train = df[df['set'] == 'train'].drop(columns='set').reset_index(drop=True)
    test = df[df['set'] == 'test'].drop(columns='set').reset_index(drop=True)
    models = {'B0_constant_median': constant_baseline(), 'B1_linear_regression': linear_baseline(as_model_input(train)),
              'random_forest': random_forest(), 'lightgbm': lightgbm(),
              'xgboost': xgboost(), 'catboost': catboost()}
    out = {name: evaluate(m, train, test) for name, m in models.items()}
    (RESULTS_DIR / 'metrics.json').write_text(json.dumps(out, indent=2))
    for name, r in out.items():
        print(name, 'CV MAE', round(r['cv']['MAE_THB']['mean']), 'test MAE', round(r['test']['MAE_THB']),
              'test R2_log', round(r['test']['R2_log'], 3), 'within20', round(r['test']['within20'], 3))
