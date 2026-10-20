import copy
import json

import numpy as np
import pandas as pd

from config import FEATURES_PATH, PROCESSED_DIR, RESULTS_DIR, SPLIT_PATH
from src.airbnb_pricing.baselines import CATEGORICAL, as_model_input, constant_baseline, linear_baseline
from src.airbnb_pricing.models import catboost, lightgbm, random_forest, tuned, xgboost
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


def predict(model, train: pd.DataFrame, test: pd.DataFrame, folds: list) -> tuple[np.ndarray, np.ndarray]:
    """THB predictions: out-of-fold for every train row (grouped 5-fold) and one fit-on-train for test."""
    X_tr, y_tr = as_model_input(train), train['price'].to_numpy()
    oof = np.zeros(len(train))
    for tr, va in folds:
        oof[va] = fit_predict(model, X_tr.iloc[tr], y_tr[tr], X_tr.iloc[va])
    return oof, fit_predict(model, X_tr, y_tr, as_model_input(test))


def score(oof: np.ndarray, test_pred: np.ndarray, train: pd.DataFrame, test: pd.DataFrame, folds: list) -> dict:
    """CV: mean/sd of per-fold metrics. Test: metrics on the held-out hosts."""
    y_tr = train['price'].to_numpy()
    per_fold = [metrics(y_tr[va], oof[va]) for _, va in folds]
    cv = {k: {'mean': float(np.mean([f[k] for f in per_fold])), 'sd': float(np.std([f[k] for f in per_fold]))}
          for k in per_fold[0]}
    return {'cv': cv, 'test': metrics(test['price'].to_numpy(), test_pred)}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(FEATURES_PATH).merge(pd.read_parquet(SPLIT_PATH), on='id')
    df[CATEGORICAL] = df[CATEGORICAL].astype('category')  # vocabulary only, so folds/test share dtype (LightGBM)
    train = df[df['set'] == 'train'].drop(columns='set').reset_index(drop=True)
    test = df[df['set'] == 'test'].drop(columns='set').reset_index(drop=True)
    return train, test


if __name__ == '__main__':
    train, test = load_data()
    folds = list(cv_folds(train))
    models = {'B0_constant_median': constant_baseline(), 'B1_linear_regression': linear_baseline(as_model_input(train)),
              'random_forest': random_forest(), 'lightgbm': lightgbm(),
              'xgboost': xgboost(), 'catboost': catboost(),
              'lightgbm_tuned': tuned('lgbm'), 'xgboost_tuned': tuned('xgb'), 'catboost_tuned': tuned('cat')}
    preds = {name: predict(m, train, test, folds) for name, m in models.items()}

    # ensemble: mean of the four tree models' log-predictions (geometric mean, same scale the models were fit on)
    members = ['random_forest', 'lightgbm', 'xgboost', 'catboost']
    preds['ensemble_trees'] = tuple(np.exp(np.mean([np.log(preds[m][i]) for m in members], axis=0)) for i in (0, 1))

    # ensemble of the three tuned boosters
    preds['ensemble_tuned'] = tuple(np.exp(np.mean([np.log(preds[m][i]) for m in ('lightgbm_tuned', 'xgboost_tuned', 'catboost_tuned')], axis=0)) for i in (0, 1))

    out = {name: score(oof, te, train, test, folds) for name, (oof, te) in preds.items()}
    (RESULTS_DIR / 'metrics.json').write_text(json.dumps(out, indent=2))
    # per-listing predictions stay local (privacy): data/processed is gitignored; bootstrap reads this
    pd.DataFrame({'id': test['id'], 'host_id': test['host_id'], 'price': test['price'],
                  **{name: te for name, (_, te) in preds.items()}}).to_parquet(PROCESSED_DIR / 'test_preds.parquet', index=False)
    for name, r in out.items():
        print(name, 'CV MAE', round(r['cv']['MAE_THB']['mean']), 'test MAE', round(r['test']['MAE_THB']),
              'test R2_log', round(r['test']['R2_log'], 3), 'within20', round(r['test']['within20'], 3))
