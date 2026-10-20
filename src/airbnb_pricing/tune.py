import json
import os
import sys

import numpy as np
import optuna

from config import RESULTS_DIR, SEED
from src.airbnb_pricing.baselines import as_model_input
from src.airbnb_pricing.evaluate import load_data
from src.airbnb_pricing.models import FACTORIES
from src.airbnb_pricing.split import cv_folds

N_TRIALS = 40
TIMEOUT_S = int(os.environ.get('TUNE_TIMEOUT_S', 1200))  # hard cap per model, 20 minutes by default


def suggest(name: str, t) -> dict:
    """Search spaces. LightGBM follows the plan (M6.3); XGBoost/CatBoost use the analogous knobs."""
    if name == 'lgbm':
        return {'num_leaves': t.suggest_int('num_leaves', 15, 127),
                'learning_rate': t.suggest_float('learning_rate', 0.02, 0.1, log=True),
                'min_child_samples': t.suggest_int('min_child_samples', 10, 100),
                'colsample_bytree': t.suggest_float('colsample_bytree', 0.5, 1.0),  # feature_fraction
                'subsample': t.suggest_float('subsample', 0.5, 1.0),                # bagging_fraction
                'reg_alpha': t.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': t.suggest_float('reg_lambda', 0, 10)}
    if name == 'xgb':
        return {'max_depth': t.suggest_int('max_depth', 3, 10),
                'learning_rate': t.suggest_float('learning_rate', 0.02, 0.1, log=True),
                'min_child_weight': t.suggest_int('min_child_weight', 1, 20),
                'colsample_bytree': t.suggest_float('colsample_bytree', 0.5, 1.0),
                'subsample': t.suggest_float('subsample', 0.5, 1.0),
                'reg_alpha': t.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': t.suggest_float('reg_lambda', 0, 10)}
    return {'depth': t.suggest_int('depth', 4, 10),
            'learning_rate': t.suggest_float('learning_rate', 0.02, 0.1, log=True),
            'l2_leaf_reg': t.suggest_float('l2_leaf_reg', 1, 10),
            'random_strength': t.suggest_float('random_strength', 0, 5),
            'bagging_temperature': t.suggest_float('bagging_temperature', 0, 1)}


def objective(trial, name, X, y_log, folds) -> float:
    """Mean RMSE on log(price) over grouped CV folds of train. The test set is never touched."""
    params = suggest(name, trial)
    rmse = []
    for tr, va in folds:
        m = FACTORIES[name](**params).fit(X.iloc[tr], y_log[tr])
        rmse.append(np.sqrt(np.mean((y_log[va] - m.predict(X.iloc[va])) ** 2)))
    return float(np.mean(rmse))


if __name__ == '__main__':
    name = sys.argv[1]  # lgbm | xgb | cat
    train, _ = load_data()
    X, y_log, folds = as_model_input(train), np.log(train['price'].to_numpy()), list(cv_folds(train))

    db = RESULTS_DIR / f'optuna_{name}.db'
    db.unlink(missing_ok=True)  # fresh study each run, so reruns reproduce
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED),
                                storage=f'sqlite:///{db}', study_name=name)
    study.optimize(lambda t: objective(t, name, X, y_log, folds), n_trials=N_TRIALS, timeout=TIMEOUT_S)

    (RESULTS_DIR / f'{name}_params.json').write_text(json.dumps(study.best_params, indent=2))
    study.trials_dataframe().to_csv(RESULTS_DIR / f'optuna_trials_{name}.csv', index=False)
    print(f'{name}: {len(study.trials)} trials, best CV RMSE(log) {study.best_value:.4f}')
    print(study.best_params)
