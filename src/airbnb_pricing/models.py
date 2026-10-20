import json

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor

from config import RESULTS_DIR, SEED
from src.airbnb_pricing.baselines import CATEGORICAL


def random_forest():
    """Ordinal-encoded categoricals; NaNs go straight to the forest (supported natively)."""
    enc = ColumnTransformer(
        [('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CATEGORICAL)],
        remainder='passthrough',
    )
    return make_pipeline(enc, RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=SEED))


def lightgbm(**params):
    """Untuned defaults unless overridden; CATEGORICAL columns must arrive as pandas 'category' dtype."""
    kw = dict(n_estimators=500, learning_rate=0.05, num_leaves=63, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              random_state=SEED, n_jobs=-1, verbose=-1)
    return LGBMRegressor(**{**kw, **params})


def xgboost(**params):
    """Same budget as lightgbm(); native categorical support via pandas 'category' dtype."""
    kw = dict(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
              tree_method='hist', enable_categorical=True, random_state=SEED, n_jobs=-1)
    return XGBRegressor(**{**kw, **params})


def catboost(**params):
    """Ordered target statistics on CATEGORICAL columns; same iterations/learning rate as the others."""
    kw = dict(iterations=500, learning_rate=0.05, depth=6, cat_features=CATEGORICAL,
              random_seed=SEED, thread_count=-1, verbose=0)
    return CatBoostRegressor(**{**kw, **params})


FACTORIES = {'lgbm': lightgbm, 'xgb': xgboost, 'cat': catboost}


def tuned(name: str):
    """Model `name` (lgbm|xgb|cat) with the Optuna best params from results/<name>_params.json."""
    return FACTORIES[name](**json.loads((RESULTS_DIR / f'{name}_params.json').read_text()))
