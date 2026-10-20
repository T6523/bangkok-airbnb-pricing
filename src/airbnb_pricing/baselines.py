import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ['neighbourhood_cleansed', 'room_type', 'property_type_grouped']
NOT_FEATURES = ['id', 'host_id', 'price'] 


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NOT_FEATURES]


def as_model_input(df: pd.DataFrame) -> pd.DataFrame:
    """Feature columns only; booleans become 0/1 so imputers and scalers accept them."""
    X = df[feature_columns(df)].copy()
    bools = X.select_dtypes('bool').columns
    X[bools] = X[bools].astype(float)
    return X


def constant_baseline() -> DummyRegressor:
    """B0: predicts the training median of log(price) for every listing."""
    return DummyRegressor(strategy='median')


def linear_baseline(X: pd.DataFrame):
    """B1: plain linear regression on log(price). Median-impute + missing flags, scale, one-hot."""
    numeric = [c for c in X.columns if c not in CATEGORICAL]
    prep = ColumnTransformer([
        ('num', make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler()), numeric),
        # categories seen <30 times are pooled so rare neighbourhoods don't get their own coefficient
        ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=30), CATEGORICAL),
    ])
    return make_pipeline(prep, LinearRegression())
