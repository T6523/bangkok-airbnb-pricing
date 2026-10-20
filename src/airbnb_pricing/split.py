import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from config import FEATURES_PATH, N_FOLDS, SEED, SPLIT_PATH, TEST_SIZE


def make_split(df: pd.DataFrame) -> pd.DataFrame:
    """Train/test split grouped by host_id: a host's listings never straddle the split."""
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(gss.split(df, groups=df['host_id']))
    out = df[['id']].copy()
    out['set'] = 'train'
    out.iloc[test_idx, out.columns.get_loc('set')] = 'test'
    assert not set(df.iloc[train_idx]['host_id']) & set(df.iloc[test_idx]['host_id'])
    return out


def cv_folds(train: pd.DataFrame):
    """Deterministic grouped 5-fold on train only"""
    return GroupKFold(n_splits=N_FOLDS).split(train, groups=train['host_id'])


if __name__ == '__main__':
    df = pd.read_parquet(FEATURES_PATH)
    split = make_split(df)
    split.to_parquet(SPLIT_PATH, index=False)
    print(split['set'].value_counts())
