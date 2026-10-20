import json

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from config import FEATURES_PATH, ROOT, SEED, SPLIT_PATH
from src.airbnb_pricing.evaluate import metrics
from src.airbnb_pricing.split import cv_folds

DOCS = ROOT / 'docs'
MIN_HOOD_N = 30  # neighbourhoods with fewer training listings are not offered in the dropdown
GRID = [(t, l) for t in (100, 200, 300) for l in (8, 16, 32)]  # (trees, leaves)
TOLERANCE = 1.01  # smallest model within 1% of the best CV MAE
MAX_KB = 500
HOOD_DROPOUT = 0.3  # share of training rows with the neighbourhood hidden, so 'unspecified' means unknown
N_PRICE_BINS = 25  # log-spaced between the price scope edges
MIN_CELL_N = 5  # heatmap cells with fewer listings are shown as empty
# row inputs for the heatmap: (label, upper edges, bin labels); a value goes to the first bin whose edge it does not exceed
DIST_DIMS = {
    'bedrooms': ('Bedrooms', [0, 1, 2, 3], ['0', '1', '2', '3', '4+']),
    'accommodates': ('Guests', [1, 2, 3, 4, 6], ['1', '2', '3', '4', '5-6', '7+']),
    'bathrooms': ('Bathrooms', [1, 1.5, 2], ['1 or less', '1.5', '2', '2.5+']),
    'minimum_nights': ('Minimum nights', [1, 6, 29], ['1', '2-6', '7-29', '30+']),
}


def build_model(trees: int, leaves: int) -> LGBMRegressor:
    return LGBMRegressor(n_estimators=trees, num_leaves=leaves, learning_rate=0.05, min_child_samples=30,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
                         random_state=SEED, n_jobs=-1, verbose=-1)


def fit(trees: int, leaves: int, X: pd.DataFrame, y: np.ndarray) -> LGBMRegressor:
    """Fit on log price with neighbourhood dropout (same seed every call)."""
    X = X.copy()
    X.loc[np.random.default_rng(SEED).random(len(X)) < HOOD_DROPOUT, 'neighbourhood'] = np.nan
    return build_model(trees, leaves).fit(X, np.log(y))


def inputs(df: pd.DataFrame, hood_code: dict, room_code: dict) -> pd.DataFrame:
    """The page's inputs. Neighbourhood is optional: anything outside the dropdown is NaN (missing)."""
    return pd.DataFrame({'bedrooms': df['bedrooms'], 'accommodates': df['accommodates'],
                         'minimum_nights': df['minimum_nights'], 'bathrooms': df['bathrooms'],
                         'room_type': df['room_type'].map(room_code),
                         'neighbourhood': df['neighbourhood_cleansed'].map(hood_code),
                         'pool': df['am_pool']}).reset_index(drop=True)


def compact(node: dict) -> dict:
    if 'leaf_value' in node:
        return {'v': node['leaf_value']}
    assert node['decision_type'] == '<='
    return {'f': node['split_feature'], 't': node['threshold'], 'd': int(node['default_left']),
            'n': int(node['missing_type'] == 'NaN'), 'l': compact(node['left_child']), 'r': compact(node['right_child'])}


def export_trees(model: LGBMRegressor) -> list:
    return [compact(t['tree_structure']) for t in model.booster_.dump_model()['tree_info']]


if __name__ == '__main__':
    df = pd.read_parquet(FEATURES_PATH).merge(pd.read_parquet(SPLIT_PATH), on='id')
    train = df[df['set'] == 'train'].reset_index(drop=True)
    test = df[df['set'] == 'test'].reset_index(drop=True)

    # categorical inputs coded 0.. by ascending median train price so trees can split on a threshold
    counts = train['neighbourhood_cleansed'].value_counts()
    keep = counts[counts >= MIN_HOOD_N].index
    order = train[train['neighbourhood_cleansed'].isin(keep)].groupby('neighbourhood_cleansed')['price'].median().sort_values()
    hood_code = {name: i for i, name in enumerate(order.index)}
    room_code = {name: i for i, name in enumerate(train.groupby('room_type')['price'].median().sort_values().index)}

    X_tr, X_te = inputs(train, hood_code, room_code), inputs(test, hood_code, room_code)
    y_tr, y_te = train['price'].to_numpy(), test['price'].to_numpy()
    folds = list(cv_folds(train))

    rows = []
    for trees, leaves in GRID:
        oof, oof_h = np.zeros(len(train)), np.zeros(len(train))
        for tr, va in folds:
            mdl = fit(trees, leaves, X_tr.iloc[tr], y_tr[tr])
            oof[va] = np.exp(mdl.predict(X_tr.iloc[va]))
            oof_h[va] = np.exp(mdl.predict(X_tr.iloc[va].assign(neighbourhood=np.nan)))
        kb = len(json.dumps(export_trees(fit(trees, leaves, X_tr, y_tr)), separators=(',', ':'))) / 1024
        rows.append({'trees': trees, 'leaves': leaves, 'cv_mae': np.mean(np.abs(y_tr - oof)),
                     'cv_mae_hidden': np.mean(np.abs(y_tr - oof_h)), 'kb': kb, 'oof': oof, 'oof_h': oof_h})
    grid = pd.DataFrame(rows)
    ok = grid[grid['kb'] <= MAX_KB]
    pick = ok[ok['cv_mae'] <= TOLERANCE * ok['cv_mae'].min()].sort_values('kb').iloc[0]
    print(grid.drop(columns=['oof', 'oof_h']).round(1).to_string(index=False))
    print(f"picked trees={int(pick['trees'])} leaves={int(pick['leaves'])} (CV MAE {pick['cv_mae']:.0f}, best {ok['cv_mae'].min():.0f}, ~{pick['kb']:.0f} KB)")

    # 80% ranges from out-of-fold residuals on train (test is not used to set them), one for each neighbourhood case
    q10, q90 = np.quantile(np.log(y_tr / pick['oof']), [0.1, 0.9])
    h10, h90 = np.quantile(np.log(y_tr / pick['oof_h']), [0.1, 0.9])
    model = fit(int(pick['trees']), int(pick['leaves']), X_tr, y_tr)
    pred = np.exp(model.predict(X_te))
    pred_h = np.exp(model.predict(X_te.assign(neighbourhood=np.nan)))
    m, m_h = metrics(y_te, pred), metrics(y_te, pred_h)
    cover = lambda p, a, b: float(np.mean((y_te >= p * np.exp(a)) & (y_te <= p * np.exp(b))))

    # price distribution per room type and row input: counts of real listings per log-spaced price bin
    # (aggregates only; cells under MIN_CELL_N are zeroed, row totals stay exact)
    allr = pd.concat([train, test], ignore_index=True)
    edges = np.geomspace(allr['price'].min(), allr['price'].max() + 1, N_PRICE_BINS + 1)
    allr['pbin'] = np.clip(np.searchsorted(edges, allr['price'], side='right') - 1, 0, N_PRICE_BINS - 1)
    counts = {}
    for k, (_, uppers, bins) in DIST_DIMS.items():
        allr['rbin'] = pd.cut(allr[k], [-np.inf] + uppers + [np.inf], labels=False)
        counts[k] = {}
        for room, code in room_code.items():
            sub = allr[(allr['room_type'] == room) & allr['rbin'].notna()]
            grid = np.zeros((len(bins), N_PRICE_BINS), dtype=int)
            for (r, pb), n in sub.groupby(['rbin', 'pbin']).size().items():
                grid[int(r), int(pb)] = n
            totals = grid.sum(axis=1)
            grid[grid < MIN_CELL_N] = 0
            counts[k][str(code)] = {'n': totals.tolist(), 'c': grid.tolist()}
    dist = {'edges': [round(float(e), 1) for e in edges], 'min_cell': MIN_CELL_N, 'counts': counts,
            'dims': {k: {'label': v[0], 'uppers': v[1], 'bins': v[2]} for k, v in DIST_DIMS.items()}}

    out = {'features': list(X_tr.columns), 'hoods': hood_code, 'rooms': room_code, 'dist': dist,
           'range': {'lo': float(np.exp(q10)), 'hi': float(np.exp(q90))},
           'range_no_hood': {'lo': float(np.exp(h10)), 'hi': float(np.exp(h90))},
           'trees': export_trees(model),
           'meta': {'n_trees': int(pick['trees']), 'n_leaves': int(pick['leaves']), 'cv_mae': float(pick['cv_mae']),
                    'test_mae': m['MAE_THB'], 'test_within20': m['within20'], 'test_r2_log': m['R2_log'],
                    'range_coverage_test': cover(pred, q10, q90),
                    'test_mae_no_hood': m_h['MAE_THB'], 'test_within20_no_hood': m_h['within20'],
                    'range_coverage_test_no_hood': cover(pred_h, h10, h90), 'n_train': len(train), 'n_test': len(test)}}
    DOCS.mkdir(exist_ok=True)
    path = DOCS / 'model.json'
    path.write_text(json.dumps(out, separators=(',', ':')))
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in out['meta'].items()}, '| range x', round(out['range']['lo'], 2), round(out['range']['hi'], 2))
    print('model.json', round(path.stat().st_size / 1024, 1), 'KB;', len(hood_code), 'neighbourhoods;', room_code)
