import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FEATURES_PATH, FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR, SPLIT_PATH

MODEL = 'ensemble_tuned'
MIN_HOOD_N = 20


def segment_row(kind: str, name, d: pd.DataFrame) -> dict:
    err = d['pred'] - d['price']
    ape = err.abs() / d['price']
    return {'segment_type': kind, 'segment': name, 'n': len(d), 'MAE_THB': err.abs().mean(),
            'MdAPE': ape.median(), 'within20': (ape <= 0.2).mean(), 'bias_THB': err.mean()}


if __name__ == '__main__':
    FIGURES_DIR.mkdir(exist_ok=True)
    feats = pd.read_parquet(FEATURES_PATH)
    split = pd.read_parquet(SPLIT_PATH)
    train_price = feats.merge(split, on='id').query("`set` == 'train'")['price']
    test = pd.read_parquet(PROCESSED_DIR / 'test_preds.parquet').rename(columns={MODEL: 'pred'})
    test = test[['id', 'price', 'pred']].merge(
        feats[['id', 'room_type', 'neighbourhood_cleansed', 'accommodates', 'availability_365',
               'number_of_reviews', 'host_listings_count']], on='id')

    # price terciles: cut points come from train only
    lo, hi = train_price.quantile([1 / 3, 2 / 3])
    test['price_tercile'] = pd.cut(test['price'], [-np.inf, lo, hi, np.inf],
                                   labels=[f'low (<{lo:.0f})', f'mid ({lo:.0f}-{hi:.0f})', f'high (>{hi:.0f})'])
    rows = [segment_row('overall', 'all', test)]
    rows += [segment_row('room_type', k, d) for k, d in test.groupby('room_type')]
    rows += [segment_row('price_tercile', k, d) for k, d in test.groupby('price_tercile', observed=True)]
    rows += [segment_row('neighbourhood', k, d) for k, d in test.groupby('neighbourhood_cleansed') if len(d) >= MIN_HOOD_N]
    seg = pd.DataFrame(rows)
    seg.to_csv(RESULTS_DIR / 'metrics_by_segment.csv', index=False)
    print(seg[seg.segment_type != 'neighbourhood'].round(3).to_string(index=False))
    hood = seg[seg.segment_type == 'neighbourhood'].sort_values('MAE_THB')
    print('\nneighbourhoods (n>=20):', len(hood), '| MAE best', hood.head(3)[['segment', 'MAE_THB']].round(0).values.tolist(),
          'worst', hood.tail(3)[['segment', 'MAE_THB']].round(0).values.tolist())

    # figures
    lim = [test[['price', 'pred']].min().min() * 0.9, test[['price', 'pred']].max().max() * 1.1]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(test['price'], test['pred'], s=4, alpha=0.3)
    ax.plot(lim, lim, 'r--', lw=1)
    ax.set(xscale='log', yscale='log', xlim=lim, ylim=lim, xlabel='Actual price (THB)', ylabel='Predicted price (THB)',
           title=f'Predicted vs actual, test ({MODEL})')
    fig.tight_layout(); fig.savefig(FIGURES_DIR / 'pred_vs_actual.png', dpi=150); plt.close(fig)

    test['log_resid'] = np.log(test['price']) - np.log(test['pred'])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(test['pred'], test['log_resid'], s=4, alpha=0.3)
    ax.axhline(0, color='r', ls='--', lw=1)
    ax.set(xscale='log', xlabel='Predicted price (THB)', ylabel='log(actual) - log(predicted)', title='Residuals vs predicted, test')
    fig.tight_layout(); fig.savefig(FIGURES_DIR / 'residual_vs_pred.png', dpi=150); plt.close(fig)

    # percentile bins of actual price (about 54 test listings each)
    test['pct'] = pd.qcut(test['price'], 100, labels=False) + 1
    pct = test.groupby('pct').apply(lambda d: pd.Series({'MAE_THB': (d['pred'] - d['price']).abs().mean(),
                                                         'bias_THB': (d['pred'] - d['price']).mean()}), include_groups=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(pct.index, pct['MAE_THB'], marker='o', ms=3); ax[0].set(xlabel='Actual price percentile', ylabel='MAE (THB)', title='MAE by price percentile')
    ax[1].plot(pct.index, pct['bias_THB'], marker='o', ms=3, color='C1'); ax[1].axhline(0, color='k', lw=0.8)
    ax[1].set(xlabel='Actual price percentile', ylabel='Mean (pred - actual), THB', title='Bias by price percentile')
    fig.tight_layout(); fig.savefig(FIGURES_DIR / 'error_by_percentile.png', dpi=150); plt.close(fig)
    print('\nMAE by percentile (every 10th):', pct['MAE_THB'].iloc[9::10].round(0).tolist(), '| top percentile MAE', round(pct['MAE_THB'].iloc[-1]))

    # 20 largest errors: printed locally only, never saved (privacy rule)
    top = test.assign(abs_err=(test['pred'] - test['price']).abs()).nlargest(20, 'abs_err')
    print('\n20 largest errors (local only):')
    print(top[['price', 'pred', 'room_type', 'accommodates', 'availability_365', 'number_of_reviews', 'host_listings_count']]
          .round(0).to_string(index=False))
    print('over-predicted:', int((top['pred'] > top['price']).sum()), '| under-predicted:', int((top['pred'] < top['price']).sum()))
