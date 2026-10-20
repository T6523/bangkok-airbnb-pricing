import itertools
import json

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RESULTS_DIR, SEED
from src.airbnb_pricing.evaluate import metrics

N_BOOT = 1000
REPORTED = ['MAE_THB', 'within20', 'R2_log']


def host_resamples(host_id: np.ndarray, n: int, rng: np.random.Generator):
    """Yield n index arrays, each built by drawing hosts (not listings) with replacement."""
    order = np.argsort(host_id, kind='stable')
    groups = np.split(order, np.unique(host_id[order], return_index=True)[1][1:])
    for _ in range(n):
        yield np.concatenate([groups[h] for h in rng.integers(0, len(groups), len(groups))])


def ci(x) -> dict:
    lo, hi = np.percentile(x, [2.5, 97.5])
    return {'lo': float(lo), 'hi': float(hi)}


if __name__ == '__main__':
    df = pd.read_parquet(PROCESSED_DIR / 'test_preds.parquet')
    models = [c for c in df.columns if c not in ('id', 'host_id', 'price')]
    y = df['price'].to_numpy()
    P = {m: df[m].to_numpy() for m in models}
    rng = np.random.default_rng(SEED)

    # one set of resampled hosts per iteration shared by every model 
    boot = {m: {k: [] for k in REPORTED} for m in models}
    for idx in host_resamples(df['host_id'].to_numpy(), N_BOOT, rng):
        for m in models:
            r = metrics(y[idx], P[m][idx])
            for k in REPORTED:
                boot[m][k].append(r[k])

    out = {'n_resamples': N_BOOT, 'resampled': 'test hosts', 'models': {}, 'mae_diff': {}}
    for m in models:
        point = metrics(y, P[m])
        out['models'][m] = {k: {'point': point[k], **ci(boot[m][k])} for k in REPORTED}
    for a, b in itertools.combinations(models, 2):  # a - b: negative means a has the lower MAE
        d = np.array(boot[a]['MAE_THB']) - np.array(boot[b]['MAE_THB'])
        c = ci(d)
        out['mae_diff'][f'{a} - {b}'] = {'point': out['models'][a]['MAE_THB']['point'] - out['models'][b]['MAE_THB']['point'],
                                          **c, 'excludes_zero': bool(c['lo'] > 0 or c['hi'] < 0)}
    (RESULTS_DIR / 'bootstrap.json').write_text(json.dumps(out, indent=2))

    for m in models:
        r = out['models'][m]['MAE_THB']
        print(f"{m:22s} MAE {r['point']:7.1f}  [{r['lo']:7.1f}, {r['hi']:7.1f}]")
    print()
    for k, v in out['mae_diff'].items():
        print(f"{k:42s} {v['point']:7.1f}  [{v['lo']:7.1f}, {v['hi']:7.1f}]  {'REAL' if v['excludes_zero'] else 'unclear'}")
