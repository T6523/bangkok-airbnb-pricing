import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from config import FIGURES_DIR, RESULTS_DIR, SEED
from src.airbnb_pricing.baselines import CATEGORICAL, as_model_input
from src.airbnb_pricing.evaluate import load_data
from src.airbnb_pricing.models import tuned

N_REPEATS = 10
SHAP_ROWS = 5000
GROUPS = {
    'size': ['accommodates', 'bedrooms', 'beds', 'bathrooms', 'bathroom_shared', 'beds_per_person', 'bedrooms_per_person'],
    'location': ['neighbourhood_cleansed', 'latitude', 'longitude', 'dist_to_center_km'],
    'type': ['room_type', 'property_type_grouped'],
    'host': ['host_tenure_months', 'host_is_superhost', 'host_identity_verified', 'host_has_profile_pic',
             'host_listings_count', 'calculated_host_listings_count', 'host_listing_count', 'is_multi_listing_host'],
    'stay_rules': ['minimum_nights', 'maximum_nights'],
    'reviews': ['number_of_reviews', 'number_of_reviews_ltm', 'reviews_per_month', 'has_reviews',
                'days_since_last_review', 'listing_age_days', 'rating_pct_in_hood'],
    'availability': ['availability_30', 'availability_60', 'availability_90', 'availability_365', 'is_inactive_like'],
    'amenities': ['amenity_count'],
}
PREFIX_GROUPS = {'review_scores_': 'reviews', 'am_': 'amenities'}


def feature_groups(columns) -> dict:
    """column -> group"""
    out = {c: g for g, cols in GROUPS.items() for c in cols}
    for c in columns:
        for prefix, g in PREFIX_GROUPS.items():
            if c.startswith(prefix):
                out[c] = g
    missing = [c for c in columns if c not in out]
    assert not missing, f'features without a group: {missing}'
    return {c: out[c] for c in columns}


def group_permutation_importance(model, X, y, groups: dict, rng) -> pd.DataFrame:
    """Increase in test MAE (THB) when a whole group of columns is shuffled together."""
    base = np.mean(np.abs(y - np.exp(model.predict(X))))
    rows = []
    for g in sorted(set(groups.values())):
        cols = [c for c, gg in groups.items() if gg == g]
        deltas = []
        for _ in range(N_REPEATS):
            idx = rng.permutation(len(X))
            Xp = X.copy()
            for c in cols:  # same row permutation for every column in the group keeps it internally consistent
                Xp[c] = X[c].iloc[idx].values
            deltas.append(np.mean(np.abs(y - np.exp(model.predict(Xp)))) - base)
        rows.append({'group': g, 'n_features': len(cols), 'mae_increase_mean': np.mean(deltas), 'mae_increase_sd': np.std(deltas)})
    return pd.DataFrame(rows).sort_values('mae_increase_mean', ascending=False)


if __name__ == '__main__':
    FIGURES_DIR.mkdir(exist_ok=True)
    train, test = load_data()
    X_tr, X_te = as_model_input(train), as_model_input(test)
    model = tuned('lgbm').fit(X_tr, np.log(train['price']))
    groups = feature_groups(list(X_te.columns))
    pd.DataFrame({'name': list(groups), 'group': list(groups.values())}).to_csv(RESULTS_DIR / 'feature_dictionary.csv', index=False)

    # 1. grouped permutation importance (test set, MAE in THB)
    imp = group_permutation_importance(model, X_te, test['price'].to_numpy(), groups, np.random.default_rng(SEED))
    imp.to_csv(RESULTS_DIR / 'importance_groups.csv', index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp['group'][::-1], imp['mae_increase_mean'][::-1], xerr=imp['mae_increase_sd'][::-1])
    ax.set_xlabel('Increase in test MAE (THB) when group is shuffled')
    ax.set_title('Grouped permutation importance (tuned LightGBM)')
    fig.tight_layout(); fig.savefig(FIGURES_DIR / 'importance_groups.png', dpi=150); plt.close(fig)
    print(imp.round(1).to_string(index=False))

    # 2. SHAP on up to 5,000 test rows; values are on the log(price) scale
    Xs = X_te.sample(min(SHAP_ROWS, len(X_te)), random_state=SEED)
    sv = shap.TreeExplainer(model).shap_values(Xs)
    top = pd.DataFrame({'feature': Xs.columns, 'mean_abs_shap': np.abs(sv).mean(axis=0)}).sort_values('mean_abs_shap', ascending=False)
    top.to_csv(RESULTS_DIR / 'shap_top_features.csv', index=False)
    print(top.head(15).round(4).to_string(index=False))

    Xplot = Xs.copy()
    Xplot[CATEGORICAL] = Xplot[CATEGORICAL].apply(lambda s: s.cat.codes)  
    shap.summary_plot(sv, Xplot, max_display=15, show=False)
    plt.gcf().savefig(FIGURES_DIR / 'shap_beeswarm.png', dpi=150, bbox_inches='tight'); plt.close('all')
