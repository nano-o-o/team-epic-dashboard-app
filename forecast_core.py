"""
=============================================================================
 forecast_core.py -- fiscal-year-aware, sparsity-featured forecasting logic
=============================================================================
Pure pandas/numpy/sklearn logic, no Streamlit dependency, so it can be
tested standalone and imported by the app (ui/pipeline.py).
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def clean_dates(raw, col_date):
    """Parses the date column, coercing anything unparseable to NaT, then
    drops rows with a missing/invalid date (they can't be assigned a fiscal
    year or month, so they'd otherwise silently corrupt downstream grouping
    -- e.g. producing NaN fiscal years). Returns (cleaned_df, n_dropped) so
    the caller can report how many rows were excluded."""
    raw = raw.copy()
    raw[col_date] = pd.to_datetime(raw[col_date], errors="coerce")
    n_before = len(raw)
    raw = raw.dropna(subset=[col_date])
    n_dropped = n_before - len(raw)
    return raw, n_dropped


def drop_duplicate_lines(raw):
    """Drops sales lines identical in every column (double-exported rows).
    Returns (cleaned_df, n_dropped)."""
    n_before = len(raw)
    raw = raw.drop_duplicates()
    return raw, n_before - len(raw)


def drop_unpriced_items(raw, col_product, item_master, master_join_col, price_col):
    """Drops every line of an item the item master lists with a blank list
    price (items missing from the master are kept). Returns (cleaned_df,
    n_items_dropped)."""
    im = item_master.assign(**{master_join_col: item_master[master_join_col].astype(str)})         .drop_duplicates(subset=[master_join_col])
    unpriced = im.loc[pd.to_numeric(im[price_col], errors="coerce").isna(), master_join_col]
    drop = raw[col_product].astype(str).isin(unpriced)
    return raw[~drop].copy(), raw.loc[drop, col_product].nunique()


def to_fiscal(dates, start_month):
    """Converts calendar dates to (FiscalYear, FiscalMonth). FiscalMonth=1 is
    the fiscal start month (e.g. April). FiscalYear is named by the calendar
    year it STARTS in (Apr 2021 - Mar 2022 = FiscalYear 2021)."""
    month = dates.dt.month
    year = dates.dt.year
    fiscal_year = np.where(month >= start_month, year, year - 1)
    fiscal_month = np.where(month >= start_month, month - start_month + 1, month + (12 - start_month) + 1)
    return fiscal_year, fiscal_month


def fiscal_to_calendar_month(fiscal_month, start_month):
    return ((fiscal_month - 1 + start_month - 1) % 12) + 1


def fiscal_month_labels(start_month):
    return [MONTH_NAMES[(fiscal_to_calendar_month(fm, start_month) - 1)] for fm in range(1, 13)]


def compute_revenue(raw, col_price, col_units, revenue_col="Revenue"):
    """Adds a Revenue column computed as Base Price x Quantity. Rows with a
    missing/blank price or quantity produce 0 revenue rather than NaN, so
    they don't silently disappear from downstream sums."""
    raw = raw.copy()
    price = pd.to_numeric(raw[col_price], errors="coerce").fillna(0)
    units = pd.to_numeric(raw[col_units], errors="coerce").fillna(0)
    raw[revenue_col] = price * units
    return raw


def filter_by_status(raw, col_status, valid_statuses):
    """Keeps only rows whose status is in valid_statuses. If col_status or
    valid_statuses is falsy, returns raw unchanged (no filtering applied)."""
    if not col_status or not valid_statuses or col_status not in raw.columns:
        return raw
    return raw[raw[col_status].isin(valid_statuses)].copy()


def aggregate_to_fiscal_monthly(raw, col_product, col_units, col_revenue, col_date, fiscal_start_month, passthrough_cols=None):
    """passthrough_cols: optional dict of {output_column_name: source_column_name}
    for per-product display attributes that don't vary over time (e.g.
    {'Category': 'Category', 'Item Code': 'Item Code'}). Each is taken as the
    first non-null value per product and merged back onto every row for that
    product -- these are NOT used as model features unless explicitly passed
    to build_feature_matrix elsewhere (e.g. 'Category')."""
    fy, fm = to_fiscal(raw[col_date], fiscal_start_month)
    raw = raw.copy()
    raw["FiscalYear"] = fy
    raw["FiscalMonth"] = fm

    agg = raw.groupby([col_product, "FiscalYear", "FiscalMonth"], as_index=False).agg(Units=(col_units, "sum"), Revenue=(col_revenue, "sum"))

    passthrough_maps = {}
    if passthrough_cols:
        for out_name, source_col in passthrough_cols.items():
            if source_col and source_col in raw.columns:
                passthrough_maps[out_name] = raw.groupby(col_product)[source_col].first()

    all_products = agg[col_product].unique()
    all_fy = sorted(agg["FiscalYear"].unique())
    full_index = pd.MultiIndex.from_product([all_products, all_fy, range(1, 13)], names=[col_product, "FiscalYear", "FiscalMonth"])
    df = agg.set_index([col_product, "FiscalYear", "FiscalMonth"]).reindex(full_index, fill_value=0).reset_index()
    for out_name, mapping in passthrough_maps.items():
        df[out_name] = df[col_product].map(mapping)
    df = df.sort_values([col_product, "FiscalYear", "FiscalMonth"]).reset_index(drop=True)
    return df


def merge_item_master(raw, col_product, item_master, master_join_col, category_map):
    """Joins category columns from a separate item-master file (e.g.
    item.xlsx) onto the sales data. category_map: {output_name: source_col_
    in_item_master}, e.g. {'Category1': 'Product Category 1'}. The join key
    is cast to string on both sides first, since a mismatch there (e.g.
    numeric IDs in one file, text in the other) would otherwise silently
    drop every match instead of raising an error."""
    item_master = item_master.copy()
    item_master[master_join_col] = item_master[master_join_col].astype(str)
    item_master = item_master.drop_duplicates(subset=[master_join_col])

    subset_cols = [master_join_col] + list(category_map.values())
    small = item_master[subset_cols].rename(columns={v: k for k, v in category_map.items()})
    small = small.rename(columns={master_join_col: col_product})

    raw = raw.copy()
    raw[col_product] = raw[col_product].astype(str)
    return raw.merge(small, on=col_product, how="left")


def compute_distinct_customers_trailing(raw, col_product, col_customer, col_date, fiscal_start_month, window=12):
    """For every (product, fiscal month) that appears in the data, counts
    the number of DISTINCT customers who bought that product in the
    TRAILING `window` months, excluding the current month (no leakage).
    Uses an incremental sliding window (add this month's customers, drop
    the month that just fell out of the window) rather than recomputing
    the full union every month -- much faster at full catalog scale.
    Returns a dataframe with [col_product, 'FiscalYear', 'FiscalMonth',
    'distinct_customers_trailing_Nm']."""
    from collections import Counter, deque

    out_col = f"distinct_customers_trailing_{window}m"
    raw = raw.copy()
    fy, fm = to_fiscal(raw[col_date], fiscal_start_month)
    raw["t"] = fy * 12 + fm
    item_month_cust = raw[[col_product, "t", col_customer]].dropna(subset=[col_customer]).drop_duplicates()

    results = []
    for item, g in item_month_cust.groupby(col_product):
        month_to_custs = g.groupby("t")[col_customer].apply(set).to_dict()
        min_t, max_t = min(month_to_custs), max(month_to_custs)
        counter = Counter()
        window_queue = deque()
        distinct_count = 0
        # Run `window` months past the last sale, until it drops out of the
        # window; stopping at max_t would read later months as 0 customers.
        for target_t in range(min_t, max_t + window + 1):
            new_t = target_t - 1
            if new_t in month_to_custs:
                custs = month_to_custs[new_t]
                window_queue.append((new_t, custs))
                for c in custs:
                    if counter[c] == 0:
                        distinct_count += 1
                    counter[c] += 1
            while window_queue and window_queue[0][0] <= target_t - window - 1:
                old_t, old_custs = window_queue.popleft()
                for c in old_custs:
                    counter[c] -= 1
                    if counter[c] == 0:
                        distinct_count -= 1
            results.append([item, target_t, distinct_count])

    result_df = pd.DataFrame(results, columns=[col_product, "t", out_col])
    result_df["FiscalYear"] = (result_df["t"] - 1) // 12
    result_df["FiscalMonth"] = ((result_df["t"] - 1) % 12) + 1
    return result_df.drop(columns=["t"])


def compute_customer_mix_features(raw, col_product, col_customer, col_date, col_units, fiscal_start_month,
                                   customer_type_map=None, industry_map=None,
                                   vip_label="VIP", wholesale_label="Wholesale and Distribution",
                                   window=12, horizon_months=18):
    """Per-(item, month) features describing WHO buys the item, over the
    trailing `window` months (current month excluded -- leakage-safe).

    MELFT's demand is extremely concentrated: a handful of VIP / wholesale
    accounts drive most unit volume, and their ordering behaviour (bulk
    restock, lumpy) differs sharply from end-user demand (steadier). Knowing
    an item's customer mix is therefore a real demand signal, not just
    reporting colour.

    customer_type_map: {customer_id -> 'VIP' | 'End User'}
    industry_map:      {customer_id -> FA sub-industry string}

    Returns [col_product, 'FiscalYear', 'FiscalMonth', 'vip_demand_share_12m',
    'wholesale_demand_share_12m', 'buyer_industry_count_12m',
    'top_industry_share_12m']. horizon_months pads the output past the last
    real order month so the features are defined for the forecast window too
    (held flat at the last real value there)."""
    from collections import Counter, deque

    raw = raw.copy()
    fy, fm = to_fiscal(raw[col_date], fiscal_start_month)
    raw["t"] = (fy * 12 + fm).astype(int)
    ct_map = customer_type_map or {}
    ind_map = industry_map or {}
    raw["_units"] = pd.to_numeric(raw[col_units], errors="coerce").fillna(0)

    grp = raw.groupby([col_product, "t"])
    per_item_month = {}
    for (item, t), g in grp:
        custs = g[col_customer]
        units = g["_units"].to_numpy(dtype=float)
        total = float(units.sum())
        vip = float(units[[ct_map.get(c) == vip_label for c in custs]].sum())
        whole = float(units[[ind_map.get(c) == wholesale_label for c in custs]].sum())
        ind_units = Counter()
        for c, u in zip(custs, units):
            ind = ind_map.get(c)
            if ind and ind not in ("- None -", "None", "Other"):
                ind_units[ind] += u
        per_item_month.setdefault(item, {})[t] = (total, vip, whole, ind_units)

    rows = []
    for item, month_map in per_item_month.items():
        if not month_map:
            continue
        min_t, max_t = min(month_map), max(month_map) + horizon_months
        win = deque()
        tot = vip = whole = 0.0
        ind_win = Counter()
        for target_t in range(min_t, max_t + 1):
            new_t = target_t - 1
            if new_t in month_map:
                d_tot, d_vip, d_whole, d_ind = month_map[new_t]
                win.append((new_t, d_tot, d_vip, d_whole, d_ind))
                tot += d_tot; vip += d_vip; whole += d_whole; ind_win += d_ind
            while win and win[0][0] <= target_t - window - 1:
                o_t, o_tot, o_vip, o_whole, o_ind = win.popleft()
                tot -= o_tot; vip -= o_vip; whole -= o_whole
                ind_win.subtract(o_ind)
                ind_win += Counter()  # drop non-positive counts
            ind_total = sum(v for v in ind_win.values() if v > 0)
            rows.append([
                item, (target_t - 1) // 12, ((target_t - 1) % 12) + 1,
                (vip / tot) if tot > 0 else 0.0,
                (whole / tot) if tot > 0 else 0.0,
                sum(1 for v in ind_win.values() if v > 0),
                (max(ind_win.values()) / ind_total) if ind_total > 0 else 0.0,
            ])
    return pd.DataFrame(rows, columns=[
        col_product, "FiscalYear", "FiscalMonth",
        "vip_demand_share_12m", "wholesale_demand_share_12m",
        "buyer_industry_count_12m", "top_industry_share_12m"])


def compute_reorder_cadence_feature(raw, col_product, col_customer, col_date, fiscal_start_month,
                                     due_factor=0.75, horizon_months=18):
    """B2B leading indicator. For every (customer, item) the customer has
    bought before, this tracks their typical re-order interval and how long
    it's been since their last order, then -- for each (item, month) -- counts
    how many of that item's customers are currently "due" (months since their
    last order >= due_factor x their own average interval).

    B2B demand is derived: a customer reorders a PLC when their line needs
    another one, on a roughly repeating cadence. Recent-sales features miss
    this entirely (the signal is "a regular buyer has gone quiet longer than
    usual", not "sales were high last month"). Everything here uses only
    orders strictly before the month in question -- leakage-safe.

    Returns [col_product, 'FiscalYear', 'FiscalMonth', 'customers_due_reorder',
    'reorder_pressure', 'reorder_readiness'] where reorder_pressure = due /
    active repeat buyers and reorder_readiness sums how overdue those buyers
    are. The column names are REORDER_CADENCE_COLS.
    horizon_months extends the output past the last real order month so the
    feature is defined for the validation / forecast window too."""
    raw = raw.copy()
    fy, fm = to_fiscal(raw[col_date], fiscal_start_month)
    raw["t"] = (fy * 12 + fm).astype(int)
    trans = raw[[col_product, col_customer, "t"]].dropna(subset=[col_customer]).drop_duplicates()

    from collections import defaultdict
    due_counts = defaultdict(int)        # (item, T) -> # customers past due
    repeat_active = defaultdict(int)     # (item, T) -> # customers with >=2 prior orders
    readiness_sum = defaultdict(float)   # (item, T) -> sum of (elapsed / mean_gap), capped

    def _accumulate(item, start_T, mean_gap, end_T):
        if mean_gap <= 0:
            return
        for T in range(start_T + 1, end_T + 1):
            elapsed = T - start_T
            repeat_active[(item, T)] += 1
            readiness_sum[(item, T)] += min(elapsed / mean_gap, 3.0)
            if elapsed >= due_factor * mean_gap:
                due_counts[(item, T)] += 1

    for (item, cust), g in trans.groupby([col_product, col_customer], sort=False):
        months = np.sort(g["t"].unique())
        if len(months) < 2:
            continue
        gaps = np.diff(months)
        # between consecutive orders: cadence estimated only from earlier gaps
        for i in range(1, len(months) - 1):
            # capped at horizon_months like the tail below: counting a buyer as active
            # past that cap would reveal that they come back.
            _accumulate(item, int(months[i]), gaps[:i].mean(), int(min(months[i + 1], months[i] + horizon_months)))
        # after the last known order: carry the full-history cadence forward
        last_month = int(months[-1])
        _accumulate(item, last_month, gaps.mean(), last_month + horizon_months)

    keys = set(repeat_active)
    if not keys:
        return pd.DataFrame(columns=[col_product, "FiscalYear", "FiscalMonth",
                                     "customers_due_reorder", "reorder_pressure",
                                     "reorder_readiness"])
    rows = []
    for (item, T) in keys:
        active = repeat_active[(item, T)]
        due = due_counts.get((item, T), 0)
        rows.append([item, (T - 1) // 12, ((T - 1) % 12) + 1, due,
                     (due / active) if active else 0.0,
                     readiness_sum.get((item, T), 0.0)])
    return pd.DataFrame(rows, columns=[col_product, "FiscalYear", "FiscalMonth",
                                       "customers_due_reorder", "reorder_pressure",
                                       "reorder_readiness"])


def add_sparsity_feature(df, col_product, train_years):
    """Coverage % = fraction of TRAINING-window fiscal-months where the item
    had any sales at all. Computed only from training years to avoid leaking
    future information into the feature. If sparsity_coverage already exists,
    it's replaced (so this is always safe to call again with a new window)."""
    df = df.drop(columns=["sparsity_coverage"], errors="ignore")
    train_only = df[df["FiscalYear"].isin(train_years)]
    coverage = train_only.groupby(col_product).apply(lambda g: (g["Units"] > 0).mean(), include_groups=False)
    coverage.name = "sparsity_coverage"
    return df.merge(coverage, on=col_product, how="left")


def order_rhythm(units):
    """Descriptive buying rhythm of one item from its zero-filled monthly units
    (oldest first, ending at the last month of history). Not a forecast: it
    reads the past only. Returns None if the item never sold. Indices count
    months from the start of `units`; next_idx may run past the end.

    gap is the median number of months between selling months (needs >= 3,
    i.e. two gaps: one gap is a coincidence, not a rhythm). last_vs_typical compares the latest selling month with the item's own
    quartiles: "big" above p75, "small" below p25 (needs >= 3 selling months)."""
    units = np.asarray(units, dtype=float)
    sold = np.flatnonzero(units > 0)
    if len(sold) == 0:
        return None
    sizes = units[sold]
    lo, mid, hi = np.percentile(sizes, [25, 50, 75])
    last_idx = int(sold[-1])
    gap = float(np.median(np.diff(sold))) if len(sold) >= 3 else None
    size_tag = None
    if len(sold) >= 3:
        size_tag = "big" if sizes[-1] > hi else "small" if sizes[-1] < lo else "typical"
    return {
        "sale_months": int(len(sold)), "months": int(len(units)),
        "last_idx": last_idx, "last_size": float(sizes[-1]),
        "gap": gap, "next_idx": None if gap is None else last_idx + int(round(gap)),
        "typical_size": float(mid), "size_lo": float(lo), "size_hi": float(hi),
        "last_vs_typical": size_tag,
    }


def compute_months_since_last_sale(df, col_product, sentinel=99):
    """For each (product, month) row, counts how many months have passed
    since that product last had Units > 0 -- using only PRIOR months, never
    the current month's own value (leakage-safe: this reflects what you'd
    genuinely know standing at the start of that month). sentinel is used
    for a product's rows before its first-ever recorded sale."""
    df = df.sort_values([col_product, "FiscalYear", "FiscalMonth"]).copy()

    def _calc(g):
        units = g["Units"].values
        n = len(units)
        result = np.full(n, sentinel)
        last_sale_pos = None
        for i in range(n):
            if last_sale_pos is not None:
                result[i] = i - last_sale_pos
            if units[i] > 0:
                last_sale_pos = i
        return pd.Series(result, index=g.index)

    return df.groupby(col_product, group_keys=False).apply(_calc, include_groups=False)


def add_features(df, col_product, fiscal_start_month):
    df = df.sort_values([col_product, "FiscalYear", "FiscalMonth"]).copy()
    grp = df.groupby(col_product)["Units"]
    df["lag_1"] = grp.shift(1)
    df["lag_2"] = grp.shift(2)
    df["lag_3"] = grp.shift(3)
    df["lag_12"] = grp.shift(12)
    # Windows run per product: shift(1) returns a flat series, so it must be
    # regrouped before rolling/expanding or one item's window spills into the next.
    prev = grp.shift(1).groupby(df[col_product])
    df["roll_mean_3"] = prev.rolling(3).mean().droplevel(0)
    df["roll_mean_6"] = prev.rolling(6).mean().droplevel(0)
    # avg_monthly_units: expanding mean over ALL of this item's prior months (leakage-safe --
    # shift(1) first, so the current month is never included in its own average).
    df["avg_monthly_units"] = prev.expanding().mean().droplevel(0)
    df["months_since_last_sale"] = compute_months_since_last_sale(df, col_product)
    calendar_month = fiscal_to_calendar_month(df["FiscalMonth"], fiscal_start_month)
    df["month_sin"] = np.sin(2 * np.pi * calendar_month / 12)
    df["month_cos"] = np.cos(2 * np.pi * calendar_month / 12)
    return df


def build_feature_matrix(df, feature_base_cols, category_cols=None):
    """category_cols: a single column name, a list of column names, or None.
    Each present category column gets one-hot encoded and added to the
    feature list."""
    if isinstance(category_cols, str):
        category_cols = [category_cols]
    feature_cols = list(feature_base_cols)
    if category_cols:
        for cat_col in category_cols:
            if cat_col and cat_col in df.columns:
                dummies = pd.get_dummies(df[cat_col], prefix=cat_col.lower())
                df = pd.concat([df, dummies], axis=1)
                feature_cols += list(dummies.columns)
    return df, feature_cols


def model_features(df, col_product, col_category, sparsity_years, fiscal_start_month,
                   feature_base=None, extra_features=None):
    """The one path from history to model inputs: sparsity (from `sparsity_years` only),
    lag/rolling features, then FEATURE_BASE (or `feature_base`) plus whichever
    `extra_features` exist. Returns (feat_df, feature_cols)."""
    df = add_sparsity_feature(df, col_product, sparsity_years)
    feat_df = add_features(df, col_product, fiscal_start_month)
    base = list(feature_base) if feature_base is not None else list(FEATURE_BASE)
    base = base + [c for c in (extra_features or []) if c in feat_df.columns and c not in base]
    return build_feature_matrix(feat_df, base, col_category)


def vectorized_seasonal_naive(history_df, target_rows, col_product, growth_clip=(0.2, 5.0)):
    """Baseline: same fiscal-month last fiscal-year, scaled by that item's
    YoY growth. Fully vectorized -- no per-row python loop. Growth is
    clipped to a sane range so a single sparse item with e.g. 1 unit ->
    50 units doesn't produce a runaway 50x multiplier."""
    lookup = history_df.set_index([col_product, "FiscalYear", "FiscalMonth"])["Units"]
    prev_keys = list(zip(target_rows[col_product], target_rows["FiscalYear"] - 1, target_rows["FiscalMonth"]))
    prev_month_units = lookup.reindex(prev_keys).fillna(0).values

    totals = history_df.groupby([col_product, "FiscalYear"])["Units"].sum()
    prev_totals = totals.reindex(list(zip(target_rows[col_product], target_rows["FiscalYear"] - 1))).fillna(0).values
    prev2_totals = totals.reindex(list(zip(target_rows[col_product], target_rows["FiscalYear"] - 2))).fillna(0).values

    with np.errstate(divide="ignore", invalid="ignore"):
        growth = np.divide(prev_totals, prev2_totals, out=np.ones_like(prev_totals, dtype=float), where=prev2_totals > 0)
    growth = np.clip(growth, growth_clip[0], growth_clip[1])
    return np.maximum(0, prev_month_units * growth)


def croston_forecast(history_df, target_rows, col_product, alpha=0.1, method="sba"):
    """Croston's method (and the Syntetos-Boylan Approximation, 'sba') --
    the classic intermittent-demand baseline. Instead of modelling units
    per month directly, it separately exponentially-smooths (a) the size of
    non-zero demands and (b) the gap between them, then forecasts a flat
    rate = smoothed_size / smoothed_interval for every future month.

    method='croston' is the original; 'sba' multiplies by (1 - alpha/2),
    which removes Croston's known positive bias and is almost always the
    better choice on real data. Returns one prediction per row of
    target_rows (the same flat rate repeated across that product's months)."""
    hist = history_df.sort_values([col_product, "FiscalYear", "FiscalMonth"])
    hist["t"] = hist["FiscalYear"] * 12 + hist["FiscalMonth"]
    rate = {}
    for prod, g in hist.groupby(col_product):
        units = g["Units"].to_numpy(dtype=float)
        nz = np.flatnonzero(units > 0)
        if len(nz) == 0:
            rate[prod] = 0.0
            continue
        if len(nz) == 1:
            rate[prod] = units[nz[0]] / len(units)
            continue
        sizes = units[nz]
        intervals = np.diff(nz, prepend=nz[0] - (nz[1] - nz[0]))
        z, p = sizes[0], intervals[0]
        for i in range(1, len(sizes)):
            z += alpha * (sizes[i] - z)
            p += alpha * (intervals[i] - p)
        r = z / p if p > 0 else 0.0
        if method == "sba":
            r *= (1 - alpha / 2)
        rate[prod] = max(0.0, r)
    return target_rows[col_product].map(rate).fillna(0.0).to_numpy(dtype=float)


class HurdleRegressor(BaseEstimator, RegressorMixin):
    """Two-stage ('hurdle') model for intermittent demand: a classifier
    predicts P(this SKU sells at all this month); a separate regressor,
    trained ONLY on months that had a sale, predicts the quantity given a
    sale. The forecast is the product of the two -- an expected-value that
    can sit anywhere between 0 and the typical order size, which fits
    'mostly zero, occasionally a spike' far better than one regressor forced
    to average over both regimes. Exposes the sklearn fit/predict interface
    so it drops straight into the existing backtest / forecast pipeline."""

    def __init__(self, base="auto"):
        self.base = base
        self.clf = None
        self.reg = None
        self._prior = 0.0

    def _make_pair(self):
        if (self.base == "lightgbm" or self.base == "auto") and LIGHTGBM_AVAILABLE:
            clf = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42,
                                min_child_samples=10, verbosity=-1, n_jobs=-1, objective="binary")
            reg = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42,
                                min_child_samples=5, verbosity=-1, n_jobs=-1,
                                objective="tweedie", tweedie_variance_power=1.2)
            return clf, reg, True
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                             random_state=42, min_samples_leaf=10)
        reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                            random_state=42, min_samples_leaf=5)
        return clf, reg, False

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        sold = y > 0
        self._prior = float(sold.mean())
        self.clf, self.reg, self._lgbm_clf = self._make_pair()
        if len(np.unique(sold)) < 2:
            # Degenerate: every training row is the same class. Fall back to
            # a constant probability times a plain mean quantity.
            self.clf = None
            self.reg = None
            self._const_qty = float(y[sold].mean()) if sold.any() else 0.0
            return self
        self.clf.fit(X, sold.astype(int))
        self.reg.fit(X[sold], y[sold])
        return self

    def predict(self, X):
        if self.clf is None:
            return np.full(len(X), self._prior * getattr(self, "_const_qty", 0.0))
        if self._lgbm_clf:
            p_sell = np.clip(self.clf.predict(X), 0, 1)
        else:
            p_sell = self.clf.predict_proba(X)[:, 1]
        qty = np.maximum(0, self.reg.predict(X))
        return p_sell * qty


def available_model_types():
    """Model types the app offers, in display order. Each entry is
    (label, is_available, note_if_unavailable).

    Trimmed from the original six: the two Tweedie-boosted models won the
    comparison and are near-identical, HistGradientBoosting is the
    dependency-free fallback -- those three are run by default. LightGBM
    (standard) and Hurdle (2-stage) stay selectable as evidence (they show,
    respectively, that the Tweedie objective matters and that an explicit
    intermittent-demand architecture doesn't beat it). Random Forest was
    dropped: slowest to fit and worst of the tree family on this data."""
    lgb_note = None if LIGHTGBM_AVAILABLE else "pip install lightgbm"
    xgb_note = None if XGBOOST_AVAILABLE else "pip install xgboost"
    return [
        ("LightGBM (Tweedie)", LIGHTGBM_AVAILABLE, lgb_note),
        ("XGBoost (Tweedie)", XGBOOST_AVAILABLE, xgb_note),
        ("HistGradientBoosting", True, None),
        ("LightGBM (standard)", LIGHTGBM_AVAILABLE, lgb_note),
        ("Hurdle (2-stage)", True, None),
    ]


def make_model(model_type):
    """Tweedie objective/variance power: suited for zero-heavy, intermittent
    demand data (mostly zeros with occasional positive spikes) -- a much
    better distributional fit than a standard regression objective for
    sparse B2B-style SKUs. variance_power ~1.1-1.5 sits between Poisson
    (p=1, count-like) and Gamma (p=2, continuous positive)."""
    if model_type == "LightGBM (Tweedie)":
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM isn't installed -- run: pip install lightgbm")
        return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42,
                              min_child_samples=5, verbosity=-1, n_jobs=-1,
                              objective="tweedie", tweedie_variance_power=1.3)
    if model_type == "LightGBM (standard)":
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM isn't installed -- run: pip install lightgbm")
        return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42,
                              min_child_samples=5, verbosity=-1, n_jobs=-1)
    if model_type == "XGBoost (Tweedie)":
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost isn't installed -- run: pip install xgboost")
        return XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42,
                             objective="reg:tweedie", tweedie_variance_power=1.3, n_jobs=-1)
    if model_type == "Hurdle (2-stage)":
        return HurdleRegressor(base="auto")
    if model_type == "HistGradientBoosting":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=42, min_samples_leaf=5)
    raise ValueError(f"Unknown model type: {model_type}")


# Prediction-interval quantiles. p10/p50/p90 = an 80% interval, the standard
# choice: wide enough to be useful for stocking, narrow enough that coverage is
# testable on only one held-out year.
QUANTILES = (0.1, 0.5, 0.9)


class ScaledQuantileRegressor(BaseEstimator, RegressorMixin):
    """Fits a quantile model on demand relative to the item's own recent level,
    Units / (roll_mean_6 + 1), and multiplies back at predict time. The scale is
    a prior-months feature, so q(Units) = scale * q(Units / scale) exactly and
    nothing leaks. Used for p10 only: fitted on raw units the p10 booster never
    climbed past ~16-19 units for any item (15959 showed 18 against a monthly
    minimum of 163); on the ratio it reaches ~61 and FY2024 p10 pinball loss
    drops 0.2555 -> 0.2485. The same change made p50 and p90 worse, so they stay
    on raw units. Without roll_mean_6 (feature switched off) it fits raw units."""

    def __init__(self, base, scale_col="roll_mean_6"):
        self.base = base
        self.scale_col = scale_col

    def _scale(self, X):
        if self.scale_col not in getattr(X, "columns", ()):
            return np.ones(len(X))
        return np.maximum(X[self.scale_col].to_numpy(dtype=float), 0) + 1

    def fit(self, X, y):
        self.base.fit(X, np.asarray(y, dtype=float) / self._scale(X))
        return self

    def predict(self, X):
        return self.base.predict(X) * self._scale(X)


def make_quantile_model(alpha):
    """A regressor that predicts the `alpha`-quantile of demand rather than its
    mean -- e.g. alpha=0.9 answers 'units we'd exceed only 10% of the time'.
    LightGBM's pinball-loss ('quantile') objective where available so it stays in
    the same model family as the point forecast; HistGradientBoosting's identical
    'quantile' loss otherwise (ships with scikit-learn, always available).
    Lower quantiles (p10) are fitted scale-free, see ScaledQuantileRegressor."""
    if alpha < 0.5:
        return ScaledQuantileRegressor(_base_quantile_model(alpha))
    return _base_quantile_model(alpha)


def _base_quantile_model(alpha):
    if LIGHTGBM_AVAILABLE:
        return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42,
                             min_child_samples=5, verbosity=-1, n_jobs=-1,
                             objective="quantile", alpha=float(alpha))
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         random_state=42, min_samples_leaf=5,
                                         loss="quantile", quantile=float(alpha))


def _pinball_loss(actual, pred, alpha):
    """Average pinball (quantile) loss -- the proper scoring rule for a quantile
    forecast. Penalises being above the actual by (1-alpha), below it by alpha,
    so it's minimised only by the true alpha-quantile."""
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    d = actual - pred
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d))) if len(d) else np.nan


def quantile_backtest(df, col_product, col_category, train_years, validation_year, fiscal_start_month,
                      quantiles=QUANTILES, extra_features=None, feature_base=None):
    """Fits one quantile model per level on train_years, predicts the held-out
    validation year, and reports whether the interval is calibrated:
      'coverage' -- share of validation SKU-months that landed inside [q_lo, q_hi]
      'target'   -- the nominal coverage it SHOULD have (q_hi - q_lo, e.g. 0.80)
      'pinball'  -- {alpha: mean pinball loss} for each level
      'n'        -- rows scored
    Same feature pipeline and leakage rules as run_backtest()."""
    feat_df, feature_cols = model_features(df, col_product, col_category, train_years, fiscal_start_month,
                                           feature_base, extra_features)

    train = feat_df[feat_df["FiscalYear"].isin(train_years)].dropna(subset=feature_cols)
    test = feat_df[feat_df["FiscalYear"] == validation_year].dropna(subset=feature_cols)
    if len(train) < 10 or len(test) == 0:
        return None

    qs = sorted(quantiles)
    preds = {}
    for q in qs:
        m = make_quantile_model(q)
        m.fit(train[feature_cols], train["Units"])
        preds[q] = np.maximum(0, m.predict(test[feature_cols]))
    # Non-crossing: a lower quantile must never sit above a higher one.
    stacked = np.sort(np.column_stack([preds[q] for q in qs]), axis=1)
    lo, hi = stacked[:, 0], stacked[:, -1]
    y = test["Units"].to_numpy(dtype=float)
    return {
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "target": float(max(qs) - min(qs)),
        "pinball": {q: _pinball_loss(y, preds[q], q) for q in qs},
        "n": int(len(y)),
        # One-sided hit rates. The upper one is what matters for stocking ("how
        # often did demand stay at or below the p90 we planned for?") and, unlike
        # two-sided coverage, it can't be inflated by a lower bound that sits at
        # zero. On demand this sparse the p10 model correctly learns "zero" for
        # most item-months, so `lower_at_zero` reports how much of the interval's
        # apparent coverage is that trivial floor.
        "upper_hit": float(np.mean(y <= hi)),
        "upper_target": float(max(qs)),
        "lower_at_zero": float(np.mean(lo <= 0)),
    }


def compute_bias_correction(res_df, model_name, df, col_product, clip=(0.7, 1.4)):
    """A mean-objective model trained on mostly-zero data tends to under-forecast
    the occasional spike. This measures, on the held-out backtest year, the
    multiplier that would have made sum(forecast) == sum(actual) -- overall and
    per sparsity tier (sparse items are where the under-forecast is worst, so a
    per-tier factor is more honest than one blanket number). Returns
    {'global': float, 'by_tier': {tier: float}, 'raw_bias': float} where
    raw_bias = sum(pred)/sum(actual) - 1 (negative => under-forecasting)."""
    pred_col = f"{model_name}_Pred"
    if res_df is None or pred_col not in res_df.columns:
        return None
    r = res_df[[col_product, "Actual", pred_col]].copy()
    g_pred, g_act = float(r[pred_col].sum()), float(r["Actual"].sum())
    glob = float(np.clip(g_act / g_pred, *clip)) if g_pred > 0 else 1.0
    by_tier = {}
    if "sparsity_coverage" in df.columns:
        cov = df.drop_duplicates(col_product).set_index(col_product)["sparsity_coverage"]
        r["tier"] = pd.cut(r[col_product].map(cov), bins=[-0.01, 0.25, 0.75, 1.01],
                           labels=["Sparse", "Medium", "Dense"])
        for tier, gg in r.groupby("tier", observed=True):
            sp, sa = float(gg[pred_col].sum()), float(gg["Actual"].sum())
            by_tier[str(tier)] = float(np.clip(sa / sp, *clip)) if sp > 0 else 1.0
    return {"global": glob, "by_tier": by_tier,
            "raw_bias": (g_pred / g_act - 1) if g_act else np.nan}


def _bias_factor_series(fc_df, correction, df, col_product):
    """Per-row multiplier: the sparsity-tier factor for each product, falling
    back to the global factor. Shared by apply_bias_correction() and the
    dashboard's before/after backtest re-score."""
    factor = np.full(len(fc_df), correction["global"], dtype=float)
    if correction.get("by_tier") and "sparsity_coverage" in df.columns:
        cov = df.drop_duplicates(col_product).set_index(col_product)["sparsity_coverage"]
        tier = pd.cut(fc_df[col_product].map(cov), bins=[-0.01, 0.25, 0.75, 1.01],
                      labels=["Sparse", "Medium", "Dense"]).astype(object)
        mapped = tier.map(correction["by_tier"]).astype(float)
        factor = mapped.fillna(correction["global"]).to_numpy()
    return factor


def apply_bias_correction(fc_df, correction, df, col_product, value_cols):
    """Multiplies each named forecast column by the per-tier bias factor. No
    rounding here -- the caller rounds units after."""
    if not correction:
        return fc_df
    fc = fc_df.copy()
    factor = _bias_factor_series(fc, correction, df, col_product)
    for c in value_cols:
        if c in fc.columns:
            fc[c] = fc[c].to_numpy(dtype=float) * factor
    return fc


FEATURE_BASE = [
    "lag_1", "lag_2", "lag_3", "lag_12", "roll_mean_3", "roll_mean_6",
    "avg_monthly_units", "months_since_last_sale",
    "month_sin", "month_cos", "sparsity_coverage",
]


def compute_feature_importance(model, feature_cols, test_X=None, test_y=None, n_repeats=5, random_state=42):
    """Returns a DataFrame of [Feature, Importance] sorted highest first,
    plus a string naming which method was used. Tree-based models
    (LightGBM, XGBoost) expose built-in impurity-based importance directly.
    HistGradientBoostingRegressor does NOT expose this
    attribute at all -- for that (and as a general fallback), this uses
    permutation importance instead: it measures how much each feature
    actually degrades held-out prediction accuracy when shuffled, which
    also tends to be more reliable than impurity-based importance (which
    can be biased toward high-cardinality features like one-hot category
    dummies)."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        method = "Built-in (impurity-based)"
    elif test_X is not None and test_y is not None and len(test_X) > 0:
        from sklearn.inspection import permutation_importance
        result = permutation_importance(model, test_X, test_y, n_repeats=n_repeats, random_state=random_state, n_jobs=1)
        importances = result.importances_mean
        method = "Permutation importance (evaluated on held-out validation data)"
    else:
        return None, None

    imp_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances}).sort_values("Importance", ascending=False).reset_index(drop=True)
    return imp_df, method


def group_importance_by_category(importance_df, category_cols):
    """One-hot encoding splits a single categorical feature (e.g. Category1)
    into many dummy columns (category1_PLC, category1_Robot, ...), which
    scatters its true combined importance across several small bars and
    makes it easy to underestimate at a glance. This collapses each
    category column's dummies back into ONE summed row (e.g. 'Category1'),
    leaving all other (non-dummy) features untouched. category_cols: the
    original category column names (e.g. ['Category1', 'Category2']) --
    dummy columns are matched by the same '{col.lower()}_' prefix used when
    they were created in build_feature_matrix()."""
    if importance_df is None or not category_cols:
        return importance_df
    remaining = importance_df.copy()
    grouped_rows = []
    for cat_col in category_cols:
        prefix = f"{cat_col.lower()}_"
        mask = remaining["Feature"].str.startswith(prefix)
        if mask.any():
            grouped_rows.append({"Feature": cat_col, "Importance": remaining.loc[mask, "Importance"].sum()})
            remaining = remaining[~mask]
    if not grouped_rows:
        return importance_df
    result = pd.concat([remaining, pd.DataFrame(grouped_rows)], ignore_index=True)
    return result.sort_values("Importance", ascending=False).reset_index(drop=True)


# =============================================================================
#  Evaluation layer -- richer, decision-relevant accuracy metrics
# =============================================================================
# The single blended SKU-month WAPE is the hardest possible test and rarely
# the number the business actually cares about. These helpers add:
#   * MASE   -- error scaled against the naive forecast's own difficulty, so
#              "< 1" means "genuinely beats naive" regardless of how spiky the
#              series is (the fair yardstick for intermittent demand).
#   * Bias   -- systematic over/under-forecast (sum(pred)/sum(actual) - 1),
#              which for inventory matters more than symmetric error size.
#   * Aggregated WAPE -- the same forecast scored after summing actual & pred
#              to the grain decisions are made at (category / quarter / total).
#              Aggregation cancels SKU-month noise; this is usually the honest
#              operational number.
#   * Active-SKU WAPE -- restricted to items that actually sold in the 12
#              months before the validation year, so discontinued/dead SKUs
#              (structural zeros) don't define the headline.

def wape(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = np.abs(actual).sum()
    return np.abs(actual - pred).sum() / denom if denom else np.nan


def forecast_bias(actual, pred):
    """sum(pred)/sum(actual) - 1. Positive => over-forecasting on aggregate."""
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = actual.sum()
    return (pred.sum() / denom - 1) if denom else np.nan


def naive_scale(df, col_product, train_years, seasonal=False):
    """Pooled mean absolute period-over-period change of each product's
    TRAINING series -- the denominator for MASE. seasonal=True uses the
    12-month difference (matches a seasonal-naive baseline); False uses the
    1-month difference (the standard choice for intermittent demand, since
    it stays well-defined when the series is mostly zeros)."""
    step = 12 if seasonal else 1
    t = df[df["FiscalYear"].isin(train_years)].sort_values([col_product, "FiscalYear", "FiscalMonth"])
    # Exclude products with no sales at all in training: their series is a
    # flat line of zeros, contributing only 0-valued differences that drag
    # the pooled scale down and artificially inflate MASE for everyone else.
    ever_sold = t.groupby(col_product)["Units"].transform("sum") > 0
    t = t[ever_sold]
    diffs = t.groupby(col_product)["Units"].diff(step).abs()
    scale = diffs.mean()
    return scale if scale and not np.isnan(scale) else np.nan


def point_metrics(actual, pred, mase_scale=None):
    """Row-level accuracy metrics for one prediction column."""
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ae = np.abs(actual - pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = ae / np.where(actual == 0, np.nan, actual)
    mae = ae.mean() if len(ae) else np.nan
    valid_scale = mase_scale is not None and not np.isnan(mase_scale) and mase_scale > 0
    return {
        "WAPE": wape(actual, pred),
        "MAE": mae,
        "Bias": forecast_bias(actual, pred),
        "MASE": (mae / mase_scale) if valid_scale else np.nan,
        "MAPE": np.nanmean(ape) if len(ape) else np.nan,
        "n_rows": int(len(actual)),
    }


def active_products(df, col_product, train_years):
    """Products with at least one unit sold in the most recent training year
    -- i.e. demonstrably still active at the moment the forecast is made
    (leakage-free: uses only training data). Everything else is a dormant or
    discontinued SKU whose all-zero validation rows only depress the headline
    without informing any real decision."""
    last_train = max(train_years)
    sold = df[(df["FiscalYear"] == last_train) & (df["Units"] > 0)][col_product].unique()
    return set(sold)


def aggregated_wape(res_df, pred_col, group_cols, actual_col="Actual"):
    """WAPE after summing actual and prediction to `group_cols` grain.
    group_cols=[] collapses everything to a single total (pure volume WAPE)."""
    if group_cols:
        g = res_df.groupby(group_cols, observed=True).agg(
            a=(actual_col, "sum"), p=(pred_col, "sum")).reset_index()
    else:
        g = pd.DataFrame({"a": [res_df[actual_col].sum()], "p": [res_df[pred_col].sum()]})
    return wape(g["a"], g["p"])


def evaluate_backtest(res_df, df, col_product, train_years, category_cols=None, fiscal_start_month=4):
    """Builds the full accuracy report for every `{model}_Pred` column in
    res_df. Returns a dict of DataFrames:
      'summary'          -- Metric x Model (WAPE / MAE / Bias / MASE / MAPE),
                            SKU-month grain, all rows
      'active_only'      -- same metrics, restricted to still-active SKUs
      'by_grain'         -- WAPE at SKU-month / category / quarter / total grain
      'by_sparsity'      -- WAPE per sparsity tier (needs sparsity_coverage on df)
    """
    pred_cols = [c[:-5] for c in res_df.columns if c.endswith("_Pred")]
    if isinstance(category_cols, str):
        category_cols = [category_cols]
    category_cols = [c for c in (category_cols or []) if c in df.columns]

    # attach category + a fiscal quarter onto the row-level results
    r = res_df.copy()
    attr_cols = category_cols + (["sparsity_coverage"] if "sparsity_coverage" in df.columns else [])
    if attr_cols:
        attrs = df.drop_duplicates(col_product).set_index(col_product)[attr_cols]
        r = r.join(attrs, on=col_product)
    r["FiscalQuarter"] = ((r["FiscalMonth"] - 1) // 3 + 1).astype(int)

    scale_ns = naive_scale(df, col_product, train_years, seasonal=False)
    active = active_products(df, col_product, train_years)
    r_active = r[r[col_product].isin(active)]

    def _summary(frame):
        rows = {}
        for m in pred_cols:
            rows[m] = point_metrics(frame["Actual"], frame[f"{m}_Pred"], mase_scale=scale_ns)
        return pd.DataFrame(rows).T[["WAPE", "MAE", "Bias", "MASE", "MAPE", "n_rows"]]

    summary = _summary(r)
    active_only = _summary(r_active)

    grain_defs = [("SKU-month", [col_product, "FiscalYear", "FiscalMonth"])]
    for c in category_cols:
        grain_defs.append((f"{c} x month", [c, "FiscalYear", "FiscalMonth"]))
    grain_defs.append(("SKU x quarter", [col_product, "FiscalYear", "FiscalQuarter"]))
    grain_defs.append(("Total (volume)", []))
    by_grain = pd.DataFrame(
        {m: {name: aggregated_wape(r, f"{m}_Pred", cols) for name, cols in grain_defs}
         for m in pred_cols}
    )

    by_sparsity = None
    if "sparsity_coverage" in r.columns:
        r2 = r.copy()
        r2["Sparsity Tier"] = pd.cut(r2["sparsity_coverage"], bins=[-0.01, 0.25, 0.75, 1.01],
                                     labels=["Sparse (<25%)", "Medium (25-75%)", "Dense (>75%)"])
        recs = []
        for tier, g in r2.groupby("Sparsity Tier", observed=True):
            rec = {"Sparsity Tier": tier, "N items": g[col_product].nunique()}
            for m in pred_cols:
                rec[m] = wape(g["Actual"], g[f"{m}_Pred"])
            recs.append(rec)
        by_sparsity = pd.DataFrame(recs)

    return {"summary": summary, "active_only": active_only,
            "by_grain": by_grain, "by_sparsity": by_sparsity}


def run_backtest(df, col_product, col_category, train_years, validation_year, fiscal_start_month, model_types, extra_features=None, feature_base=None):
    """Trains every requested model type on the same train/test split and
    returns one comparison table (Metric x [Baseline, model1, model2, ...]).
    extra_features: optional list of feature columns already merged onto df
    by the caller (e.g. 'distinct_customers_trailing_12m'), included in the
    feature set if present. feature_base: the core time-series feature list
    to use (defaults to FEATURE_BASE) -- pass a subset to let the caller
    turn individual features off. Returns (res_df, metrics, feature_cols,
    trained_models, skipped, test_data) where trained_models is {model_type:
    fitted_model}, skipped is a list of (model_type, reason) for any
    requested model that couldn't be trained (e.g. its library isn't
    installed), and test_data is (test[feature_cols], test['Units']) for use
    with compute_feature_importance()."""
    feat_df, feature_cols = model_features(df, col_product, col_category, train_years, fiscal_start_month,
                                           feature_base, extra_features)

    train = feat_df[feat_df["FiscalYear"].isin(train_years)].dropna(subset=feature_cols)
    test = feat_df[feat_df["FiscalYear"] == validation_year].dropna(subset=feature_cols)

    if len(train) < 10 or len(test) == 0:
        return None, None, None, {}, [], (None, None)

    train_history = df[df["FiscalYear"].isin(train_years)]
    res_df = test[[col_product, "FiscalYear", "FiscalMonth", "Units"]].copy().rename(columns={"Units": "Actual"})
    actual_sum = res_df["Actual"].sum()

    metric_names = ["MAPE (row-average %)", "WAPE (volume-weighted %)", "MAE (avg units off by)",
                    "RMSE (big misses weigh more)"]
    metrics_data = {"Metric": metric_names}

    def _record(name, preds):
        """Adds one prediction column + its AE column to res_df and its
        [MAPE, WAPE, MAE, RMSE] row to metrics_data."""
        preds = np.asarray(preds, dtype=float)
        ae = (res_df["Actual"] - preds).abs()
        res_df[f"{name}_Pred"] = preds
        res_df[f"{name}_AE"] = ae
        metrics_data[name] = [
            (ae / res_df["Actual"].replace(0, np.nan)).mean(),
            ae.sum() / actual_sum,
            ae.mean(),
            np.sqrt((ae ** 2).mean()),
        ]

    _record("Baseline", vectorized_seasonal_naive(train_history, test, col_product))
    _record("Croston (SBA)", croston_forecast(train_history, test, col_product, method="sba"))

    trained_models = {}
    skipped = []
    for mt in model_types:
        try:
            model = make_model(mt)
        except ImportError as e:
            skipped.append((mt, str(e)))
            continue
        model.fit(train[feature_cols], train["Units"])
        _record(mt, np.maximum(0, model.predict(test[feature_cols])))
        trained_models[mt] = model

    metrics = pd.DataFrame(metrics_data)
    return res_df, metrics, feature_cols, trained_models, skipped, (test[feature_cols], test["Units"])


# The two experimental B2B feature blocks. Each is produced by ONE expensive
# function (compute_reorder_cadence_feature / compute_customer_mix_features),
# so callers gate the computation on "any column in the block is wanted" and
# then keep only the wanted columns. Named here -- not re-listed by callers --
# so the dashboard build and feature_ablation() can never disagree about which
# columns a block contains.
REORDER_CADENCE_COLS = ["customers_due_reorder", "reorder_pressure", "reorder_readiness"]
CUSTOMER_MIX_COLS = ["vip_demand_share_12m", "wholesale_demand_share_12m",
                     "buyer_industry_count_12m", "top_industry_share_12m"]

FEATURE_GROUPS = [
    # (label, category columns to add, extra numeric feature columns to add)
    ("Base (lags, rolling, calendar, sparsity)", [], []),
    ("+ Product category", ["Category1", "Category2", "Category"], []),
    ("+ List price", [], ["ListPrice"]),
    ("+ Distinct customers (12m)", [], ["distinct_customers_trailing_12m"]),
    ("+ Reorder cadence", [], REORDER_CADENCE_COLS),
    ("+ Customer mix", [], CUSTOMER_MIX_COLS),
]


def feature_ablation(df, col_product, train_years, validation_year, fiscal_start_month,
                     model_type="LightGBM (Tweedie)"):
    """Adds feature groups one at a time and reports what each is worth.
    Only groups whose columns are actually present in df are included, so
    this reflects the feature set the app was actually built with. Returns
    [Step, WAPE_SKU_month, WAPE_Category_month, MASE, Bias, n_features]."""
    cat_present, extra_present, rows = [], [], []
    steps = []
    for label, cats, extras in FEATURE_GROUPS:
        cats_here = [c for c in cats if c in df.columns]
        extras_here = [c for c in extras if c in df.columns]
        if not cats_here and not extras_here and label != FEATURE_GROUPS[0][0]:
            continue
        cat_present += cats_here
        extra_present += extras_here
        steps.append((label, list(dict.fromkeys(cat_present)), list(dict.fromkeys(extra_present))))

    for label, cats, extras in steps:
        res_df, _, feat_cols, trained, _, _ = run_backtest(
            df, col_product, cats or None, train_years, validation_year,
            fiscal_start_month, [model_type], extra_features=extras)
        if res_df is None:
            continue
        rep = evaluate_backtest(res_df, add_sparsity_feature(df, col_product, train_years),
                                col_product, train_years, category_cols=cats or None,
                                fiscal_start_month=fiscal_start_month)
        s = rep["summary"].loc[model_type]
        grain = rep["by_grain"][model_type]
        cat_grain = next((grain[k] for k in grain.index if k.startswith("Category1") or k.startswith("Category ")), np.nan)
        rows.append({"Step": label, "WAPE (SKU-month)": s["WAPE"],
                     "WAPE (Category-month)": cat_grain, "MASE": s["MASE"],
                     "Bias": s["Bias"], "n_features": len(feat_cols)})
    return pd.DataFrame(rows)


def walk_forward_backtest(df, col_product, col_category, fiscal_start_month, model_types,
                          all_years, min_train_years=2, extra_features=None, feature_base=None):
    """Rolling-origin evaluation: expand the training window one fiscal year
    at a time and score the next untouched year each time
    (train 2021-22 -> test 2023; train 2021-23 -> test 2024; ...).

    This is more honest than a single hold-out -- it shows whether accuracy
    is stable as history grows -- and it directly demonstrates the
    continuous-improvement loop (retrain as new data lands). Returns a long
    DataFrame [ValidationYear, Model, WAPE, MAE, Bias, MASE] with the
    seasonal-naive Baseline included as a Model value each year."""
    all_years = sorted(all_years)
    records = []
    for k in range(min_train_years, len(all_years)):
        train_years = all_years[:k]
        val_year = all_years[k]
        res_df, metrics, _, _, _, _ = run_backtest(
            df, col_product, col_category, train_years, val_year,
            fiscal_start_month, model_types, extra_features=extra_features, feature_base=feature_base)
        if res_df is None:
            continue
        scale = naive_scale(add_sparsity_feature(df, col_product, train_years), col_product, train_years)
        for name in [c[:-5] for c in res_df.columns if c.endswith("_Pred")]:
            m = point_metrics(res_df["Actual"], res_df[f"{name}_Pred"], mase_scale=scale)
            records.append({"ValidationYear": val_year, "Model": name,
                            "WAPE": m["WAPE"], "MAE": m["MAE"], "Bias": m["Bias"], "MASE": m["MASE"]})
    return pd.DataFrame(records)


# The two ways a backtest can be scored. run_backtest builds every validation
# month's lags from the ACTUAL months before it, so it measures a forecast made
# one month ahead. The forecast MELFT receives is made once, at the start of the
# year, and forecast_future feeds its own predictions forward as the lags -- a
# forecast made up to twelve months ahead. Same model, harder test.
HORIZON_LABELS = ("1 month ahead", "12 months ahead")


def horizon_backtest(df, col_product, col_category, train_years, validation_year, fiscal_start_month,
                     model_type, extra_features=None, feature_base=None, quantiles=None):
    """Re-runs the real forecasting path -- train_final_model + forecast_future --
    from history that ends the March before `validation_year`, and scores the
    resulting 12-month forecast against that year's actuals. This is exactly
    what build_everything does for the live forecast, moved back one year, so
    nothing about it can drift from the shipped forecast.

    quantiles: optional levels (e.g. QUANTILES) to also forecast the p10/p50/p90
    range 12 months ahead, so its calibration can be checked at the horizon it
    is actually used at.

    Returns [col_product, FiscalMonth, Actual, Pred] (+ q10/q50/q90 when asked),
    or None when there is no earlier year to train on."""
    validation_year = int(validation_year)
    train_years = sorted(int(y) for y in train_years if int(y) < validation_year)
    if not train_years:
        return None
    hist = df[df["FiscalYear"] < validation_year]
    model, qmodels, feature_cols = train_final_model(
        hist, col_product, col_category, train_years[:-1], train_years[-1], fiscal_start_month,
        model_type, extra_features=extra_features, feature_base=feature_base, quantiles=quantiles)
    if model is None:
        return None
    fc_df = forecast_future(add_sparsity_feature(hist, col_product, train_years), model, col_product,
                            col_category, feature_cols, validation_year, fiscal_start_month,
                            quantile_models=qmodels or None)
    qcols = [c for c in fc_df.columns if c.startswith("q") and c[1:].isdigit()]
    actual = (df[df["FiscalYear"] == validation_year][[col_product, "FiscalMonth", "Units"]]
              .rename(columns={"Units": "Actual"}))
    out = actual.merge(fc_df[[col_product, "FiscalMonth", "Final_Forecast"] + qcols],
                       on=[col_product, "FiscalMonth"], how="left")
    out["Pred"] = out["Final_Forecast"].fillna(0.0)
    return out[[col_product, "FiscalMonth", "Actual", "Pred"] + qcols].reset_index(drop=True)


def horizon_comparison(res_df, model_name, horizon_pred, df, col_product, train_years,
                       category_cols=None, fiscal_start_month=4):
    """Scores the chosen model 1 month ahead (run_backtest's predictions) and 12
    months ahead (horizon_backtest's) on IDENTICAL rows, through the same
    evaluate_backtest used everywhere else.

    Returns (table, scored): table is Metric x HORIZON_LABELS; scored is the
    res_df-shaped frame with one '<label>_Pred' column per horizon."""
    one, twelve = HORIZON_LABELS
    keys = [col_product, "FiscalMonth"]
    scored = res_df[[col_product, "FiscalYear", "FiscalMonth", "Actual", f"{model_name}_Pred"]].rename(
        columns={f"{model_name}_Pred": f"{one}_Pred"})
    scored = scored.merge(horizon_pred[keys + ["Pred"]].rename(columns={"Pred": f"{twelve}_Pred"}),
                          on=keys, how="left")
    scored[f"{twelve}_Pred"] = scored[f"{twelve}_Pred"].fillna(0.0)
    rep = evaluate_backtest(scored, add_sparsity_feature(df, col_product, train_years), col_product,
                            train_years, category_cols=category_cols, fiscal_start_month=fiscal_start_month)
    table = {}
    for lbl in HORIZON_LABELS:
        s, g = rep["summary"].loc[lbl], rep["by_grain"][lbl]
        col = {"WAPE, item x month": s["WAPE"]}
        for name in g.index:
            if name.endswith("x month") and name != "SKU-month":
                col[f"WAPE, {name}"] = g[name]
        if "SKU x quarter" in g.index:
            col["WAPE, item x quarter"] = g["SKU x quarter"]
        col["Bias"] = s["Bias"]
        col["MASE"] = s["MASE"]
        table[lbl] = col
    return pd.DataFrame(table), scored


def train_final_model(df, col_product, col_category, train_years, validation_year, fiscal_start_month,
                      model_type, extra_features=None, feature_base=None, quantiles=None):
    """Fits the point model on train_years + validation_year (all the history we
    have). If `quantiles` is given (e.g. QUANTILES = (0.1, 0.5, 0.9)) it also
    fits one quantile model per level on the IDENTICAL rows and feature columns,
    guaranteeing they line up with the point model at predict time.

    Returns (point_model, {alpha: quantile_model}, feature_cols); the dict is empty
    without `quantiles`, and point_model is None when there are too few rows."""
    feat_df, feature_cols = model_features(df, col_product, col_category, train_years + [validation_year],
                                           fiscal_start_month, feature_base, extra_features)

    final_years = train_years + [validation_year]
    final_train = feat_df[feat_df["FiscalYear"].isin(final_years)].dropna(subset=feature_cols)
    if len(final_train) < 10:
        return None, {}, feature_cols
    model = make_model(model_type)
    model.fit(final_train[feature_cols], final_train["Units"])
    qmodels = {}
    for q in sorted(quantiles or ()):
        qm = make_quantile_model(q)
        qm.fit(final_train[feature_cols], final_train["Units"])
        qmodels[q] = qm
    return model, qmodels, feature_cols


def forecast_future(df, model, col_product, col_category, feature_cols, target_year, fiscal_start_month,
                    sentinel=99, quantile_models=None):
    """Predicts every product for every month of target_year using compact
    per-product NumPy arrays (a rolling window of each product's most recent
    12 months), instead of repeatedly growing and re-sorting a full-size
    dataframe. This matters a lot at scale: the old approach re-ran a full
    sort + groupby + shift over an ever-growing dataframe 12 times, which
    got very slow with thousands of SKUs. This version does O(products)
    NumPy work per month instead of O(entire growing history) pandas work
    per month.

    avg_monthly_units and months_since_last_sale are carried forward with
    running per-product accumulators (seeded from real history, then updated
    using each month's own forecast, mirroring the leakage-safe training-time
    recurrence). distinct_customers_trailing_12m -- if present -- is held at
    its last REAL known value for the whole forecast horizon, since there's
    no way to know future customer activity; this is a known limitation,
    most relevant for months further into the forecast."""
    products = df[col_product].unique()
    n = len(products)

    category_cols = col_category if isinstance(col_category, list) else ([col_category] if col_category else [])
    cat_maps = {c: df.set_index(col_product)[c].to_dict() for c in category_cols if c in df.columns}
    sparsity_map = df.drop_duplicates(col_product).set_index(col_product)["sparsity_coverage"].to_dict() if "sparsity_coverage" in df.columns else {}

    # Features this function recomputes month-by-month from the rolling
    # forecast window. Everything else in feature_cols is either a one-hot
    # category dummy (built from cat_maps) or an "extra" numeric feature
    # (distinct_customers_trailing_12m, reorder cadence signals, ...) that we
    # can't project forward -- those are held flat at each product's last
    # real observed value across the whole horizon (see held_arrays below).
    computed_features = set(FEATURE_BASE)  # includes sparsity_coverage
    held_cols = [c for c in feature_cols
                 if c not in computed_features and c in df.columns
                 and pd.api.types.is_numeric_dtype(df[c])]
    dummy_cols = [c for c in feature_cols if c not in computed_features and c not in held_cols]
    cat_dummy_matrix = np.zeros((n, len(dummy_cols)))
    if dummy_cols and cat_maps:
        dummy_index = {c: j for j, c in enumerate(dummy_cols)}
        for i, p in enumerate(products):
            for cat_col, cmap in cat_maps.items():
                col_name = f"{cat_col.lower()}_{cmap.get(p)}"
                if col_name in dummy_index:
                    cat_dummy_matrix[i, dummy_index[col_name]] = 1

    sparsity_arr = np.array([sparsity_map.get(p, np.nan) for p in products]) if sparsity_map else None

    # Build a (products x 12) matrix of each product's most recent 12
    # historical months, in chronological order, aligned to `products`.
    hist_sorted = df.sort_values([col_product, "FiscalYear", "FiscalMonth"])
    pivot = hist_sorted.pivot_table(index=col_product, columns=["FiscalYear", "FiscalMonth"], values="Units", aggfunc="first")
    pivot = pivot.reindex(products)
    pivot = pivot[sorted(pivot.columns, key=lambda c: (c[0] * 12 + c[1]))]
    if pivot.shape[1] >= 12:
        recent12 = pivot.iloc[:, -12:].to_numpy(dtype=float)
    else:
        pad = np.full((n, 12 - pivot.shape[1]), np.nan)
        recent12 = np.concatenate([pad, pivot.to_numpy(dtype=float)], axis=1)

    # --- Seed avg_monthly_units: running (sum, count) over ALL real history ---
    hist_sum = df.groupby(col_product)["Units"].sum().reindex(products).fillna(0).to_numpy(dtype=float)
    hist_count = df.groupby(col_product)["Units"].count().reindex(products).fillna(0).to_numpy(dtype=float)

    # --- Seed months_since_last_sale: what the recurrence would give a
    # hypothetical row appended right after real history (reuses the same
    # tested function rather than re-deriving the arithmetic by hand) ---
    seed_df = pd.concat([
        df[[col_product, "FiscalYear", "FiscalMonth", "Units"]],
        pd.DataFrame({col_product: products, "FiscalYear": target_year, "FiscalMonth": 0, "Units": 0}),
    ], ignore_index=True)
    seed_df["months_since_last_sale"] = compute_months_since_last_sale(seed_df, col_product, sentinel=sentinel)
    months_since = seed_df[(seed_df["FiscalYear"] == target_year) & (seed_df["FiscalMonth"] == 0)] \
        .set_index(col_product)["months_since_last_sale"].reindex(products).fillna(sentinel).to_numpy(dtype=float)

    # --- extra numeric features (customer counts, reorder cadence): hold
    #     each at its last known real value for the whole forecast horizon ---
    held_arrays = {}
    for c in held_cols:
        last = df.dropna(subset=[c]).sort_values([col_product, "FiscalYear", "FiscalMonth"]) \
            .groupby(col_product)[c].last().reindex(products).fillna(0)
        held_arrays[c] = last.to_numpy(dtype=float)

    forecast_rows = []
    for fm in range(1, 13):
        lag_1, lag_2, lag_3, lag_12 = recent12[:, -1], recent12[:, -2], recent12[:, -3], recent12[:, 0]
        roll_mean_3 = np.nanmean(recent12[:, -3:], axis=1)
        roll_mean_6 = np.nanmean(recent12[:, -6:], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            avg_monthly_units = np.where(hist_count > 0, hist_sum / hist_count, 0)

        calendar_month = fiscal_to_calendar_month(fm, fiscal_start_month)
        month_sin = np.full(n, np.sin(2 * np.pi * calendar_month / 12))
        month_cos = np.full(n, np.cos(2 * np.pi * calendar_month / 12))

        feat_dict = {
            "lag_1": lag_1, "lag_2": lag_2, "lag_3": lag_3, "lag_12": lag_12,
            "roll_mean_3": roll_mean_3, "roll_mean_6": roll_mean_6,
            "avg_monthly_units": avg_monthly_units, "months_since_last_sale": months_since,
            "month_sin": month_sin, "month_cos": month_cos,
        }
        if sparsity_arr is not None:
            feat_dict["sparsity_coverage"] = sparsity_arr
        for c, arr in held_arrays.items():
            feat_dict[c] = arr
        # Build the frame in one shot (numeric feats + one-hot dummies) rather
        # than inserting columns one by one, which fragments the DataFrame.
        X = pd.concat([
            pd.DataFrame(feat_dict),
            pd.DataFrame(cat_dummy_matrix, columns=dummy_cols),
        ], axis=1)
        X = X.reindex(columns=feature_cols, fill_value=0)

        target_rows = pd.DataFrame({col_product: products, "FiscalYear": target_year, "FiscalMonth": fm})
        baseline_preds = vectorized_seasonal_naive(df, target_rows, col_product)

        valid_mask = ~np.isnan(lag_1) & ~np.isnan(lag_2) & ~np.isnan(lag_3) & ~np.isnan(roll_mean_3)
        ml_preds = np.full(n, np.nan)
        X_valid = X.loc[valid_mask].fillna(0)
        if model is not None and valid_mask.any():
            ml_preds[valid_mask] = np.maximum(0, model.predict(X_valid))

        final_preds = np.where(~np.isnan(ml_preds), ml_preds, baseline_preds)

        # Quantile models (optional): predicted on the SAME feature rows as the
        # point model. Where the point model couldn't run (missing lags), the
        # band collapses onto the fallback forecast rather than inventing a range.
        q_cols_this = {}
        if quantile_models:
            for q, qm in quantile_models.items():
                qp = final_preds.astype(float).copy()
                if valid_mask.any():
                    qp[valid_mask] = np.maximum(0, qm.predict(X_valid))
                q_cols_this[q] = qp

        for i, p in enumerate(products):
            row = [p, target_year, fm, baseline_preds[i],
                   (None if np.isnan(ml_preds[i]) else ml_preds[i]), final_preds[i]]
            for q in sorted(q_cols_this):
                row.append(q_cols_this[q][i])
            forecast_rows.append(row)

        # Carry every running feature forward using THIS month's own forecast,
        # exactly mirroring the leakage-safe training-time recurrence (next
        # month's features may use this month's outcome; this month's could not).
        recent12 = np.concatenate([recent12[:, 1:], final_preds.reshape(-1, 1)], axis=1)
        hist_sum = hist_sum + final_preds
        hist_count = hist_count + 1
        sold_this_month = final_preds > 0
        months_since = np.where(sold_this_month, 1, np.minimum(months_since + 1, sentinel))

    cols = [col_product, "FiscalYear", "FiscalMonth", "Baseline_Forecast", "ML_Forecast", "Final_Forecast"]
    if quantile_models:
        qcols = [f"q{int(round(q * 100))}" for q in sorted(quantile_models)]
        cols += qcols
    out = pd.DataFrame(forecast_rows, columns=cols)
    if quantile_models and len(qcols) >= 2:
        # Enforce non-crossing quantiles row-by-row (q10 <= q50 <= q90).
        out[qcols] = np.sort(out[qcols].to_numpy(dtype=float), axis=1)
    return out

# =============================================================================
#  Category-level (aggregate-grain) forecasting
# =============================================================================
# Two ways to get a category number: sum the item forecasts up to it
# ("bottom-up"), or aggregate the history first and forecast the category
# series itself ("direct"). Planning happens at the category grain, so it is
# worth knowing which is better -- and measuring it is what justifies the
# bottom-up forecast being the plan of record.
#
# Everything below reuses the item-level pipeline unchanged: add_features,
# run_backtest, train_final_model and forecast_future are all generic over
# their grouping column, so "forecast a category" is literally the same code
# with `col_product` replaced by `Category1`.

UNKNOWN_LEVEL = "(unknown)"


def aggregate_to_level(df, level_col, value_cols=("Units", "Revenue")):
    """Rolls an item-level fiscal-monthly frame up to `level_col` x FiscalYear
    x FiscalMonth, re-filling the grid with zeros so every level has a
    complete 12-month series in every fiscal year present. That zero-fill is
    what lets the same lag/rolling features be computed here -- a category
    with a gap month must read as zero demand, not as a missing row.

    Items with no category are grouped under UNKNOWN_LEVEL rather than
    dropped, so the level totals still add up to the catalog total."""
    if level_col not in df.columns:
        raise ValueError(f"'{level_col}' isn't a column on this frame.")
    d = df.copy()
    d[level_col] = d[level_col].fillna(UNKNOWN_LEVEL).astype(str).replace("nan", UNKNOWN_LEVEL)
    keep = [c for c in value_cols if c in d.columns]
    agg = d.groupby([level_col, "FiscalYear", "FiscalMonth"], as_index=False)[keep].sum()
    levels = agg[level_col].unique()
    years = sorted(agg["FiscalYear"].unique())
    idx = pd.MultiIndex.from_product([levels, years, range(1, 13)],
                                     names=[level_col, "FiscalYear", "FiscalMonth"])
    out = agg.set_index([level_col, "FiscalYear", "FiscalMonth"]).reindex(idx, fill_value=0).reset_index()
    return out.sort_values([level_col, "FiscalYear", "FiscalMonth"]).reset_index(drop=True)


def level_map_from(df, col_product, level_col):
    """{product -> level} lookup, with missing categories mapped to
    UNKNOWN_LEVEL so no item silently vanishes from a roll-up."""
    m = df.drop_duplicates(col_product).set_index(col_product)[level_col]
    return m.fillna(UNKNOWN_LEVEL).astype(str).replace("nan", UNKNOWN_LEVEL).to_dict()


def rollup_backtest_to_level(res_df, col_product, level_col, level_map, pred_col):
    """Sums an item-level backtest result up to level x month -- the
    'bottom-up' forecast, scored at the grain planning happens at."""
    r = res_df.copy()
    r[level_col] = r[col_product].map(level_map).fillna(UNKNOWN_LEVEL)
    return r.groupby([level_col, "FiscalYear", "FiscalMonth"], as_index=False).agg(
        Actual=("Actual", "sum"), Pred=(pred_col, "sum"))


def level_accuracy_breakdown(res_df, df, col_product, level_col, pred_col, top_n_contributors=3):
    """Per-category accuracy on the held-out year, plus the two diagnostics
    that explain a bad number instead of just reporting it:

      * `Top item share` -- the largest single item's share of that category's
        actual volume. A category at 80% is one project SKU, not a demand
        stream, and no model will forecast its reorder timing. This is the
        honest reason such a category's WAPE is high.
      * the contributors frame -- which items produced the absolute error.

    Returns (breakdown, contributors)."""
    level_map = level_map_from(df, col_product, level_col)
    r = res_df.copy()
    r[level_col] = r[col_product].map(level_map).fillna(UNKNOWN_LEVEL)
    r["_ae"] = (r["Actual"] - r[pred_col]).abs()

    rows = []
    for lv, g in r.groupby(level_col, observed=True):
        by_item = g.groupby(col_product).agg(a=("Actual", "sum"), p=(pred_col, "sum"), ae=("_ae", "sum"))
        actual = float(by_item["a"].sum())
        top_share = float(by_item["a"].max() / actual) if actual > 0 else np.nan
        # Scored at the level's own monthly grain, matching how a category
        # forecast is actually consumed (a monthly S&OP number, not a total).
        by_month = g.groupby("FiscalMonth").agg(a=("Actual", "sum"), p=(pred_col, "sum"))
        rows.append({
            level_col: lv,
            "Actual": actual,
            "Forecast": float(by_item["p"].sum()),
            "WAPE": wape(by_month["a"], by_month["p"]),
            "Bias": forecast_bias(by_month["a"], by_month["p"]),
            "Items sold": int((by_item["a"] > 0).sum()),
            "Top item share": top_share,
        })
    breakdown = pd.DataFrame(rows).sort_values("Actual", ascending=False).reset_index(drop=True)

    contrib = (r.groupby([level_col, col_product], observed=True)
               .agg(Actual=("Actual", "sum"), Forecast=(pred_col, "sum"), AbsError=("_ae", "sum"))
               .reset_index())
    contrib = (contrib.sort_values([level_col, "AbsError"], ascending=[True, False])
               .groupby(level_col, observed=True).head(top_n_contributors).reset_index(drop=True))
    total_ae = r["_ae"].sum()
    contrib["Share of total error"] = contrib["AbsError"] / total_ae if total_ae > 0 else np.nan
    return breakdown, contrib


def level_method_comparison(df, col_product, level_col, train_years, validation_year,
                            fiscal_start_month, model_type, extra_features=None,
                            feature_base=None, category_cols=None, sku_res_df=None):
    """Scores forecasting DIRECTLY at `level_col` grain against summing an
    item-level forecast up to the same grain, on the held-out
    `validation_year`.

    sku_res_df: an existing item-level backtest for the SAME train/validation
    split. The app passes the one from its own build so the bottom-up row
    matches the headline WAPE exactly rather than being a near-miss
    re-computation; pass None to run that backtest here (e.g. for an earlier
    origin year in a stability check).

    Returns (table, detail):
      table  -- [Method, Approach, WAPE, Bias, Forecast, Actual], one row per method
      detail -- [Level, FiscalMonth, Method, Actual, Pred] for charting

    The item-level seasonal-naive baseline is deliberately NOT reported here:
    it scales each series by its own clipped YoY growth, a rule designed for
    sparse item series, so applying it to a dense category series measures
    the rule's mismatch rather than anything about the grain. classical.py
    supplies the plain "repeat last year" seasonal naive instead."""
    level_df = aggregate_to_level(df, level_col)
    lv_res, _, _, _, _, _ = run_backtest(
        level_df, level_col, None, list(train_years), validation_year, fiscal_start_month,
        [model_type], feature_base=feature_base)
    if lv_res is None:
        raise ValueError(f"Not enough history to backtest at {level_col} grain.")

    if sku_res_df is None:
        sku_res_df, _, _, _, _, _ = run_backtest(
            df, col_product, category_cols, list(train_years), validation_year, fiscal_start_month,
            [model_type], extra_features=extra_features, feature_base=feature_base)
    if sku_res_df is None or f"{model_type}_Pred" not in sku_res_df.columns:
        raise ValueError("No item-level backtest available to roll up.")

    level_map = level_map_from(df, col_product, level_col)
    bu = rollup_backtest_to_level(sku_res_df, col_product, level_col, level_map, f"{model_type}_Pred")

    series = [("Bottom-up (sum of item forecasts)", "Bottom-up", bu)]
    for pred_col, label in [(f"{model_type}_Pred", f"{model_type} trained at {level_col}"),
                            ("Croston (SBA)_Pred", f"Croston / SBA at {level_col}")]:
        if pred_col in lv_res.columns:
            s = lv_res[[level_col, "FiscalYear", "FiscalMonth", "Actual", pred_col]].rename(
                columns={pred_col: "Pred"})
            series.append((label, "Direct", s))

    rows, details = [], []
    for label, approach, s in series:
        rows.append({"Method": label, "Approach": approach,
                     "WAPE": wape(s["Actual"], s["Pred"]), "Bias": forecast_bias(s["Actual"], s["Pred"]),
                     "Forecast": float(s["Pred"].sum()), "Actual": float(s["Actual"].sum())})
        d = s.rename(columns={level_col: "Level"})[["Level", "FiscalMonth", "Actual", "Pred"]].copy()
        d["Method"] = label
        details.append(d)
    return pd.DataFrame(rows), pd.concat(details, ignore_index=True)


def level_direct_forecast(df, level_col, train_years, validation_year, fiscal_start_month,
                          model_type, forecast_year, feature_base=None):
    """Trains the model on the level-aggregated history and forecasts
    `forecast_year` directly at that grain -- the 'direct' counterpart to
    summing the item forecasts. Reuses train_final_model + forecast_future
    unchanged (both are generic over the grouping column).

    Returns a [level, Direct] frame of annual totals, or None if the level
    series is too short to train on."""
    level_df = aggregate_to_level(df, level_col)
    model, _, feature_cols = train_final_model(
        level_df, level_col, None, list(train_years), validation_year, fiscal_start_month,
        model_type, feature_base=feature_base)
    if model is None:
        return None
    ready = add_sparsity_feature(level_df, level_col, list(train_years) + [validation_year])
    fc_out = forecast_future(ready, model, level_col, None, feature_cols, forecast_year, fiscal_start_month)
    return (fc_out.groupby(level_col, as_index=False)["Final_Forecast"].sum()
            .rename(columns={"Final_Forecast": "Direct"}))
