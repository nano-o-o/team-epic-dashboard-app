"""
Shared setup for the model-improvement experiments (improve_*.py).

Rebuilds the dashboard's DEFAULTS build without Streamlit -- same data prep,
same features (experimental B2B off, bias correction off), LightGBM (Tweedie),
every item scored, discontinued ones included -- and caches, per validation year:

  * res_df rows   : the exact item-month rows the app scores (run_backtest's
                    validation rows), so every experiment is scored on the
                    same rows as the app's headline numbers
  * hz            : the 12-month-ahead forecast (horizon_backtest), i.e. the
                    forecast as it is actually used, made the March before
  * sigma         : each item's error sigma from the PRIOR year's 12-month-ahead
                    backtest (the app's honest sigma), in-sample fallback if none
  * trailing      : units sold in the 12 months before the validation year

Nothing here writes to the app. Run from the project root:
    .venv/Scripts/python.exe experiments/improve_00_baseline.py
"""
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import forecast_core as fc  # noqa: E402
import inventory as inv  # noqa: E402

CACHE = os.path.join(ROOT, "experiments", "_cache")
os.makedirs(CACHE, exist_ok=True)

P = "Item Internal ID"
COL_UNITS, COL_PRICE, COL_DATE = "Quantity", "Base Price", "Sales Order Date"
COL_STATUS, COL_CUSTOMER, COL_CODE = "Sales Order Status", "Customer Internal ID", "Item Code"
FISCAL_START = 4
MODEL = "LightGBM (Tweedie)"
PRED = f"{MODEL}_Pred"
ALL_YEARS = [2021, 2022, 2023, 2024]
LEAD, REVIEW, SERVICE, LEGACY = 2.0, 3.0, 0.95, 1.5
SERVICE_LEVELS = (0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99)
FLAT_MONTHS = (1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12)
TIER_BINS, TIER_LABELS = [-0.01, 0.25, 0.75, 1.01], ["Sparse", "Medium", "Dense"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _cached(name, fn):
    # Pickle is safe here: the cache only ever holds files these scripts wrote
    # themselves into experiments/_cache/ (a local, unshared folder). Delete the
    # folder to force a rebuild after any change to the data or forecast_core.
    path = os.path.join(CACHE, name + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    t = time.time()
    out = fn()
    with open(path, "wb") as f:
        pickle.dump(out, f)
    log(f"built {name} in {time.time() - t:.0f}s")
    return out


# --------------------------------------------------------------------------- data
def _prepare():
    raw = pd.read_csv(os.path.join(ROOT, "data", "cleaned_data.csv"))
    raw.columns = [c.strip() if isinstance(c, str) else c for c in raw.columns]
    raw, _ = fc.clean_dates(raw, COL_DATE)
    raw[P] = raw[P].astype(str)
    raw, _ = fc.drop_duplicate_lines(raw)
    raw = fc.filter_by_status(raw, COL_STATUS, None)          # app default: every status
    raw = fc.compute_revenue(raw, COL_PRICE, COL_UNITS, revenue_col="Revenue")

    im = pd.read_excel(os.path.join(ROOT, "data", "Item_master.xlsx"))
    im.columns = [c.strip() if isinstance(c, str) else c for c in im.columns]
    attr = {"Category1": "Product Category 1", "Category2": "Product Category 2",
            "ProductionStatus": "Production Status", "ListPrice": "List price"}
    raw = fc.merge_item_master(raw, P, im, P, attr)
    raw, _ = fc.drop_unpriced_items(raw, P, im, P, "List price")
    raw["ListPrice"] = pd.to_numeric(raw["ListPrice"], errors="coerce")
    passthrough = {"Item Code": COL_CODE, "Category1": "Category1", "Category2": "Category2",
                   "ProductionStatus": "ProductionStatus", "ListPrice": "ListPrice"}
    df = fc.aggregate_to_fiscal_monthly(raw, P, COL_UNITS, "Revenue", COL_DATE, FISCAL_START,
                                        passthrough_cols=passthrough)
    df_full = df.copy()
    # Discontinued items stay in: the app scores its backtests on every item (dropping
    # them by today's status is survivorship look-ahead) and drops them only from the
    # forward forecast, which these experiments don't make.
    df["ListPrice"] = pd.to_numeric(df["ListPrice"], errors="coerce").fillna(0)

    cust = fc.compute_distinct_customers_trailing(raw, P, COL_CUSTOMER, COL_DATE, FISCAL_START)
    keys = [P, "FiscalYear", "FiscalMonth"]
    df = df.merge(cust, on=keys, how="left")
    df["distinct_customers_trailing_12m"] = df["distinct_customers_trailing_12m"].fillna(0)
    return dict(raw=raw, df=df, df_full=df_full,
                extra=["ListPrice", "distinct_customers_trailing_12m"],
                cats=["Category1", "Category2"], base=list(fc.FEATURE_BASE))


def prepared():
    return _cached("prepared", _prepare)


def price_map(prep):
    return prep["df"].drop_duplicates(P).set_index(P)["ListPrice"].to_dict()


def train_years_for(vy):
    return [y for y in ALL_YEARS if y < vy]


# ---------------------------------------------------------------------- baseline
def _baseline(vy):
    prep = prepared()
    df, extra, cats, base = prep["df"], prep["extra"], prep["cats"], prep["base"]
    tys = train_years_for(vy)
    res_df, metrics, *_ = fc.run_backtest(df, P, cats, tys, vy, FISCAL_START, [MODEL],
                                          extra_features=extra, feature_base=base)
    hz = fc.horizon_backtest(df, P, cats, tys, vy, FISCAL_START, MODEL, extra_features=extra,
                             feature_base=base, quantiles=fc.QUANTILES)
    rows = res_df[[P, "FiscalYear", "FiscalMonth", "Actual", PRED]].rename(columns={PRED: "pred_1m"})
    rows = rows.merge(hz[[P, "FiscalMonth", "Pred", "q10", "q50", "q90"]], on=[P, "FiscalMonth"], how="left")
    rows = rows.rename(columns={"Pred": "base"})
    for c in ("base", "q10", "q50", "q90"):
        rows[c] = rows[c].fillna(0.0)

    prior, prior_train = vy - 1, [y for y in tys if y < vy - 1]
    hp = None
    if prior_train:
        hp = fc.horizon_backtest(df, P, cats, prior_train, prior, FISCAL_START, MODEL,
                                 extra_features=extra, feature_base=base)
    if hp is not None:
        sigma = inv.forecast_error_sigma(hp.rename(columns={"Pred": PRED}), MODEL, P).to_dict()
        basis = f"FY{prior}"
    else:
        sigma = inv.forecast_error_sigma(rows.rename(columns={"base": PRED}), MODEL, P).to_dict()
        basis = f"FY{vy} in-sample (no earlier year)"
    pre = df[df["FiscalYear"] < vy]
    t = pre["FiscalYear"] * 12 + pre["FiscalMonth"]
    trailing = pre[t > t.max() - 12].groupby(P)["Units"].sum().to_dict()
    return dict(vy=vy, tys=tys, rows=rows, hp=hp, sigma=sigma, sigma_basis=basis, trailing=trailing)


def baseline(vy):
    return _cached(f"baseline_FY{vy}", lambda: _baseline(vy))


def fair_sigma(b):
    """Sigma for the stocking replay that uses nothing from the replayed year.
    Where a prior-year 12-month-ahead backtest exists (FY2024) this is exactly the
    app's honest sigma. Where it doesn't (FY2023), the app falls back to errors
    measured ON the replayed year -- a leak that lifts its FY2023 fill rate to
    ~99% -- so use each item's monthly demand std over the 24 months before the
    year instead (on FY2024 that proxy gives 80.8% vs the app's 81.1% at the same
    budget, so it is a like-for-like stand-in)."""
    if b["hp"] is not None:
        return b["sigma"]
    df = prepared()["df"]
    pre = df[df["FiscalYear"] < b["vy"]]
    t = pre["FiscalYear"] * 12 + pre["FiscalMonth"]
    return pre[t > t.max() - 24].groupby(P)["Units"].std().to_dict()


# ----------------------------------------------------------------------- scoring
_TIERS = {}


def tiers(df, tys):
    """product -> Sparse / Medium / Dense from the training window (cached: the
    groupby-apply behind it takes seconds and scoring calls it many times)."""
    key = (id(df), tuple(tys))
    if key not in _TIERS:
        cov = fc.add_sparsity_feature(df, P, tys).drop_duplicates(P).set_index(P)["sparsity_coverage"]
        _TIERS[key] = pd.cut(cov, bins=TIER_BINS, labels=TIER_LABELS).astype(str)
    return _TIERS[key]


def score(rows, pred_col, df, tys):
    """Point-forecast scorecard on the app's validation rows."""
    r = rows[[P, "FiscalMonth", "Actual", pred_col]].copy()
    attrs = df.drop_duplicates(P).set_index(P)[["Category1"]]
    r = r.join(attrs, on=P)
    r["Q"] = (r["FiscalMonth"] - 1) // 3 + 1
    r["tier"] = r[P].map(tiers(df, tys))
    out = {
        "WAPE item-month": fc.wape(r["Actual"], r[pred_col]),
        "Bias": fc.forecast_bias(r["Actual"], r[pred_col]),
        "WAPE Cat1-month": fc.aggregated_wape(r, pred_col, ["Category1", "FiscalMonth"]),
        "WAPE item-quarter": fc.aggregated_wape(r, pred_col, [P, "Q"]),
        "WAPE item-year": fc.aggregated_wape(r, pred_col, [P]),
        "Total units": float(r[pred_col].sum()),
    }
    for t in TIER_LABELS:
        g = r[r["tier"] == t]
        out[f"WAPE {t}"] = fc.wape(g["Actual"], g[pred_col])
    return out


def forecast_frontier(rows, pred_col, sigma, trailing, price, label, service_levels=SERVICE_LEVELS):
    """The app's forecast-driven policy family (z * sigma * sqrt(P)), ordering to
    `pred_col`, replayed on the validation year. Also returns the flat-rule family
    for that forecast. Family names are prefixed with `label`."""
    fc_like = rows[[P, "FiscalMonth", pred_col]].rename(columns={pred_col: "Final_Forecast"})
    fr = inv.policy_frontier(rows, fc_like, P, sigma, lead_time_months=LEAD, review_period_months=REVIEW,
                             service_levels=service_levels, flat_months=FLAT_MONTHS, list_price=price,
                             trailing_actual_by_product=trailing)
    fr["Family"] = fr["Family"].map({inv.FAMILY_MODEL: f"{label}", inv.FAMILY_FLAT: f"{label} | flat rule"})
    return fr


def replay_policies(rows, policies, price):
    """policies: {label: (family, setting, {product -> S})}. Same replay and
    scoring as the app (demand-only items, lost sales, quarterly review)."""
    s = {k: v[2] for k, v in policies.items()}
    summary, _ = inv.service_level_comparison(rows, P, s, lead_time_months=LEAD,
                                              review_period_months=REVIEW, list_price=price)
    meta = pd.DataFrame([(k, v[0], v[1]) for k, v in policies.items()], columns=["Policy", "Family", "Setting"])
    return meta.merge(summary, on="Policy")


def fill_at(frontier, family, budget):
    return inv.fill_rate_at_budget(frontier, budget, family=family)


def money(v):
    return f"฿{v / 1e9:.2f}bn" if v >= 1e9 else f"฿{v / 1e6:,.0f}M"


def pct(v):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1%}"
