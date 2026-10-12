"""Idea 3 -- forecast per customer x item: "will this customer reorder this item,
and how much?", then add up to the item.

Structure (direct 12-month model, no recursion):
  * Pair model   one row per (customer, item) active in the 24 months before an
                 origin month; features describe that relationship (volume over
                 3/6/12/24 months, order months, months since last order, typical
                 gap between orders and how overdue the next one is, order size),
                 the customer (volume, breadth, recency) and the item (volume,
                 customer count). Target = units the pair buys in the next 12 months.
  * New-pair model  one row per item: units bought in the next 12 months by
                 customers who had NOT bought that item in the prior 24 months
                 (new or returning buyers) -- the part a pair model cannot see.
  * Item forecast = sum of pair predictions + new-pair prediction, spread over the
                 12 months either flat or with the app's own monthly shape.

Training origins are every month from month 6 to the last one whose 12-month
target window closes before the validation year. Validated at FY2023 (origin =
month 24) and FY2024 (origin = month 36). Sigma for the FY2024 replay comes from
the pair model's own FY2023 errors; FY2023 has no earlier year, so it uses the
baseline's. LightGBM Tweedie, random_state 42.
"""
import os

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from improve_common import (prepared, baseline, fair_sigma, score, forecast_frontier, fill_at, price_map,
                            money, pct, log, P, COL_CUSTOMER, COL_UNITS, COL_DATE, FISCAL_START, SERVICE, ROOT)
import forecast_core as fc  # after improve_common, which puts the project root on sys.path

OUT = os.path.join(ROOT, "experiments", "_results")
os.makedirs(OUT, exist_ok=True)
prep = prepared()
df, price = prep["df"], price_map(prep)
keep_items = set(df[P].unique())                      # every item, discontinued included, as the app's backtests

raw = prep["raw"][[COL_CUSTOMER, P, COL_DATE, COL_UNITS]].copy()
raw = raw[raw[P].isin(keep_items)]
fy, fm = fc.to_fiscal(raw[COL_DATE], FISCAL_START)
raw["t"] = (np.asarray(fy) - 2021) * 12 + np.asarray(fm)        # Apr 2021 = 1 ... Mar 2025 = 48
raw["C"] = raw[COL_CUSTOMER].astype(str)
pm = raw.groupby(["C", P, "t"], as_index=False)[COL_UNITS].sum().rename(columns={COL_UNITS: "q"})
pm = pm[pm["q"] > 0]
log(f"pair-months: {len(pm):,}; pairs: {pm[['C', P]].drop_duplicates().shape[0]:,}")


def features(t0):
    """Pair rows (active in (t0-24, t0]) and item rows, using data with t <= t0 only."""
    h = pm[pm["t"] <= t0]
    w = lambda k: h[h["t"] > t0 - k]
    g = h.groupby(["C", P])
    pair = pd.DataFrame({
        "first_t": g["t"].min(), "last_t": g["t"].max(), "n_all": g["t"].count(), "q_all": g["q"].sum()})
    for k in (3, 6, 12, 24):
        wk = w(k).groupby(["C", P])
        pair[f"q{k}"] = wk["q"].sum()
        if k in (12, 24):
            pair[f"n{k}"] = wk["t"].count()
    pair = pair.fillna({c: 0 for c in pair.columns if c.startswith(("q", "n"))})
    pair = pair[pair["q24"] > 0].reset_index()
    pair["since_last"] = t0 - pair["last_t"]
    pair["since_first"] = t0 - pair["first_t"]
    pair["mean_gap"] = np.where(pair["n_all"] > 1, (pair["last_t"] - pair["first_t"]) / (pair["n_all"] - 1), -1)
    pair["overdue"] = np.where(pair["mean_gap"] > 0, pair["since_last"] / pair["mean_gap"], -1)
    pair["qty_per_order"] = pair["q_all"] / pair["n_all"]
    c12 = w(12).groupby("C").agg(cust_q12=("q", "sum"), cust_items12=(P, "nunique"))
    c_last = h.groupby("C")["t"].max().rename("cust_last")
    i12 = w(12).groupby(P).agg(item_q12=("q", "sum"), item_cust12=("C", "nunique"))
    i24 = w(24).groupby(P).agg(item_q24=("q", "sum"))
    pair = pair.join(c12, on="C").join(c_last, on="C").join(i12, on=P).join(i24, on=P)
    pair["cust_since_last"] = t0 - pair["cust_last"]
    pair = pair.drop(columns=["first_t", "last_t", "cust_last"]).fillna(0)

    # item rows: recent volume from buyers who were new to the item in the last 12 months
    first_pair = pm.groupby(["C", P])["t"].min().rename("pair_first").reset_index()
    hh = h.merge(first_pair, on=["C", P])
    new12 = hh[(hh["t"] > t0 - 12) & (hh["pair_first"] > t0 - 12)].groupby(P)["q"].sum().rename("new_q12")
    items = pd.DataFrame(index=pd.Index(sorted(keep_items), name=P))
    items = items.join(i12).join(i24).join(new12).fillna(0).reset_index()
    return pair, items


def targets(t0, pair, items):
    fut = pm[(pm["t"] > t0) & (pm["t"] <= t0 + 12)]
    pair_y = fut.groupby(["C", P], as_index=False)["q"].sum()
    y_pair = pair[["C", P]].merge(pair_y, on=["C", P], how="left")["q"].fillna(0).to_numpy(dtype=float)
    # Future volume from buyers who were NOT active pairs at the origin.
    fut_new = fut.merge(pair[["C", P]].assign(_active=True), on=["C", P], how="left")
    fut_new = fut_new[fut_new["_active"].isna()]
    y_new = items[P].map(fut_new.groupby(P)["q"].sum()).fillna(0).to_numpy(dtype=float)
    return y_pair, y_new


PAIR_X = ["n_all", "q_all", "q3", "q6", "q12", "q24", "n12", "n24", "since_last", "since_first", "mean_gap",
          "overdue", "qty_per_order", "cust_q12", "cust_items12", "cust_since_last", "item_q12", "item_cust12",
          "item_q24"]
ITEM_X = ["item_q12", "item_cust12", "item_q24", "new_q12"]


def model():
    return LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                         objective="tweedie", tweedie_variance_power=1.3, random_state=42, verbosity=-1, n_jobs=-1)


def annual_forecast(T):
    """Item -> forecast units over months T+1..T+12, trained on origins 6..T-12."""
    Xp, yp, Xi, yi = [], [], [], []
    for t0 in range(6, T - 12 + 1):
        pair, items = features(t0)
        y_pair, y_new = targets(t0, pair, items)
        Xp.append(pair[PAIR_X]); yp.append(y_pair); Xi.append(items[ITEM_X]); yi.append(y_new)
    Xp, yp, Xi, yi = pd.concat(Xp), np.concatenate(yp), pd.concat(Xi), np.concatenate(yi)
    log(f"  origin {T}: training on {len(Xp):,} pair rows, {len(Xi):,} item rows")
    mp, mi = model().fit(Xp, yp), model().fit(Xi, yi)
    pair, items = features(T)
    pair["pred"] = np.maximum(0, mp.predict(pair[PAIR_X]))
    items["pred_new"] = np.maximum(0, mi.predict(items[ITEM_X]))
    out = items.set_index(P)["pred_new"].add(pair.groupby(P)["pred"].sum(), fill_value=0)
    return out, pair.groupby(P)["pred"].sum()


all_rows, sigma_pair_prev = [], None
for vy in (2023, 2024):
    T = (vy - 2021) * 12
    b = baseline(vy)
    rows = b["rows"].copy()
    tys, trailing = b["tys"], b["trailing"]
    sigma0 = fair_sigma(b)
    base_fr = forecast_frontier(rows, "base", sigma0, trailing, price, "Baseline")
    budget = float(base_fr[(base_fr["Family"] == "Baseline") & (base_fr["Setting"].round(2) == SERVICE)]
                   ["Avg inventory value"].iloc[0])
    ann, pair_only = annual_forecast(T)
    a = rows[P].map(ann).fillna(0.0).to_numpy()
    base_tot = rows.groupby(P)["base"].transform("sum").to_numpy()
    shape = np.where(base_tot > 0, rows["base"].to_numpy() / np.maximum(base_tot, 1e-9), 1 / 12)
    variants = {
        "Baseline (app, 12 months ahead)": rows["base"].to_numpy(),
        "Pair model, flat months": a / 12,
        "Pair model, app's monthly shape": a * shape,
        "Pair model without new-buyer part, flat": rows[P].map(pair_only).fillna(0.0).to_numpy() / 12,
        "50/50 blend of pair model and app": 0.5 * (a * shape) + 0.5 * rows["base"].to_numpy(),
    }
    sigma = sigma0 if sigma_pair_prev is None else {**sigma0, **sigma_pair_prev}
    for name, pred in variants.items():
        rows["_p"] = pred
        s = score(rows, "_p", df, tys)
        sig = sigma0 if name.startswith("Baseline") else sigma
        fr = forecast_frontier(rows, "_p", sig, trailing, price, "v")
        m95 = fr[(fr["Family"] == "v") & (fr["Setting"].round(2) == SERVICE)].iloc[0]
        s.update({"Variant": name, "FY": vy, "Fill @ baseline budget": fill_at(fr, "v", budget),
                  "95% plan fill": m95["Fill rate"], "95% plan budget": m95["Avg inventory value"]})
        all_rows.append(s)
        log(f"  FY{vy} {name}: WAPE {pct(s['WAPE item-month'])}, item-year {pct(s['WAPE item-year'])}, "
            f"fill @ budget {pct(s['Fill @ baseline budget'])}")
    # this year's errors become next year's honest sigma for the pair model (flat split)
    act = df[df["FiscalYear"] == vy][[P, "Units"]]
    err = act["Units"] - act[P].map(ann).fillna(0.0) / 12
    sigma_pair_prev = err.groupby(act[P]).std().dropna().to_dict()

res = pd.DataFrame(all_rows)
res.to_csv(os.path.join(OUT, "03_customer_item.csv"), index=False)
cols = ["WAPE item-month", "Bias", "WAPE Cat1-month", "WAPE item-quarter", "WAPE item-year", "WAPE Sparse",
        "WAPE Dense", "Total units", "Fill @ baseline budget", "95% plan fill", "95% plan budget"]
for vy in (2024, 2023):
    t = res[res["FY"] == vy].set_index("Variant")[cols]
    print(f"\n=== FY{vy} ===")
    print(t.to_string(formatters={c: ((lambda v: f"{v:,.0f}") if c == "Total units" else money if c.endswith("plan budget") else pct)
                                  for c in cols}))
