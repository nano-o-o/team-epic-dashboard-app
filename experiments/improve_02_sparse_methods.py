"""Idea 2 -- different methods for different items (ADIDA / temporal aggregation).

Keeps the app's ML forecast for Dense (and optionally Medium) items and swaps in
a simple rate forecast for Sparse items, which the model scores at ~120% WAPE.
The rate is spread flat over the 12 months: for an item that sells in fewer
than 1 month in 4, predicting WHICH month is guesswork, and the stocking policy
only needs the level over the protection period.

  Rate methods (all use only history before the validation year):
    12m mean / 24m mean      average monthly units over the last 12 / 24 months
    Croston SBA (monthly)    the classic intermittent-demand rate
    ADIDA quarterly SBA      aggregate to quarters, SBA on the quarterly series, / 3
    ADIDA annual SES         yearly totals, simple exponential smoothing (alpha 0.3), / 12
    ML annual, flattened     the app's own 12-month forecast, summed and spread flat
                             (isolates "timing" from "level")

Each hybrid's sigma for the switched items comes from that method's own errors on
the PRIOR year, so the stocking replay stays honest. Scored at FY2024 and FY2023.
"""
import os

import numpy as np
import pandas as pd

from improve_common import (prepared, baseline, fair_sigma, score, tiers, forecast_frontier, fill_at, price_map,
                            money, pct, log, P, SERVICE, ROOT)
import forecast_core as fc  # after improve_common, which puts the project root on sys.path

OUT = os.path.join(ROOT, "experiments", "_results")
os.makedirs(OUT, exist_ok=True)
prep = prepared()
df, price = prep["df"], price_map(prep)


def rates(vy):
    """{method: Series(product -> monthly rate)} from history before vy."""
    hist = df[df["FiscalYear"] < vy]
    t = hist["FiscalYear"] * 12 + hist["FiscalMonth"]
    tmax = t.max()
    out = {
        "12m mean": hist[t > tmax - 12].groupby(P)["Units"].sum() / 12,
        "24m mean": hist[t > tmax - 24].groupby(P)["Units"].sum() / min(24, hist["FiscalYear"].nunique() * 12),
    }
    items = pd.DataFrame({P: hist[P].unique()})
    out["Croston SBA (monthly)"] = pd.Series(fc.croston_forecast(hist, items, P), index=items[P])
    qh = hist.assign(Q=(hist["FiscalMonth"] - 1) // 3 + 1).groupby([P, "FiscalYear", "Q"], as_index=False)["Units"].sum()
    qh = qh.rename(columns={"Q": "FiscalMonth"})
    out["ADIDA quarterly SBA"] = pd.Series(fc.croston_forecast(qh, items, P), index=items[P]) / 3
    yearly = hist.groupby([P, "FiscalYear"])["Units"].sum().unstack(fill_value=0.0).sort_index(axis=1)
    level = yearly.iloc[:, 0].astype(float)
    for c in yearly.columns[1:]:
        level = level + 0.3 * (yearly[c] - level)
    out["ADIDA annual SES"] = level / 12
    return out


all_rows = []
for vy in (2024, 2023):
    b = baseline(vy)
    rows = b["rows"].copy()
    tys, trailing = b["tys"], b["trailing"]
    sigma0 = fair_sigma(b)
    tier = rows[P].map(tiers(df, tys))
    base_fr = forecast_frontier(rows, "base", sigma0, trailing, price, "Baseline")
    budget = float(base_fr[(base_fr["Family"] == "Baseline") & (base_fr["Setting"].round(2) == SERVICE)]
                   ["Avg inventory value"].iloc[0])
    log(f"FY{vy}: baseline 95% budget {money(budget)}; tiers {tier.drop_duplicates().size}")

    now, prior = rates(vy), rates(vy - 1)
    prior_actual = df[df["FiscalYear"] == vy - 1][[P, "FiscalMonth", "Units"]]
    ml_annual = rows.groupby(P)["base"].transform("sum") / 12
    now["ML annual, flattened"] = None

    def evaluate(name, pred, sigma):
        rows["_p"] = pred
        s = score(rows, "_p", df, tys)
        fr = forecast_frontier(rows, "_p", sigma, trailing, price, "v")
        m95 = fr[(fr["Family"] == "v") & (fr["Setting"].round(2) == SERVICE)].iloc[0]
        s.update({"Variant": name, "FY": vy, "Fill @ baseline budget": fill_at(fr, "v", budget),
                  "95% plan fill": m95["Fill rate"], "95% plan budget": m95["Avg inventory value"]})
        all_rows.append(s)
        log(f"  FY{vy} {name}: WAPE {pct(s['WAPE item-month'])}, fill @ budget {pct(s['Fill @ baseline budget'])}")

    evaluate("Baseline (app, 12 months ahead)", rows["base"].to_numpy(), sigma0)
    for method in now:
        if method == "ML annual, flattened":
            rate_now = ml_annual.to_numpy()
            sig_method = {}
        else:
            rate_now = rows[P].map(now[method]).fillna(0.0).to_numpy()
            pa = prior_actual.assign(_r=prior_actual[P].map(prior[method]).fillna(0.0))
            sig_method = (pa["Units"] - pa["_r"]).groupby(pa[P]).std().to_dict()
        for switched in (("Sparse",), ("Sparse", "Medium")):
            mask = tier.isin(switched).to_numpy()
            pred = np.where(mask, rate_now, rows["base"].to_numpy())
            sigma = dict(sigma0)
            if sig_method:
                for p_ in rows.loc[mask, P].unique():
                    if p_ in sig_method and pd.notna(sig_method[p_]):
                        sigma[p_] = sig_method[p_]
            evaluate(f"{method} on {' + '.join(switched)}", pred, sigma)
        evaluate(f"{method} on ALL items", rate_now,
                 {**sigma0, **{k: v for k, v in sig_method.items() if pd.notna(v)}})

res = pd.DataFrame(all_rows)
res.to_csv(os.path.join(OUT, "02_sparse_methods.csv"), index=False)
cols = ["WAPE item-month", "Bias", "WAPE item-quarter", "WAPE item-year", "WAPE Sparse", "WAPE Medium",
        "WAPE Dense", "Fill @ baseline budget", "95% plan fill", "95% plan budget"]
for vy in (2024, 2023):
    t = res[res["FY"] == vy].set_index("Variant")[cols]
    print(f"\n=== FY{vy} ===")
    print(t.to_string(formatters={c: (money if c.endswith("plan budget") else pct) for c in cols}))
