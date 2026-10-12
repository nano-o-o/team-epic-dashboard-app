"""Idea 1 -- what would outside information (a sales plan, an order backlog) be worth?

MELFT's data has no backlog, quotation or sales-plan field (checked against the
data dictionary; only ~550 open orders exist, all at the very end of the export),
so this cannot be tested with real advance-order data. Instead it measures the
VALUE OF THAT INFORMATION: how much the 12-month-ahead forecast and the stocking
replay improve if the forecast is re-levelled with a plan of a given quality.

  Plan variants (applied to the app's own 12-month-ahead forecast):
    perfect total        scale every item so the catalog total matches the year's actual
    perfect Category1    scale each Category1 to its actual total (an oracle upper bound)
    noisy Category1 ±s   the same, with a plan that is off by N(0, s) per category (30 draws)
    naive plan           category totals = last year's actual (no oracle -- anyone has this)
  Order-visibility variants (a backlog that reveals the first k months):
    visible k months     months 1..k replaced by actual demand; the rest re-levelled per
                         Category1 by actual/forecast over those k months (clipped 0.5x-2x)

Scored at FY2024 and FY2023 on the app's validation rows. Stocking: the app's
forecast-driven policy family ordering to each forecast (sigma unchanged), read at
the baseline 95% plan's own budget.
"""
import os

import numpy as np
import pandas as pd

from improve_common import (prepared, baseline, fair_sigma, score, forecast_frontier, fill_at, price_map,
                            money, pct, log, P, SERVICE, ROOT)
import forecast_core as fc  # after improve_common, which puts the project root on sys.path

OUT = os.path.join(ROOT, "experiments", "_results")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)
prep = prepared()
df, price = prep["df"], price_map(prep)
cat1 = df.drop_duplicates(P).set_index(P)["Category1"].fillna("(unknown)")


def relevel(rows, target_by_cat, src="base"):
    """Scale each Category1's forecast so it sums to target_by_cat[cat]."""
    r = rows.copy()
    c = r[P].map(cat1)
    f_tot = r.groupby(c)[src].transform("sum")
    tgt = c.map(target_by_cat).astype(float)
    factor = np.where(f_tot > 0, tgt / f_tot, 1.0)
    return r[src].to_numpy() * np.nan_to_num(factor, nan=1.0)


all_rows = []
for vy in (2024, 2023):
    b = baseline(vy)
    rows = b["rows"].copy()
    tys, sigma, trailing = b["tys"], fair_sigma(b), b["trailing"]
    actual_cat = rows.groupby(rows[P].map(cat1))["Actual"].sum()
    prev = df[df["FiscalYear"] == vy - 1]
    naive_cat = prev.groupby(prev[P].map(cat1))["Units"].sum()

    base_fr = forecast_frontier(rows, "base", sigma, trailing, price, "Baseline")
    budget = float(base_fr[(base_fr["Family"] == "Baseline") & (base_fr["Setting"].round(2) == SERVICE)]
                   ["Avg inventory value"].iloc[0])
    log(f"FY{vy}: baseline 95% budget {money(budget)}")

    variants = {"Baseline (app, 12 months ahead)": rows["base"].to_numpy()}
    tot = rows["Actual"].sum() / max(rows["base"].sum(), 1e-9)
    variants["Perfect total plan"] = rows["base"].to_numpy() * tot
    variants["Perfect Category1 plan"] = relevel(rows, actual_cat.to_dict())
    variants["Naive plan (Category1 = last year)"] = relevel(rows, naive_cat.to_dict())
    naive_tot = prev["Units"].sum() / max(rows["base"].sum(), 1e-9)
    variants["Naive plan (total = last year)"] = rows["base"].to_numpy() * naive_tot

    for k in (1, 2, 3):
        known = rows["FiscalMonth"] <= k
        c = rows[P].map(cat1)
        a_k = rows[known].groupby(c[known])["Actual"].sum()
        f_k = rows[known].groupby(c[known])["base"].sum()
        ratio = (a_k / f_k.replace(0, np.nan)).clip(0.5, 2.0).fillna(1.0)
        rest = rows["base"].to_numpy() * c.map(ratio).fillna(1.0).to_numpy()
        variants[f"Visible first {k} month(s), rest re-levelled"] = np.where(known, rows["Actual"], rest)
        variants[f"Visible first {k} month(s), rest unchanged"] = np.where(known, rows["Actual"], rows["base"])

    def evaluate(name, pred, draws=1):
        rows["_p"] = pred
        s = score(rows, "_p", df, tys)
        # Months 1-3 are copied from actuals in the visibility variants, so they
        # also get scored on months 4-12 only, where every variant is a forecast.
        late = rows[rows["FiscalMonth"] >= 4]
        s["WAPE item-month, months 4-12"] = fc.wape(late["Actual"], late["_p"])
        s["Bias, months 4-12"] = fc.forecast_bias(late["Actual"], late["_p"])
        s["WAPE Cat1-month, months 4-12"] = fc.aggregated_wape(
            late.assign(_c=late[P].map(cat1)), "_p", ["_c", "FiscalMonth"])
        fr = forecast_frontier(rows, "_p", sigma, trailing, price, "v")
        m95 = fr[(fr["Family"] == "v") & (fr["Setting"].round(2) == SERVICE)].iloc[0]
        s.update({"Variant": name, "FY": vy,
                  "Fill @ baseline budget": fill_at(fr, "v", budget),
                  "95% plan fill": m95["Fill rate"], "95% plan budget": m95["Avg inventory value"]})
        return s

    for name, pred in variants.items():
        all_rows.append(evaluate(name, pred))
        log(f"  FY{vy} {name}: done")

    for s_noise in (0.10, 0.20, 0.30):
        acc = []
        for _ in range(30):
            eps = rng.normal(0, s_noise, len(actual_cat))
            plan = (actual_cat * np.clip(1 + eps, 0.1, None)).to_dict()
            acc.append(evaluate(f"Category1 plan, off by ±{s_noise:.0%} (avg of 30)", relevel(rows, plan)))
        m = pd.DataFrame(acc)
        avg = m.drop(columns=["Variant", "FY"]).mean().to_dict()
        avg.update({"Variant": acc[0]["Variant"], "FY": vy})
        all_rows.append(avg)
        log(f"  FY{vy} noisy plan ±{s_noise:.0%}: done")

res = pd.DataFrame(all_rows)
res.to_csv(os.path.join(OUT, "01_plan_value.csv"), index=False)
cols = ["WAPE item-month", "Bias", "WAPE Cat1-month", "WAPE item-month, months 4-12", "Bias, months 4-12",
        "WAPE Cat1-month, months 4-12", "WAPE item-quarter", "WAPE item-year",
        "WAPE Sparse", "WAPE Dense", "Fill @ baseline budget", "95% plan fill", "95% plan budget"]
for vy in (2024, 2023):
    t = res[res["FY"] == vy].set_index("Variant")[cols]
    print(f"\n=== FY{vy} ===")
    print(t.to_string(formatters={c: (money if c.endswith("plan budget") else pct) for c in cols}))
