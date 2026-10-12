"""Idea 6 -- Poisson (count-based) safety stock for slow movers.

The app sets stock with a normal ("bell curve") approximation:
    S = forecast x P + z * sigma * sqrt(P)          P = lead + review = 5 months
For an item that sells 0-2 units a month the bell curve is a poor fit: demand is
a small whole number and can't go below zero. The textbook fix for spare parts is
to treat demand over P as a count distribution and stock its service-level
percentile. Families compared, all replayed on the held-out year with the app's
exact simulation, at the SAME average stock value:

  Baseline            the app (normal, sigma from each item's own forecast error)
  Poisson             S = Poisson percentile, mean = forecast x P. Assumes
                      variance = mean (orders arrive independently, one unit each)
  Hybrid              Poisson for slow movers (forecast < 1 unit/month), the app's
                      normal rule for everything else
  Negative binomial   count distribution with the same mean but the item's own
                      variance (sigma^2 x P) -- Poisson's fix when B2B orders come
                      in lumps, so variance > mean ("overdispersion")

Run from the project root:
    .venv/Scripts/python.exe experiments/improve_06_poisson_safety_stock.py
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from improve_common import (prepared, baseline, fair_sigma, forecast_frontier, replay_policies, fill_at, price_map,
                            money, pct, log, P, LEAD, REVIEW, SERVICE, LEGACY, ROOT)
import inventory as inv

OUT = os.path.join(ROOT, "experiments", "_results")
os.makedirs(OUT, exist_ok=True)
PROT = int(LEAD + REVIEW)
SLOW = 1.0  # units / month: below this an item counts as a slow mover
LEVELS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995)
prep = prepared()
df, price = prep["df"], price_map(prep)

summary = []
for vy in (2024, 2023):
    b = baseline(vy)
    rows, trailing = b["rows"], b["trailing"]
    sig = fair_sigma(b)  # no leak at FY2023 (see improve_common.fair_sigma)

    base_fr = forecast_frontier(rows, "base", sig, trailing, price, "Baseline (normal)", service_levels=LEVELS)
    b95 = base_fr[(base_fr["Family"] == "Baseline (normal)") & (base_fr["Setting"].round(3) == SERVICE)].iloc[0]
    legacy = base_fr[(base_fr["Family"] == "Baseline (normal) | flat rule") & (base_fr["Setting"] == LEGACY)].iloc[0]

    monthly = rows.groupby(P)["base"].sum() / 12
    products, m = monthly.index, monthly.to_numpy()
    dormant = inv._dormant_mask(products, m, trailing)
    mu = m * PROT
    sd_month = inv._resolve_sigma(products, m, sig, 0.7)
    var = np.maximum((sd_month ** 2) * PROT, mu * 1.0001)  # NB needs variance > mean
    slow = m < SLOW

    # How far from Poisson is the demand? Share of items whose monthly variance
    # over the 24 months before the year exceeds the mean (Poisson says equal).
    pre = df[df["FiscalYear"] < vy]
    t = pre["FiscalYear"] * 12 + pre["FiscalMonth"]
    st = pre[t > t.max() - 24].groupby(P)["Units"].agg(["mean", "var"])
    st = st[st["mean"] > 0]
    ratio = (st["var"] / st["mean"]).median()
    over = float((st["var"] > 1.5 * st["mean"]).mean())
    log(f"FY{vy}: {slow.sum():,} of {len(m):,} items are slow movers; median variance/mean = {ratio:.1f} "
        f"(Poisson = 1); {over:.0%} of selling items have variance > 1.5x mean")

    pol = {}
    for sl in LEVELS:
        s_normal = np.round(mu + inv._z(sl) * sd_month * np.sqrt(PROT))
        s_pois = poisson.ppf(sl, np.maximum(mu, 1e-12))
        s_nb = nbinom.ppf(sl, mu ** 2 / np.maximum(var - mu, 1e-12), np.clip(mu / var, 1e-12, 1 - 1e-12))
        for fam, s in (("Poisson", s_pois), ("Hybrid (Poisson for slow movers)", np.where(slow, s_pois, s_normal)),
                       ("Negative binomial", s_nb)):
            s = np.where(dormant | (mu <= 0), 0.0, np.nan_to_num(s))
            pol[f"{fam} {sl}"] = (fam, sl, dict(zip(products, s)))
    fr = pd.concat([base_fr, replay_policies(rows, pol, price)], ignore_index=True)
    fr.to_csv(os.path.join(OUT, f"06_frontiers_FY{vy}.csv"), index=False)

    budgets = {"today's 1.5-month rule": float(legacy["Avg inventory value"]),
               "app's 95% plan": float(b95["Avg inventory value"])}
    fams = ["Baseline (normal)", "Poisson", "Hybrid (Poisson for slow movers)", "Negative binomial"]
    table = pd.DataFrame({f: {f"{k} ({money(v)})": fill_at(fr, f, v) for k, v in budgets.items()} for f in fams}).T
    for f in fams:
        g = fr[fr["Family"] == f]
        table.loc[f, "budget range swept"] = f"{money(g['Avg inventory value'].min())}-{money(g['Avg inventory value'].max())}"
        at95 = g[g["Setting"].round(3) == SERVICE].iloc[0]
        table.loc[f, "at 95% setting"] = f"{pct(at95['Fill rate'])} on {money(at95['Avg inventory value'])}"
    print(f"\n=== FY{vy}: fill rate at the same stock budget ===")
    print(table.to_string(formatters={c: pct for c in table.columns if "(" in c}))
    summary += [{"FY": vy, "Family": f, "Budget": c, "Fill": table.loc[f, c]} for f in fams
                for c in table.columns if "(" in c]
    summary += [{"FY": vy, "Family": "diagnostic", "Budget": "median variance/mean", "Fill": ratio},
                {"FY": vy, "Family": "diagnostic", "Budget": "share variance > 1.5x mean", "Fill": over}]

pd.DataFrame(summary).to_csv(os.path.join(OUT, "06_poisson_summary.csv"), index=False)
