"""Idea 4 -- set stock for the business outcome (fill rate per baht), not for WAPE.

Every policy family below is replayed on the held-out year with the app's exact
simulation (quarterly review, 2-month lead, lost sales, scored on items with
demand), using only what was known at the start of that year. Families are
compared at the SAME average stock value, read off each family's own frontier.

  Baseline            the app: forecast x P + z * sigma_month * sqrt(P)    (z swept)
  Baseline flat rule  forecast x P + N months of forecast                  (N swept)
  Quantile            P x (p50 + k * (p90 - p50)) from the 12-month-ahead quantile models (k swept)
  Sigma over P        forecast x P + z * sigma_P, sigma_P measured directly on 5-month
                      windows of the prior year's errors (no sqrt(P) independence assumption)
  Empirical           S = q-th percentile of the item's own 5-month demand over the last
                      24 months -- no model at all                          (q swept)
  Price-aware         per item, the S that maximises expected units served minus
                      lambda * list price * S, on the empirical 5-month demand
                      distribution (lambda swept) -- puts buffer where it buys the most
                      fill rate per baht
  Price-aware + trend the same, with each item's empirical demand scaled by its
                      forecast vs last year's sales (clipped 0.5x-2x)
"""
import os

import numpy as np
import pandas as pd

from improve_common import (prepared, baseline, fair_sigma, forecast_frontier, replay_policies, fill_at, price_map,
                            money, pct, log, P, LEAD, REVIEW, SERVICE, LEGACY, ROOT)
import inventory as inv

OUT = os.path.join(ROOT, "experiments", "_results")
os.makedirs(OUT, exist_ok=True)
PROT = int(LEAD + REVIEW)
prep = prepared()
df, price = prep["df"], price_map(prep)
Z_LEVELS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995)


def window_demand(vy, months=24):
    """items x windows matrix of PROT-month demand sums over the `months` before vy."""
    pre = df[df["FiscalYear"] < vy].copy()
    pre["t"] = pre["FiscalYear"] * 12 + pre["FiscalMonth"]
    pre = pre[pre["t"] > pre["t"].max() - months]
    piv = pre.pivot_table(index=P, columns="t", values="Units", aggfunc="sum").fillna(0.0)
    a = piv.to_numpy(dtype=float)
    c = np.cumsum(np.pad(a, ((0, 0), (1, 0))), axis=1)
    return piv.index, c[:, PROT:] - c[:, :-PROT]


summary_rows = []
for vy in (2024, 2023):
    b = baseline(vy)
    rows, sigma, trailing = b["rows"], b["sigma"], b["trailing"]
    fams = []

    base_fr = forecast_frontier(rows, "base", sigma, trailing, price, "Baseline", service_levels=Z_LEVELS)
    fams.append(base_fr)
    b95 = base_fr[(base_fr["Family"] == "Baseline") & (base_fr["Setting"].round(3) == SERVICE)].iloc[0]
    legacy = base_fr[(base_fr["Family"] == "Baseline | flat rule") & (base_fr["Setting"] == LEGACY)].iloc[0]
    log(f"FY{vy}: baseline 95% {pct(b95['Fill rate'])} on {money(b95['Avg inventory value'])}")

    monthly = rows.groupby(P)["base"].sum() / 12
    products = monthly.index
    dormant = inv._dormant_mask(products, monthly.to_numpy(), trailing)

    # --- Baseline with an honest sigma at BOTH origins -------------------------
    # FY2023 has no earlier year to measure forecast error on, so the app's
    # fallback measures sigma on the replayed year itself -- a leak that inflates
    # its fill rate (99% on FY2023). This variant uses each item's own monthly
    # demand std over the 24 months before the year instead: known in advance,
    # and identical in construction at FY2023 and FY2024.
    pre = df[df["FiscalYear"] < vy].copy()
    pre_t = pre["FiscalYear"] * 12 + pre["FiscalMonth"]
    demand_sd = pre[pre_t > pre_t.max() - 24].groupby(P)["Units"].std().to_dict()
    fams.append(forecast_frontier(rows, "base", demand_sd, trailing, price,
                                  "Baseline, sigma = pre-year demand std", service_levels=Z_LEVELS))

    # --- Quantile family ----------------------------------------------------
    q = rows.groupby(P)[["q50", "q90"]].mean().reindex(products).fillna(0.0)
    pol = {}
    for k in (0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6):
        s = np.where(dormant, 0.0, np.round(PROT * (q["q50"] + k * (q["q90"] - q["q50"])).clip(lower=0)))
        pol[f"Quantile k={k}"] = ("Quantile (p50 + k(p90-p50))", float(k), dict(zip(products, s)))
    fams.append(replay_policies(rows, pol, price))

    # --- Sigma measured over the protection period ---------------------------
    pol = {}
    if b["hp"] is not None:
        hp = b["hp"].sort_values([P, "FiscalMonth"])
        e = (hp["Actual"] - hp["Pred"]).to_numpy()
        piv = pd.DataFrame({P: hp[P].to_numpy(), "m": hp["FiscalMonth"].to_numpy(), "e": e}) \
            .pivot_table(index=P, columns="m", values="e", aggfunc="sum").fillna(0.0)
        a = piv.to_numpy()
        c = np.cumsum(np.pad(a, ((0, 0), (1, 0))), axis=1)
        sig_p = pd.Series((c[:, PROT:] - c[:, :-PROT]).std(axis=1), index=piv.index)
        sig_month = inv._resolve_sigma(products, monthly.to_numpy(), sigma, 0.7)
        sp = sig_p.reindex(products).to_numpy()
        sp = np.where(np.isnan(sp) | (sp <= 0), sig_month * np.sqrt(PROT), sp)
        for sl in Z_LEVELS:
            s = np.where(dormant, 0.0, np.round(monthly.to_numpy() * PROT + inv._z(sl) * sp))
            pol[f"SigmaP {sl}"] = ("Sigma over P (measured)", sl, dict(zip(products, s)))
        fams.append(replay_policies(rows, pol, price))

    # --- Empirical, model-free -------------------------------------------------
    idx, W = window_demand(vy)
    pol = {}
    for ql in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0):
        s = np.round(np.quantile(W, ql, axis=1))
        pol[f"Empirical q={ql}"] = ("Empirical 5-month demand percentile", ql, dict(zip(idx, s)))
    fams.append(replay_policies(rows, pol, price))

    # --- Price-aware (marginal fill per baht) -------------------------------------
    pr = pd.Series(price).reindex(idx).fillna(0.0).to_numpy()
    fc_year = rows.groupby(P)["base"].sum().reindex(idx).fillna(0.0).to_numpy()
    tr_year = pd.Series(trailing).reindex(idx).fillna(0.0).to_numpy()
    trend = np.clip(np.where(tr_year > 0, fc_year / np.maximum(tr_year, 1e-9), 1.0), 0.5, 2.0)

    def price_aware(Wm, label):
        cand = [np.unique(np.concatenate([[0.0], np.round(w)])) for w in Wm]
        gain = [np.array([np.minimum(s, w).mean() for s in cs]) for cs, w in zip(cand, Wm)]
        pol = {}
        for lam in np.geomspace(1e-7, 1e-2, 16):
            s_best = np.array([cs[np.argmax(g - lam * max(p, 1.0) * cs)] for cs, g, p in zip(cand, gain, pr)])
            pol[f"{label} lam={lam:.1e}"] = (label, float(lam), dict(zip(idx, s_best)))
        return replay_policies(rows, pol, price), pol

    pols = {}
    for W_, label in ((W, "Price-aware (empirical)"), (W * trend[:, None], "Price-aware + forecast trend")):
        f_, pols[label] = price_aware(W_, label)
        fams.append(f_)
    # Sensitivity: does the result hinge on the 24-month history window?
    for months in (12, 36):
        idx_m, W_m = window_demand(vy, months=months)
        if W_m.shape[1] < 2:
            continue
        idx, W = idx_m, W_m  # price_aware reads idx / pr from the enclosing scope
        pr = pd.Series(price).reindex(idx).fillna(0.0).to_numpy()
        label = f"Price-aware (empirical, {months}-month history)"
        f_, pols[label] = price_aware(W, label)
        fams.append(f_)

    # --- Price-aware on the ML forecast ---------------------------------------------
    # Same marginal fill-per-baht rule, but each item's 5-month demand is taken as
    # Normal(forecast x P, sigma x sqrt(P)) -- the app's own demand model -- instead
    # of the empirical history. Tests whether the forecast adds anything once stock
    # is allocated by value. Expected units served from normal loss: mu - sd * L(z).
    from scipy.stats import norm
    sig_fair = fair_sigma(b)
    mu = monthly.to_numpy() * PROT
    sd = inv._resolve_sigma(products, monthly.to_numpy(), sig_fair, 0.7) * np.sqrt(PROT)
    prn = pd.Series(price).reindex(products).fillna(0.0).clip(lower=1.0).to_numpy()
    S = np.maximum(0, np.round(mu[:, None] + np.linspace(-2, 4, 31)[None, :] * sd[:, None]))
    S = np.concatenate([np.zeros((len(mu), 1)), S], axis=1)
    zz = (S - mu[:, None]) / np.maximum(sd[:, None], 1e-9)
    served = np.minimum(S, np.clip(mu[:, None] - sd[:, None] * (norm.pdf(zz) - zz * (1 - norm.cdf(zz))), 0, None))
    pol = {}
    for lam in np.geomspace(1e-7, 1e-2, 16):
        pick = np.argmax(served - lam * prn[:, None] * S, axis=1)
        s_best = np.where(dormant, 0.0, S[np.arange(len(mu)), pick])
        pol[f"PA-normal lam={lam:.1e}"] = ("Price-aware + ML forecast (normal)", float(lam), dict(zip(products, s_best)))
    fams.append(replay_policies(rows, pol, price))
    pols["Price-aware + ML forecast (normal)"] = pol
    log(f"FY{vy}: all families replayed")

    fr = pd.concat(fams, ignore_index=True)

    # --- Diagnostics at a matched budget: WHO gets the stock? ---------------------
    # A per-baht rule can win on fill rate by starving expensive items. Compare the
    # honest baseline's 95% policy with the price-aware policy nearest its budget:
    # stockouts, items left with no stock, and fill rate by price tercile.
    fair_fr = fr[fr["Family"] == "Baseline, sigma = pre-year demand std"] if vy == 2023 else fr[fr["Family"] == "Baseline"]
    ref = fair_fr[fair_fr["Setting"].round(3) == SERVICE].iloc[0]
    sig_ref = demand_sd if vy == 2023 else sigma
    sd_ref = inv._resolve_sigma(products, monthly.to_numpy(), sig_ref, 0.7)
    s_ref = np.where(dormant, 0.0, np.round(monthly.to_numpy() * PROT + inv._z(SERVICE) * sd_ref * np.sqrt(PROT)))
    compare = {f"Baseline 95% ({'demand-std sigma' if vy == 2023 else 'app sigma'})": dict(zip(products, s_ref))}
    for fam in ("Price-aware (empirical, 36-month history)", "Price-aware (empirical)", "Price-aware + ML forecast (normal)"):
        f = fr[fr["Family"] == fam]
        if not len(f):
            continue
        near = f.iloc[(f["Avg inventory value"] - ref["Avg inventory value"]).abs().argmin()]
        compare[f"{fam} @ {money(near['Avg inventory value'])}"] = pols[fam][near["Policy"]][2]
    _, per = inv.service_level_comparison(rows, P, compare, lead_time_months=LEAD,
                                          review_period_months=REVIEW, list_price=price)
    diag = []
    for label, r in per.items():
        r = r[r["demand"] > 0].copy()
        r["price"] = r[P].map(price).fillna(0.0)
        r["tier"] = pd.qcut(r["price"].rank(method="first"), 3, labels=["cheapest third", "middle third", "priciest third"])
        row = {"Policy": label, "Fill rate": r["served"].sum() / r["demand"].sum(),
               "Avg stock value": money(float((r["avg_on_hand"] * r["price"]).sum())),
               "Items stocked out": int((r["unmet"] > 1e-9).sum()),
               "Items with zero stock": int((r["order_up_to"] <= 0).sum()), "Items scored": len(r)}
        for t, g in r.groupby("tier", observed=True):
            row[f"Fill, {t}"] = g["served"].sum() / g["demand"].sum()
        diag.append(row)
    diag = pd.DataFrame(diag)
    diag.to_csv(os.path.join(OUT, f"04_diagnostics_FY{vy}.csv"), index=False)
    print(f"\n=== FY{vy}: who gets the stock (matched budget) ===")
    print(diag.to_string(index=False, formatters={c: pct for c in diag.columns if c.startswith("Fill")}))
    fr.to_csv(os.path.join(OUT, f"04_frontiers_FY{vy}.csv"), index=False)
    budgets = {"today's 1.5-month rule budget": float(legacy["Avg inventory value"]),
               "baseline 95% plan budget": float(b95["Avg inventory value"])}
    vals = fr["Avg inventory value"]
    budgets["0.75x plan budget"] = 0.75 * budgets["baseline 95% plan budget"]
    budgets["1.25x plan budget"] = 1.25 * budgets["baseline 95% plan budget"]
    print(f"\n=== FY{vy}: fill rate at the same stock budget ===")
    table = {}
    for fam in fr["Family"].unique():
        f = fr[fr["Family"] == fam]
        table[fam] = {f"{k} ({money(v)})": fill_at(fr, fam, v) for k, v in budgets.items()}
        table[fam]["budget range swept"] = f"{money(f['Avg inventory value'].min())}–{money(f['Avg inventory value'].max())}"
        table[fam]["best fill"] = f["Fill rate"].max()
    t = pd.DataFrame(table).T
    for c in t.columns:
        if c != "budget range swept":
            summary_rows += [{"FY": vy, "Family": fam, "Budget": c, "Fill": t.loc[fam, c]} for fam in t.index]
    print(t.to_string(formatters={c: pct for c in t.columns if c != "budget range swept"}))

pd.DataFrame(summary_rows).to_csv(os.path.join(OUT, "04_stocking_summary.csv"), index=False)
