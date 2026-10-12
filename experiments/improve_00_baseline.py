"""Step 0: rebuild the app's defaults baseline and check it reproduces the
recorded numbers before any experiment is compared against it.

Expected (defaults build, FY2024): item-month WAPE 67.5% (1 month ahead) and
66.6% (12 months ahead), bias -31.8% (12 months), fill rate 81.1% for the 95%
plan at ~฿1.05bn, flat rule 79.3% at that budget, today's 1.5-month rule 70.9%
on ~฿0.54bn. FY2023 12-month WAPE 118.7%."""
from improve_common import (prepared, baseline, score, forecast_frontier, fill_at, price_map,
                            money, pct, log, SERVICE, LEGACY)

prep = prepared()
df, price = prep["df"], price_map(prep)
log(f"prepared: {df['Item Internal ID'].nunique():,} items, {len(df):,} item-months")

for vy in (2024, 2023):
    b = baseline(vy)
    rows = b["rows"]
    s1, s12 = score(rows, "pred_1m", df, b["tys"]), score(rows, "base", df, b["tys"])
    print(f"\n=== FY{vy} (train {b['tys']}, sigma from {b['sigma_basis']}) — {len(rows):,} item-months ===")
    for k in s12:
        v1, v12 = s1[k], s12[k]
        fmt = (lambda v: f"{v:,.0f}") if k == "Total units" else pct
        print(f"  {k:<20} 1m {fmt(v1):>10}   12m {fmt(v12):>10}")
    fr = forecast_frontier(rows, "base", b["sigma"], b["trailing"], price, "Baseline")
    mine = fr[(fr["Family"] == "Baseline") & (fr["Setting"].round(2) == SERVICE)].iloc[0]
    cur = fr[(fr["Family"] == "Baseline | flat rule") & (fr["Setting"] == LEGACY)].iloc[0]
    flat_same = fill_at(fr, "Baseline | flat rule", mine["Avg inventory value"])
    print(f"  95% plan: fill {pct(mine['Fill rate'])} on {money(mine['Avg inventory value'])}; "
          f"flat rule at that budget {pct(flat_same)}; 1.5-month rule {pct(cur['Fill rate'])} on "
          f"{money(cur['Avg inventory value'])}; items scored {int(mine['Items scored']):,}")
