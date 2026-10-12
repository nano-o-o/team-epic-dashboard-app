"""Idea 5 -- a different model per dashboard page: rejected.

Each page reports the same forecast at a different grain, so "use the best model
for each page" only means something if the model ranking flips between grains.
It does, twice -- and neither flip survives inspection:

  * The one row that changes hands for a visible reason is Total (volume), which
    is arithmetically |aggregate bias|. It rewards whichever model under-forecasts
    LEAST, not whichever forecasts best, so HistGradientBoosting wins it while
    being ~11 WAPE points worse on the individual items the stocking decisions
    are actually made from.
  * Re-running one year earlier, three of the five per-grain winners change, and
    the FY2023 Total-volume winner is not even the FY2024 one. Picking a winner
    per grain on the validation year is selecting on the test set.

Pass A reproduces the app's own WAPE-by-grain matrix (1 month ahead, exactly what
Model Accuracy shows) for the three default models plus the Baseline and Croston,
at FY2024 and FY2023. Pass B repeats it 12 months ahead at FY2024 -- the horizon
the pages actually consume. Reads only; changes nothing in the app.
"""
import os

import pandas as pd

from improve_common import prepared, train_years_for, log, P, FISCAL_START, ROOT
import forecast_core as fc  # after improve_common, which puts the project root on sys.path

OUT = os.path.join(ROOT, "experiments", "_results")
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 200)

# The app's PREFERRED_DEFAULTS (dashboard_app.py). run_backtest adds Baseline
# and Croston columns of its own, which evaluate_backtest then scores too.
MODELS = ["LightGBM (Tweedie)", "XGBoost (Tweedie)", "HistGradientBoosting"]
SHIPPED = "LightGBM (Tweedie)"

prep = prepared()
df, extra, cats, base = prep["df"], prep["extra"], prep["cats"], prep["base"]
all_rows, winners = [], {}


def _record(by_grain, vy, horizon):
    """Print a grain x model matrix, note its per-grain winner, and tidy it for CSV."""
    print(f"\n===== FY{vy} -- WAPE by grain, {horizon} (lower is better) =====")
    print((by_grain * 100).round(1).to_string())
    print("\nwinner per grain:")
    for g, row in by_grain.iterrows():
        print(f"   {g:22s} {row.idxmin():24s} {row.min():.1%}   "
              f"(shipped {SHIPPED} {row[SHIPPED]:.1%})")
        winners[(vy, horizon, g)] = row.idxmin()
        all_rows.append({"FY": vy, "Horizon": horizon, "Grain": g,
                         "Winner": row.idxmin(), **row.to_dict()})


def _backtest(vy):
    tys = train_years_for(vy)
    log(f"FY{vy}: backtesting {len(MODELS)} models on train {tys} ...")
    res_df, _metrics, *_ = fc.run_backtest(df, P, cats, tys, vy, FISCAL_START, MODELS,
                                           extra_features=extra, feature_base=base)
    return tys, res_df


# --------------------------------------------------------- Pass A: 1 month ahead
for vy in (2024, 2023):
    tys, res_df = _backtest(vy)
    rep = fc.evaluate_backtest(res_df, df, P, tys, category_cols=cats,
                               fiscal_start_month=FISCAL_START)
    _record(rep["by_grain"], vy, "1 month ahead")
    print(f"\n--- FY{vy} headline metrics, SKU-month ---")
    print((rep["summary"][["WAPE", "Bias", "MASE"]] * 100).round(1).to_string())

print("\n\n===== Does the per-grain winner hold at both origin years? =====")
grains = [g for (fy, h, g) in winners if fy == 2024 and h == "1 month ahead"]
for g in grains:
    w24, w23 = winners[(2024, "1 month ahead", g)], winners[(2023, "1 month ahead", g)]
    print(f"   {g:22s} FY2024 -> {w24:24s} FY2023 -> {w23:24s} "
          f"{'same' if w24 == w23 else '*** FLIPS ***'}")

# ------------------------------------------------------ Pass B: 12 months ahead
vy = 2024
tys, res_df = _backtest(vy)
rows = res_df[[P, "FiscalYear", "FiscalMonth", "Actual"]].copy()
for m in MODELS:
    log(f"FY{vy}: 12-month-ahead forecast, {m} ...")
    hz = fc.horizon_backtest(df, P, cats, tys, vy, FISCAL_START, m,
                             extra_features=extra, feature_base=base)
    rows = rows.merge(hz[[P, "FiscalMonth", "Pred"]].rename(columns={"Pred": f"{m}_Pred"}),
                      on=[P, "FiscalMonth"], how="left")
    rows[f"{m}_Pred"] = rows[f"{m}_Pred"].fillna(0.0)

rep12 = fc.evaluate_backtest(rows, df, P, tys, category_cols=cats,
                             fiscal_start_month=FISCAL_START)
_record(rep12["by_grain"], vy, "12 months ahead")
print(f"\n--- FY{vy} headline metrics 12 months ahead, SKU-month ---")
print((rep12["summary"][["WAPE", "Bias", "MASE"]] * 100).round(1).to_string())
print(f"\n--- FY{vy} total forecast units vs actual (the level each model lands on) ---")
for m in MODELS:
    print(f"   {m:24s} {rows[f'{m}_Pred'].sum():>12,.0f}")
print(f"   {'ACTUAL':24s} {rows['Actual'].sum():>12,.0f}")

pd.DataFrame(all_rows).to_csv(os.path.join(OUT, "05_per_grain_models.csv"), index=False)
print(f"\nwrote {os.path.join(OUT, '05_per_grain_models.csv')}")
