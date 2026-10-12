"""Idea 7: does tuning the shipped LightGBM (Tweedie) help, and is forecast_future's
"every item sold last month" quirk worth fixing?

(a) 13 hyperparameter settings, 1 month ahead, at the FY2024 and FY2023 origins.
(b) The settings that moved WAPE, 12 months ahead at FY2024, with fill rate at the
    default 95% plan's budget (sigma from each variant's own FY2023 12-month error,
    as the app does).
(c) forecast_future marks an item "sold" when its forecast is > 0, and Tweedie
    forecasts are never 0; score thresholds of 0.5 and 1.0 units instead.
Writes _results/07_tuning.csv; run log in _results/07_run.log."""
import inspect
import os

import pandas as pd
from lightgbm import LGBMRegressor

from improve_common import (prepared, baseline, score, forecast_frontier, fill_at, price_map,
                            log, pct, money, P, MODEL, PRED, SERVICE, ROOT)
import forecast_core as fc  # noqa: E402  (improve_common puts the project root on sys.path)
import inventory as inv  # noqa: E402

OUT = os.path.join(ROOT, "experiments", "_results")
prep = prepared()
df, cats, extra, base, price = prep["df"], prep["cats"], prep["extra"], prep["base"], price_map(prep)
DEFAULT = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=5,
               tweedie_variance_power=1.3)
GRID = [{}, {"tweedie_variance_power": 1.1}, {"tweedie_variance_power": 1.5},
        {"tweedie_variance_power": 1.7}, {"min_child_samples": 20}, {"min_child_samples": 50},
        {"num_leaves": 15}, {"num_leaves": 63}, {"n_estimators": 150}, {"n_estimators": 600},
        {"learning_rate": 0.02, "n_estimators": 750},
        {"subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8}, {"reg_lambda": 5.0}]
FILL_CHECK = [{"tweedie_variance_power": 1.5}, {"tweedie_variance_power": 1.7},
              {"subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8}]
_make_model, _forecast_future = fc.make_model, fc.forecast_future


def use(change):
    p = {**DEFAULT, **change}
    fc.make_model = lambda mt: (LGBMRegressor(random_state=42, verbosity=-1, n_jobs=-1, objective="tweedie", **p)
                                if mt == MODEL else _make_model(mt))


def horizon(vy):
    tys = [y for y in (2021, 2022, 2023) if y < vy]
    return fc.horizon_backtest(df, P, cats, tys, vy, 4, MODEL, extra_features=extra, feature_base=base)


def name(change):
    return ", ".join(f"{k}={v}" for k, v in change.items()) or "default"


records = []
# (a) 1 month ahead, both origins
for change in GRID:
    use(change)
    rec = {"Part": "a: 1 month ahead", "Setting": name(change)}
    for vy in (2024, 2023):
        tys = [y for y in (2021, 2022, 2023) if y < vy]
        res, *_ = fc.run_backtest(df, P, cats, tys, vy, 4, [MODEL], extra_features=extra, feature_base=base)
        rec[f"WAPE FY{vy}"] = fc.wape(res["Actual"], res[PRED])
        rec[f"Bias FY{vy}"] = fc.forecast_bias(res["Actual"], res[PRED])
    log(f"(a) {rec['Setting']}: FY2024 {pct(rec['WAPE FY2024'])} bias {pct(rec['Bias FY2024'])} | "
        f"FY2023 {pct(rec['WAPE FY2023'])} bias {pct(rec['Bias FY2023'])}")
    records.append(rec)

# (b) 12 months ahead at FY2024 + fill rate at the default plan's budget
b = baseline(2024)
rows0 = b["rows"][[P, "FiscalYear", "FiscalMonth", "Actual"]]
fr0 = forecast_frontier(b["rows"], "base", b["sigma"], b["trailing"], price, "M")
m0 = fr0[(fr0["Family"] == "M") & (fr0["Setting"].round(2) == SERVICE)].iloc[0]
budget = m0["Avg inventory value"]
log(f"(b) default: 95% plan {pct(m0['Fill rate'])} on {money(budget)}")
for change in FILL_CHECK:
    use(change)
    sigma = inv.forecast_error_sigma(horizon(2023).rename(columns={"Pred": PRED}), MODEL, P).to_dict()
    rows = rows0.merge(horizon(2024)[[P, "FiscalMonth", "Pred"]], on=[P, "FiscalMonth"], how="left").fillna({"Pred": 0})
    s = score(rows, "Pred", df, [2021, 2022, 2023])
    fr = forecast_frontier(rows, "Pred", sigma, b["trailing"], price, "M")
    m = fr[(fr["Family"] == "M") & (fr["Setting"].round(2) == SERVICE)].iloc[0]
    rec = {"Part": "b: 12 months ahead FY2024", "Setting": name(change), "WAPE FY2024": s["WAPE item-month"],
           "Bias FY2024": s["Bias"], "Plan fill": m["Fill rate"], "Plan budget": m["Avg inventory value"],
           "Fill at default budget": fill_at(fr, "M", budget)}
    log(f"(b) {rec['Setting']}: WAPE {pct(rec['WAPE FY2024'])} bias {pct(rec['Bias FY2024'])} | 95% plan "
        f"{pct(rec['Plan fill'])} on {money(rec['Plan budget'])} | at {money(budget)}: {pct(rec['Fill at default budget'])}")
    records.append(rec)
fc.make_model = _make_model

# (c) "sold this month" threshold in forecast_future
src = inspect.getsource(_forecast_future)
for thr in (None, 0.5, 1.0):
    if thr is None:
        fc.forecast_future = _forecast_future
    else:
        patched = src.replace("sold_this_month = final_preds > 0", f"sold_this_month = final_preds >= {thr}")
        assert patched != src, "forecast_future changed; update the patch"
        ns = {}
        exec(patched, fc.__dict__, ns)
        fc.forecast_future = ns["forecast_future"]
    rec = {"Part": "c: sold-month threshold, 12 months ahead", "Setting": "> 0 (shipped)" if thr is None else f">= {thr}"}
    for vy in (2024, 2023):
        tys = [y for y in (2021, 2022, 2023) if y < vy]
        rows = baseline(vy)["rows"][[P, "FiscalMonth", "Actual"]].merge(
            horizon(vy)[[P, "FiscalMonth", "Pred"]], on=[P, "FiscalMonth"], how="left").fillna({"Pred": 0})
        s = score(rows, "Pred", df, tys)
        rec[f"WAPE FY{vy}"], rec[f"Bias FY{vy}"] = s["WAPE item-month"], s["Bias"]
    log(f"(c) {rec['Setting']}: FY2024 {pct(rec['WAPE FY2024'])} | FY2023 {pct(rec['WAPE FY2023'])}")
    records.append(rec)
fc.forecast_future = _forecast_future

pd.DataFrame(records).to_csv(os.path.join(OUT, "07_tuning.csv"), index=False)
