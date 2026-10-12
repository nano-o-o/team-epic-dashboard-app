"""
=============================================================================
 classical.py -- classical time-series cross-check at the AGGREGATE level
=============================================================================
SARIMA / ETS assume smooth, regularly-spaced, seasonal demand with several
seasonal cycles of history. That describes the *aggregate* series (total
units, total revenue, category totals) -- but NOT the per-SKU series, which
is ~78% zeros. So we use classical methods where they're valid: as an
independent sanity check on the top-line number, next to the ML model's
forecast summed to the same level.

Requires statsmodels.
=============================================================================
"""
import warnings

import numpy as np
import pandas as pd

from forecast_core import fiscal_to_calendar_month, forecast_bias, wape


def _calendar_index(fy, fm, fiscal_start_month):
    cal_m = fiscal_to_calendar_month(int(fm), fiscal_start_month)
    cal_y = int(fy) + (1 if cal_m < fiscal_start_month else 0)
    return pd.Period(freq="M", year=cal_y, month=cal_m)


def monthly_aggregate_series(df, value_col, fiscal_start_month, level_value=None, level_col=None):
    """Total `value_col` per calendar month as a Period-indexed Series."""
    d = df
    if level_col and level_value is not None:
        d = d[d[level_col] == level_value]
    g = d.groupby(["FiscalYear", "FiscalMonth"], as_index=False)[value_col].sum()
    g["period"] = [_calendar_index(r.FiscalYear, r.FiscalMonth, fiscal_start_month) for r in g.itertuples()]
    g = g.sort_values("period")
    s = pd.Series(g[value_col].to_numpy(dtype=float), index=pd.PeriodIndex(g["period"], freq="M"))
    return s.asfreq("M").fillna(0.0)


def _fit_forecast(y_train, horizon, seasonal_periods=12):
    """Returns {method_name: np.array(horizon)}. Falls back gracefully when a
    model can't be estimated from the short history available."""
    out = {}
    y = pd.Series(np.asarray(y_train, dtype=float))
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        # Simple seasonal exponential smoothing (no explicit trend term):
        # with only 3 training years an additive/multiplicative trend
        # over-extrapolates badly on this series.
        m = ExponentialSmoothing(y, trend=None, seasonal="add",
                                 seasonal_periods=seasonal_periods,
                                 initialization_method="estimated").fit()
        out["ETS (seasonal)"] = np.maximum(0.0, np.asarray(m.forecast(horizon), dtype=float))
    except ImportError:  # statsmodels is optional
        pass
    except Exception as e:  # a failed fit drops that method's row; say so rather than hide it
        warnings.warn(f"ETS fit failed ({len(y)} months): {e}")
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        mod = SARIMAX(y, order=(1, 1, 1), seasonal_order=(0, 1, 1, seasonal_periods),
                      enforce_stationarity=False, enforce_invertibility=False)
        m = mod.fit(disp=False)
        # L-BFGS can stop at a worse local optimum on this short series: removing
        # 63 units from the catalog total moved the FY2025 forecast 188,781 ->
        # 198,367 at a log-likelihood of -210.8 vs -204.6. A Nelder-Mead polish
        # from its result finds the better fit; keep whichever scores higher.
        # Polishing one start isn't always enough: with discontinued items back in
        # the history, the FY2021-23 catalog fit stopped at -82.73 (FY2024 forecast
        # 217,217, WAPE 30.9%) where other starts reach -82.10 (150,782, 22.3%).
        # Four fixed AR/MA sign starts found the better optimum on every series tried.
        sigma2 = np.asarray(m.params)[-1]
        starts = [m.params] + [np.r_[ar, ma, -0.5, sigma2] for ar in (-0.5, 0.5) for ma in (-0.5, 0.5)]
        for sp in starts:
            try:
                m2 = mod.fit(start_params=sp, method="nm", maxiter=5000, disp=False)
            except Exception:  # an extra start that fails just isn't a better fit
                continue
            if m2.llf > m.llf:
                m = m2
        out["SARIMA (1,1,1)(0,1,1,12)"] = np.maximum(0.0, np.asarray(m.forecast(horizon), dtype=float))
    except ImportError:
        pass
    except Exception as e:
        warnings.warn(f"SARIMA fit failed ({len(y)} months): {e}")
    if len(y) >= seasonal_periods:  # seasonal-naive reference
        last_season = y.to_numpy()[-seasonal_periods:]
        reps = int(np.ceil(horizon / seasonal_periods))
        out["Seasonal naive"] = np.tile(last_season, reps)[:horizon]
    return out


def aggregate_crosscheck(df, ml_forecast_df, fiscal_start_month, value_col="Units",
                         ml_value_col="Final_Forecast", ml_backtest_monthly=None):
    """Compares the ML forecast against ETS and SARIMA at the aggregate
    (total) level. Returns (backtest_df, forward_df):

      backtest_df -- accuracy on the last 12 months of `df` (the held-out
                     validation year), aggregate monthly grain:
                     [Method, WAPE, MAE, TotalForecast, TotalActual]
      forward_df  -- each method's forecast for the next 12 months:
                     [Method, ForecastTotal]

    ml_backtest_monthly: optional Series/array of the ML model's predicted
    monthly totals for the validation year (from the SKU-level backtest), so
    the ML row appears in backtest_df too.
    """
    full = monthly_aggregate_series(df, value_col, fiscal_start_month)
    y_train, y_test = full.iloc[:-12], full.iloc[-12:]  # validation year is the last 12 months

    fits = _fit_forecast(y_train, horizon=12)
    if ml_backtest_monthly is not None and len(ml_backtest_monthly) == len(y_test):
        fits["ML model (this project)"] = np.asarray(ml_backtest_monthly, dtype=float)
    actual = y_test.to_numpy()
    back_rows = [{"Method": name, "WAPE": wape(actual, fc), "MAE": float(np.mean(np.abs(actual - fc))),
                  "TotalForecast": float(fc.sum()), "TotalActual": float(actual.sum())}
                 for name, fc in fits.items()]
    backtest_df = pd.DataFrame(back_rows).sort_values("WAPE").reset_index(drop=True)

    # --- forward: refit on ALL history, forecast the next 12 months ---
    fwd_fits = _fit_forecast(full, horizon=12)
    fwd_rows = [{"Method": n, "ForecastTotal": float(np.sum(v))} for n, v in fwd_fits.items()]
    if ml_forecast_df is not None and ml_value_col in ml_forecast_df.columns:
        fwd_rows.append({"Method": "ML model (this project)",
                         "ForecastTotal": float(ml_forecast_df[ml_value_col].sum())})
    forward_df = pd.DataFrame(fwd_rows)
    return backtest_df, forward_df


def per_level_crosscheck(df, level_col, train_years, validation_year, fiscal_start_month,
                         value_col="Units"):
    """Fits ETS / SARIMA / seasonal-naive to EACH level's own monthly series
    and scores them together at level x month grain -- the same idea as
    aggregate_crosscheck, one level of the hierarchy down.

    This is where classical methods are most defensible outside the grand
    total: a Category1 series is dense (no zero months) and has 3-4 seasonal
    cycles. It is also where they start to strain -- 24-36 observations is
    thin for a seasonal SARIMA, and a level whose fit diverges shows up here
    as an enormous WAPE rather than being quietly hidden. That is reported
    as-is, because "the classical model failed to estimate on this much
    history" is the finding.

    `df` may be either an item-level or an already-aggregated frame; it is
    grouped by level_col either way. Only fiscal years in
    train_years + [validation_year] are used, so this works for an earlier
    origin year as well as the most recent one.

    Returns (table, detail):
      table  -- [Method, Approach, WAPE, Bias, Forecast, Actual]
      detail -- [Level, FiscalMonth, Method, Actual, Pred]
    """
    years =[int(y) for y in train_years] + [int(validation_year)]
    d = df[df["FiscalYear"].isin(years)].copy()
    d[level_col] = d[level_col].fillna("(unknown)").astype(str).replace("nan", "(unknown)")

    frames = []
    for lv, g in d.groupby(level_col, observed=True):
        s = monthly_aggregate_series(g, value_col, fiscal_start_month)
        if len(s) < 24:  # need at least one full seasonal cycle to train on
            continue
        y_train, y_test = s.iloc[:-12], s.iloc[-12:]
        for name, preds in _fit_forecast(y_train, horizon=12).items():
            frames.append(pd.DataFrame({
                "Level": lv, "FiscalMonth": range(1, 13), "Method": name,
                "Actual": y_test.to_numpy(dtype=float), "Pred": np.asarray(preds, dtype=float),
            }))
    if not frames:
        return pd.DataFrame(columns=["Method", "Approach", "WAPE", "Bias", "Forecast", "Actual"]), \
               pd.DataFrame(columns=["Level", "FiscalMonth", "Method", "Actual", "Pred"])

    detail = pd.concat(frames, ignore_index=True)
    detail["Method"] = detail["Method"] + f" at {level_col}"
    rows = []
    for name, g in detail.groupby("Method"):
        rows.append({"Method": name, "Approach": "Direct",
                     "WAPE": wape(g["Actual"], g["Pred"]), "Bias": forecast_bias(g["Actual"], g["Pred"]),
                     "Forecast": float(g["Pred"].sum()), "Actual": float(g["Actual"].sum())})
    return pd.DataFrame(rows), detail
