"""
Category-level hierarchical / top-down forecasting experiment.

RESULT (real MELFT data, FY2024 validation): top-down and proportionally
reconciled forecasts do NOT beat the direct global SKU model here --
SKU-month WAPE 0.73-0.75 vs 0.69 direct. The direct LightGBM already pools
cross-sectional signal through its one-hot Category features, so re-deriving
it hierarchically only loses SKU-specific timing signal. A 50/50 blend of
direct + top-down ("middle-out") is a marginal help on Dense items only.
Kept as a documented negative result, not wired into the app.
"""
import numpy as np, pandas as pd
from harness import load_prepared, COL_PRODUCT, TRAIN_YEARS, VALIDATION_YEAR, FISCAL_START
import forecast_core as fc

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 25)
raw, df, _ = load_prepared()

CATCOL = "Category2"
MODEL = "LightGBM (Tweedie)"


def forecast_at_level(frame, level_col, train_years, val_year):
    """Aggregate to `level_col` x fiscal-month, train the GBM on that dense
    series, return predictions for val_year as {(level, fy, fm): pred}."""
    agg = frame.groupby([level_col, "FiscalYear", "FiscalMonth"], as_index=False)["Units"].sum()
    # full monthly grid per level
    levels = agg[level_col].unique()
    fyrs = sorted(agg["FiscalYear"].unique())
    idx = pd.MultiIndex.from_product([levels, fyrs, range(1, 13)],
                                     names=[level_col, "FiscalYear", "FiscalMonth"])
    agg = agg.set_index([level_col, "FiscalYear", "FiscalMonth"]).reindex(idx, fill_value=0).reset_index()
    agg = fc.add_sparsity_feature(agg, level_col, train_years)
    feat = fc.add_features(agg, level_col, FISCAL_START)
    feat, cols = fc.build_feature_matrix(feat, fc.FEATURE_BASE, None)
    tr = feat[feat["FiscalYear"].isin(train_years)].dropna(subset=cols)
    te = feat[feat["FiscalYear"] == val_year].dropna(subset=cols)
    m = fc.make_model(MODEL); m.fit(tr[cols], tr["Units"])
    te = te.copy(); te["pred"] = np.maximum(0, m.predict(te[cols]))
    return te.set_index([level_col, "FiscalYear", "FiscalMonth"])["pred"].to_dict(), agg


# --- SKU share of its category, from the last training year (leakage-safe) ---
last_ty = max(TRAIN_YEARS)
share_base = df[df["FiscalYear"] == last_ty].groupby([CATCOL, COL_PRODUCT])["Units"].sum().reset_index()
cat_tot = share_base.groupby(CATCOL)["Units"].transform("sum")
share_base["share"] = np.where(cat_tot > 0, share_base["Units"] / cat_tot, 0.0)
share_map = share_base.set_index(COL_PRODUCT)["share"].to_dict()
cat_of = df.drop_duplicates(COL_PRODUCT).set_index(COL_PRODUCT)[CATCOL].to_dict()

cat_pred, _ = forecast_at_level(df, CATCOL, TRAIN_YEARS, VALIDATION_YEAR)

# --- build the validation res_df with a direct SKU forecast to compare ---
res_df, metrics, feat_cols, trained, skipped, _ = fc.run_backtest(
    df, COL_PRODUCT, ["Category1", "Category2"], TRAIN_YEARS, VALIDATION_YEAR,
    FISCAL_START, [MODEL], extra_features=["distinct_customers_trailing_12m"])

# top-down prediction for each validation row
def td(row):
    c = cat_of.get(row[COL_PRODUCT])
    return cat_pred.get((c, row["FiscalYear"], row["FiscalMonth"]), 0.0) * share_map.get(row[COL_PRODUCT], 0.0)
res_df["TopDown (Cat2)_Pred"] = res_df.apply(td, axis=1)
res_df["TopDown (Cat2)_AE"] = (res_df["Actual"] - res_df["TopDown (Cat2)_Pred"]).abs()

# middle-out: average of direct SKU forecast and top-down
res_df["MiddleOut_Pred"] = 0.5 * (res_df[f"{MODEL}_Pred"] + res_df["TopDown (Cat2)_Pred"])
res_df["MiddleOut_AE"] = (res_df["Actual"] - res_df["MiddleOut_Pred"]).abs()

# reconciled: scale direct SKU preds within each category-month to match cat forecast
r = res_df.copy()
r["cat"] = r[COL_PRODUCT].map(cat_of)
grp = r.groupby(["cat", "FiscalYear", "FiscalMonth"])[f"{MODEL}_Pred"].transform("sum")
r["catfc"] = r.apply(lambda x: cat_pred.get((x["cat"], x["FiscalYear"], x["FiscalMonth"]), np.nan), axis=1)
factor = np.where(grp > 0, r["catfc"] / grp, 1.0)
res_df["Reconciled_Pred"] = np.where(np.isfinite(factor), r[f"{MODEL}_Pred"] * factor, r[f"{MODEL}_Pred"])
res_df["Reconciled_AE"] = (res_df["Actual"] - res_df["Reconciled_Pred"]).abs()

df_sp = fc.add_sparsity_feature(df, COL_PRODUCT, TRAIN_YEARS)
rep = fc.evaluate_backtest(res_df, df_sp, COL_PRODUCT, TRAIN_YEARS,
                           category_cols=["Category1", "Category2"], fiscal_start_month=FISCAL_START)
print("=== summary (all SKU-months) ===")
print(rep["summary"].round(3).to_string())
print("\n=== by sparsity (WAPE) ===")
print(rep["by_sparsity"].round(3).to_string(index=False))
print("\n=== by grain (WAPE) ===")
print(rep["by_grain"].round(3).to_string())
