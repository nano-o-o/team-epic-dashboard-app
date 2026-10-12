"""Feature ablation: does each group of engineered features actually earn its
place, or is the model already at its floor with just the obvious lags?

Adds one group at a time and reports SKU-month WAPE (FY2024 validation) plus
Category1 x month WAPE. Also runs the fullest set through walk-forward so we
can see whether extra features cause overfitting (they'd make the earlier
validation year worse)."""
import numpy as np, pandas as pd, time
from harness import (load_prepared, ITEM_MASTER, COL_PRODUCT, COL_CUSTOMER, COL_DATE, COL_UNITS,
                     TRAIN_YEARS, VALIDATION_YEAR, FISCAL_START, _ROOT)
import os, forecast_core as fc

pd.set_option("display.width", 200)

raw, df, _ = load_prepared()

# attach item-master + customer-master derived columns/features
im = pd.read_excel(ITEM_MASTER); im.columns = [c.strip() for c in im.columns]
im["Item Internal ID"] = im["Item Internal ID"].astype(str)
df["ProductionStatus"] = df[COL_PRODUCT].map(im.set_index("Item Internal ID")["Production Status"].to_dict())
df["ListPrice"] = pd.to_numeric(df[COL_PRODUCT].map(im.set_index("Item Internal ID")["List price"].to_dict()), errors="coerce").fillna(0)

cm = pd.read_excel(os.path.join(_ROOT, "data", "Customer_master.xlsx")); cm.columns = [c.strip() for c in cm.columns]
ct_map = cm.set_index("Customer Internal ID")["Customer Type"].to_dict()
ind_map = cm.set_index("Customer Internal ID")["FA Sub Industries"].to_dict()

cad = fc.compute_reorder_cadence_feature(raw, COL_PRODUCT, COL_CUSTOMER, COL_DATE, FISCAL_START)
mix = fc.compute_customer_mix_features(raw, COL_PRODUCT, COL_CUSTOMER, COL_DATE, COL_UNITS, FISCAL_START,
                                       customer_type_map=ct_map, industry_map=ind_map)
# distinct_customers_trailing_12m is already merged by load_prepared()
for d in (cad, mix):
    df = df.merge(d, on=[COL_PRODUCT, "FiscalYear", "FiscalMonth"], how="left")
cad_cols = ["customers_due_reorder", "reorder_pressure", "reorder_readiness"]
mix_cols = ["vip_demand_share_12m", "wholesale_demand_share_12m", "buyer_industry_count_12m", "top_industry_share_12m"]
for c in cad_cols + mix_cols + ["distinct_customers_trailing_12m", "ListPrice"]:
    df[c] = df[c].fillna(0)

# FEATURE_BASE in forecast_core = lag_1,2,3,12 / roll_mean_3,6 / avg_monthly_units /
# months_since_last_sale / month_sin,cos / sparsity_coverage.
# We can't easily strip FEATURE_BASE without touching core, so we ablate the
# CATEGORY + EXTRA feature groups (the "created" ones the teacher is asking about)
# on top of the always-on base.
STEPS = [
    ("A. base only (lags/rolling/calendar/sparsity)", None, []),
    ("B. + Category1/2",                              ["Category1", "Category2"], []),
    ("C. + ProductionStatus",                        ["Category1", "Category2", "ProductionStatus"], []),
    ("D. + ListPrice",                               ["Category1", "Category2", "ProductionStatus"], ["ListPrice"]),
    ("E. + distinct customers",                      ["Category1", "Category2", "ProductionStatus"], ["ListPrice", "distinct_customers_trailing_12m"]),
    ("F. + reorder cadence",                         ["Category1", "Category2", "ProductionStatus"], ["ListPrice", "distinct_customers_trailing_12m"] + cad_cols),
    ("G. + customer mix (FULL)",                     ["Category1", "Category2", "ProductionStatus"], ["ListPrice", "distinct_customers_trailing_12m"] + cad_cols + mix_cols),
]

print(f"{'step':46s} {'WAPE SKUmo':>11s} {'WAPE Cat1mo':>12s} {'MASE':>6s} {'Bias':>7s} {'#feat':>6s}")
for label, cats, extra in STEPS:
    res_df, metrics, feat_cols, trained, sk, _ = fc.run_backtest(
        df, COL_PRODUCT, cats, TRAIN_YEARS, VALIDATION_YEAR, FISCAL_START,
        ["LightGBM (Tweedie)"], extra_features=extra)
    rep = fc.evaluate_backtest(res_df, fc.add_sparsity_feature(df, COL_PRODUCT, TRAIN_YEARS),
                               COL_PRODUCT, TRAIN_YEARS, category_cols=cats, fiscal_start_month=FISCAL_START)
    s = rep["summary"].loc["LightGBM (Tweedie)"]
    g = rep["by_grain"]["LightGBM (Tweedie)"]
    cat1 = g.get("Category1 x month", float("nan"))
    print(f"{label:46s} {s['WAPE']:>11.3f} {cat1:>12.3f} {s['MASE']:>6.2f} {s['Bias']:>+7.1%} {len(feat_cols):>6d}")

print("\n--- walk-forward, FULL feature set (overfitting check) ---")
wf = fc.walk_forward_backtest(df, COL_PRODUCT, ["Category1", "Category2", "ProductionStatus"],
                              FISCAL_START, ["LightGBM (Tweedie)"], all_years=[2021, 2022, 2023, 2024],
                              min_train_years=2, extra_features=["ListPrice", "distinct_customers_trailing_12m"] + cad_cols + mix_cols)
print(wf[wf.Model == "LightGBM (Tweedie)"].round(3).to_string(index=False))
