"""Test the new customer-mix / list-price / Production-Status features,
the walk-forward backtest, and the inventory-planning + obsolescence layer
on the real MELFT data."""
import time, numpy as np, pandas as pd
from harness import (load_prepared, DATA, ITEM_MASTER, COL_PRODUCT, COL_CUSTOMER,
                     COL_DATE, COL_UNITS, TRAIN_YEARS, VALIDATION_YEAR, FISCAL_START, _ROOT)
import os
import forecast_core as fc
import inventory as inv

pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)

raw, df, _ = load_prepared()

# --- item master: Production Status + List price ---
im = pd.read_excel(ITEM_MASTER); im.columns = [c.strip() for c in im.columns]
im["Item Internal ID"] = im["Item Internal ID"].astype(str)
status_map = im.set_index("Item Internal ID")["Production Status"].to_dict()
price_map = im.set_index("Item Internal ID")["List price"].to_dict()
df["ProductionStatus"] = df[COL_PRODUCT].map(status_map)
df["ListPrice"] = pd.to_numeric(df[COL_PRODUCT].map(price_map), errors="coerce").fillna(0)

# --- customer master: type + industry maps ---
cm = pd.read_excel(os.path.join(_ROOT, "data", "Customer_master.xlsx"))
cm.columns = [c.strip() for c in cm.columns]
ct_map = cm.set_index("Customer Internal ID")["Customer Type"].to_dict()
ind_map = cm.set_index("Customer Internal ID")["FA Sub Industries"].to_dict()

t0 = time.time()
mix = fc.compute_customer_mix_features(raw, COL_PRODUCT, COL_CUSTOMER, COL_DATE, COL_UNITS,
                                       FISCAL_START, customer_type_map=ct_map, industry_map=ind_map)
print(f"customer-mix features: {time.time()-t0:.1f}s, rows={len(mix)}")
print(mix.describe().round(3).to_string())

cad = fc.compute_reorder_cadence_feature(raw, COL_PRODUCT, COL_CUSTOMER, COL_DATE, FISCAL_START)
for extra_df in (mix, cad):
    df = df.merge(extra_df, on=[COL_PRODUCT, "FiscalYear", "FiscalMonth"], how="left")
mix_cols = ["vip_demand_share_12m", "wholesale_demand_share_12m", "buyer_industry_count_12m", "top_industry_share_12m"]
cad_cols = ["customers_due_reorder", "reorder_pressure", "reorder_readiness"]
for c in mix_cols + cad_cols + ["ListPrice"]:
    df[c] = df[c].fillna(0)

cats = ["Category1", "Category2", "ProductionStatus"]

for label, extra in [
    ("baseline extras (cust count only)", ["distinct_customers_trailing_12m"]),
    ("+ cadence", ["distinct_customers_trailing_12m"] + cad_cols),
    ("+ cadence + customer-mix + list price", ["distinct_customers_trailing_12m"] + cad_cols + mix_cols + ["ListPrice"]),
]:
    res_df, metrics, feat_cols, trained, sk, (tX, tY) = fc.run_backtest(
        df, COL_PRODUCT, cats, TRAIN_YEARS, VALIDATION_YEAR, FISCAL_START,
        ["LightGBM (Tweedie)"], extra_features=extra)
    rep = fc.evaluate_backtest(res_df, fc.add_sparsity_feature(df, COL_PRODUCT, TRAIN_YEARS),
                               COL_PRODUCT, TRAIN_YEARS, category_cols=cats, fiscal_start_month=FISCAL_START)
    print(f"\n--- {label} ---")
    print(rep["summary"].loc[["Baseline", "LightGBM (Tweedie)"]].round(3).to_string())
    if "mix" in label:
        imp, method = fc.compute_feature_importance(trained["LightGBM (Tweedie)"], feat_cols, tX, tY)
        imp = fc.group_importance_by_category(imp, cats)
        print(imp.head(16).to_string(index=False))

# --- walk-forward ---
print("\n=== WALK-FORWARD BACKTEST ===")
wf = fc.walk_forward_backtest(df, COL_PRODUCT, cats, FISCAL_START, ["LightGBM (Tweedie)"],
                              all_years=[2021, 2022, 2023, 2024], min_train_years=2,
                              extra_features=["distinct_customers_trailing_12m"] + cad_cols + mix_cols + ["ListPrice"])
print(wf.round(3).to_string(index=False))

# --- final model + forecast + inventory plan ---
extra = ["distinct_customers_trailing_12m"] + cad_cols + mix_cols + ["ListPrice"]
fm, _, fcols = fc.train_final_model(df, COL_PRODUCT, cats, TRAIN_YEARS, VALIDATION_YEAR, FISCAL_START,
                                 "LightGBM (Tweedie)", extra_features=extra)
df_ws = fc.add_sparsity_feature(df, COL_PRODUCT, TRAIN_YEARS + [VALIDATION_YEAR])
future = fc.forecast_future(df_ws, fm, COL_PRODUCT, cats, fcols, 2025, FISCAL_START)
print("\nforecast total units FY2025:", round(future["Final_Forecast"].sum()))

sigma = inv.forecast_error_sigma(res_df, "LightGBM (Tweedie)", COL_PRODUCT)
plan = inv.build_inventory_plan(future, COL_PRODUCT, horizon_months=12,
                                lead_time_months=2.0, review_period_months=3.0, service_level=0.95,
                                sigma_by_product=sigma.to_dict(), list_price=price_map)
print("\n=== INVENTORY PLAN (sample: top 8 by forecast) ===")
top = plan.sort_values("forecast_horizon_units", ascending=False).head(8)
print(top[[COL_PRODUCT, "forecast_monthly_units", "safety_stock", "reorder_point",
           "order_up_to_level", "legacy_safety_stock", "safety_stock_delta_vs_legacy", "recommended_action"]].to_string(index=False))
print("\ntotal safety-stock value (THB):", round(plan["safety_stock_value_thb"].sum()))

obs = inv.obsolescence_flags(future, df_ws, COL_PRODUCT, production_status=status_map, list_price=price_map)
print(f"\n=== OBSOLESCENCE FLAGS: {len(obs)} items ===")
print(obs.head(12).to_string(index=False))
print("total value at risk (THB):", round(obs["value_at_risk_thb"].sum()))
print("by reason:", obs["reasons"].value_counts().to_dict())
