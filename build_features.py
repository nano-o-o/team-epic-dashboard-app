"""
=============================================================================
 build_features.py -- reproducible offline feature build
=============================================================================
Produces the full model-ready feature matrix as a versioned file you can
open and inspect, instead of only ever seeing it computed inside the app.

    python build_features.py

Writes:
    features.parquet          -- one row per (Item, FiscalYear, FiscalMonth),
                                 every engineered feature, leakage-safe
    features_dictionary.csv   -- what each column is + its leakage note

IMPORTANT: this script and the app (ui/pipeline.py) call the SAME functions in
forecast_core.py -- there is one implementation of every feature, used both
offline (here) and at runtime (the app). Nothing is re-derived by hand, so
the two cannot drift apart.

One feature is deliberately NOT baked in here: `sparsity_coverage`. It is
the fraction of *training-window* months an item had any sales, so its value
depends on which years you train on -- computing it once, offline, would
leak information from the validation/forecast years back into training. The
app adds it per split (see forecast_core.add_sparsity_feature). Everything in
this file only ever looks backward in absolute calendar time, so it is safe
to precompute.
=============================================================================
"""
import pandas as pd

import forecast_core as fc

SALES_FILE = "data/cleaned_data.csv"
ITEM_MASTER_FILE = "data/Item_master.xlsx"
CUSTOMER_MASTER_FILE = "data/Customer_master.xlsx"

COL_PRODUCT = "Item Internal ID"
COL_CUSTOMER = "Customer Internal ID"
COL_UNITS = "Quantity"
COL_PRICE = "Base Price"
COL_DATE = "Sales Order Date"
COL_STATUS = "Sales Order Status"
FISCAL_START_MONTH = 4

# Only the lean feature set the ablation kept. Flip to True to also emit the
# experimental B2B columns (reorder cadence + customer mix).
INCLUDE_EXPERIMENTAL = False


FEATURE_DOC = {
    "Units": ("Target: units sold that fiscal month (0 where the item had no order).", "target"),
    "Revenue": ("Base Price x Quantity, summed for the month.", "target-derived"),
    "lag_1": ("Units 1 month ago.", "backward-only (shift 1)"),
    "lag_2": ("Units 2 months ago.", "backward-only"),
    "lag_3": ("Units 3 months ago.", "backward-only"),
    "lag_12": ("Units the same month last year (yearly seasonality).", "backward-only"),
    "roll_mean_3": ("Mean units over the trailing 3 months.", "backward-only (shift 1 then roll)"),
    "roll_mean_6": ("Mean units over the trailing 6 months.", "backward-only"),
    "avg_monthly_units": ("Average units per month over all of this item's prior months (zero months included).", "backward-only (shift 1, then an expanding mean within the item)"),
    "months_since_last_sale": ("Months since this item last had Units > 0, using prior months only.", "backward-only"),
    "month_sin": ("sin(2 pi * calendar month / 12) -- cyclical calendar encoding.", "deterministic"),
    "month_cos": ("cos(2 pi * calendar month / 12).", "deterministic"),
    "Category1": ("Primary product family, from the item master.", "static attribute"),
    "Category2": ("Sub-category, from the item master.", "static attribute"),
    "ProductionStatus": ("Continue / Discontinue, from the item master. Used for obsolescence flags, not as a model feature.", "static attribute"),
    "ListPrice": ("Official list price (฿), from the item master.", "static attribute"),
    "distinct_customers_trailing_12m": ("Count of distinct customers who bought this item in the trailing 12 months.", "backward-only (trailing window, current month excluded)"),
    "customers_due_reorder": ("Repeat buyers whose gap since last order exceeds their own average interval.", "backward-only"),
    "reorder_pressure": ("customers_due_reorder / repeat buyers.", "backward-only"),
    "reorder_readiness": ("Sum over repeat buyers of (months elapsed / their mean interval), capped.", "backward-only"),
    "vip_demand_share_12m": ("Share of trailing-12m units bought by VIP-type customers.", "backward-only"),
    "wholesale_demand_share_12m": ("Share of trailing-12m units bought by Wholesale & Distribution customers.", "backward-only"),
    "buyer_industry_count_12m": ("Distinct FA sub-industries buying this item in the trailing 12 months.", "backward-only"),
    "top_industry_share_12m": ("Largest single industry's share of trailing-12m units.", "backward-only"),
}


def main():
    print("Loading sales data...")
    raw = pd.read_csv(SALES_FILE)
    raw.columns = raw.columns.str.strip()
    raw, n_dropped = fc.clean_dates(raw, COL_DATE)
    if n_dropped:
        print(f"  dropped {n_dropped:,} rows with invalid dates")
    raw[COL_PRODUCT] = raw[COL_PRODUCT].astype(str)
    raw, n_dup = fc.drop_duplicate_lines(raw)
    print(f"  dropped {n_dup:,} exact duplicate lines")
    raw = fc.compute_revenue(raw, COL_PRICE, COL_UNITS, revenue_col="Revenue")

    print("Joining item master (category, production status, list price)...")
    im = pd.read_excel(ITEM_MASTER_FILE)
    im.columns = im.columns.str.strip()
    attr_map = {"Category1": "Product Category 1", "Category2": "Product Category 2",
                "ProductionStatus": "Production Status", "ListPrice": "List price"}
    attr_map = {k: v for k, v in attr_map.items() if v in im.columns}
    raw = fc.merge_item_master(raw, COL_PRODUCT, im, "Item Internal ID", attr_map)
    if "ListPrice" in raw.columns:
        raw, n_unpriced = fc.drop_unpriced_items(raw, COL_PRODUCT, im, "Item Internal ID", "List price")
        print(f"  dropped {n_unpriced:,} items with no list price")
        raw["ListPrice"] = pd.to_numeric(raw["ListPrice"], errors="coerce")

    passthrough = {k: k for k in attr_map}
    passthrough["Item Code"] = "Item Code"
    df = fc.aggregate_to_fiscal_monthly(raw, COL_PRODUCT, COL_UNITS, "Revenue", COL_DATE,
                                        FISCAL_START_MONTH, passthrough_cols=passthrough)
    if "ListPrice" in df.columns:
        df["ListPrice"] = pd.to_numeric(df["ListPrice"], errors="coerce").fillna(0)

    print("Building time-series features (lags, rolling means, calendar, recency)...")
    df = fc.add_features(df, COL_PRODUCT, FISCAL_START_MONTH)

    print("Building trailing distinct-customer count...")
    cust = fc.compute_distinct_customers_trailing(raw, COL_PRODUCT, COL_CUSTOMER, COL_DATE, FISCAL_START_MONTH)
    df = df.merge(cust, on=[COL_PRODUCT, "FiscalYear", "FiscalMonth"], how="left")
    df["distinct_customers_trailing_12m"] = df["distinct_customers_trailing_12m"].fillna(0)

    if INCLUDE_EXPERIMENTAL:
        print("Building experimental B2B features (reorder cadence + customer mix)...")
        cad = fc.compute_reorder_cadence_feature(raw, COL_PRODUCT, COL_CUSTOMER, COL_DATE, FISCAL_START_MONTH)
        df = df.merge(cad, on=[COL_PRODUCT, "FiscalYear", "FiscalMonth"], how="left")
        cm = pd.read_excel(CUSTOMER_MASTER_FILE)
        cm.columns = cm.columns.str.strip()
        ct = cm.set_index("Customer Internal ID")["Customer Type"].to_dict()
        ind = cm.set_index("Customer Internal ID")["FA Sub Industries"].to_dict()
        mix = fc.compute_customer_mix_features(raw, COL_PRODUCT, COL_CUSTOMER, COL_DATE, COL_UNITS,
                                               FISCAL_START_MONTH, customer_type_map=ct, industry_map=ind)
        df = df.merge(mix, on=[COL_PRODUCT, "FiscalYear", "FiscalMonth"], how="left")
        for c in fc.REORDER_CADENCE_COLS + fc.CUSTOMER_MIX_COLS:
            df[c] = df[c].fillna(0)

    df = df.sort_values([COL_PRODUCT, "FiscalYear", "FiscalMonth"]).reset_index(drop=True)

    out_file = "data/features.parquet"
    df.to_parquet(out_file, index=False)
    print(f"\nWrote {out_file}: {len(df):,} rows x {df.shape[1]} columns")
    print(f"  products: {df[COL_PRODUCT].nunique():,}  |  fiscal years: {sorted(df['FiscalYear'].unique())}")
    print(f"  zero-demand share: {(df['Units'] == 0).mean():.1%}")

    doc = pd.DataFrame(
        [{"column": c,
          "description": FEATURE_DOC.get(c, ("(identifier / passthrough)", "n/a"))[0],
          "leakage_note": FEATURE_DOC.get(c, ("", "n/a"))[1]}
         for c in df.columns]
    )
    doc.to_csv("data/features_dictionary.csv", index=False)
    print("Wrote features_dictionary.csv")

    print("\nNote: sparsity_coverage is added per train/validation split by the app "
          "(add_sparsity_feature) and is intentionally NOT in this file -- see the module docstring.")


if __name__ == "__main__":
    main()
