"""
Reusable experiment harness: loads the real MELFT data + item master the same
way ui.pipeline.build_everything does, and exposes a prepared `df`
(fiscal-monthly, zero-filled, with Category1/Category2 and
distinct_customers_trailing_12m) plus the train/validation year split.

Run from anywhere:  python experiments/hierarchical_reconciliation.py
(paths below resolve relative to the project root, one level up from here).
"""
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import forecast_core as fc

DATA = os.path.join(_ROOT, "data", "cleaned_data.csv")
ITEM_MASTER = os.path.join(_ROOT, "data", "Item_master.xlsx")

COL_PRODUCT = "Item Internal ID"
COL_UNITS = "Quantity"
COL_PRICE = "Base Price"
COL_DATE = "Sales Order Date"
COL_STATUS = "Sales Order Status"
COL_CUSTOMER = "Customer Internal ID"
FISCAL_START = 4

TRAIN_YEARS = [2021, 2022, 2023]
VALIDATION_YEAR = 2024


def load_prepared():
    raw = pd.read_csv(DATA)
    raw.columns = [c.strip() if isinstance(c, str) else c for c in raw.columns]
    raw, n_dropped = fc.clean_dates(raw, COL_DATE)
    raw[COL_PRODUCT] = raw[COL_PRODUCT].astype(str)
    raw, _ = fc.drop_duplicate_lines(raw)
    raw = fc.compute_revenue(raw, COL_PRICE, COL_UNITS, revenue_col="Revenue")

    im = pd.read_excel(ITEM_MASTER)
    im.columns = [c.strip() if isinstance(c, str) else c for c in im.columns]
    raw, _ = fc.drop_unpriced_items(raw, COL_PRODUCT, im, "Item Internal ID", "List price")
    cat_map = {"Category1": "Product Category 1", "Category2": "Product Category 2"}
    raw = fc.merge_item_master(raw, COL_PRODUCT, im, "Item Internal ID", cat_map)

    passthrough = {"Category1": "Category1", "Category2": "Category2", "Item Code": "Item Code"}
    df = fc.aggregate_to_fiscal_monthly(raw, COL_PRODUCT, COL_UNITS, "Revenue", COL_DATE,
                                        FISCAL_START, passthrough_cols=passthrough)

    cust = fc.compute_distinct_customers_trailing(raw, COL_PRODUCT, COL_CUSTOMER, COL_DATE, FISCAL_START)
    df = df.merge(cust, on=[COL_PRODUCT, "FiscalYear", "FiscalMonth"], how="left")
    df["distinct_customers_trailing_12m"] = df["distinct_customers_trailing_12m"].fillna(0)

    return raw, df, n_dropped


if __name__ == "__main__":
    raw, df, n_dropped = load_prepared()
    print("raw rows:", len(raw), "dropped dates:", n_dropped)
    print("prepared df rows:", len(df))
    print("products:", df[COL_PRODUCT].nunique())
    print("fiscal years:", sorted(df["FiscalYear"].unique()))
    print("zero share:", (df["Units"] == 0).mean())
    print(df.head())
