"""Fixed settings shared by every page: build defaults, feature toggles, column labels,
colours, navigation and the stocking-rule defaults. Plain values only; nothing here draws.
"""
import os


# Stocking rules the build plans with. The Inventory Plan page changes them
# live (session key "inv_rules") and every page reads the re-made plan.
INV_DEFAULTS = dict(lead_time=2.0, review=3.0, service=0.95, legacy=1.5)


# =========================== DEFAULTS -- editable on the Build Forecast page ============
# Matches the real MELFT Sales Order History export:
# SO Internal ID, Sales Order Date, Sales Order Number, Sales Order Status,
# Customer Internal ID, Company Name, Item Internal ID, Item Code, Base Price, Quantity
DEFAULT_SHEET_NAME = "Sales Order History"
DEFAULT_COL_PRODUCT = "Item Internal ID"   # grouping key -- reliable, unlike names/codes
DEFAULT_COL_ITEM_CODE = "Item Code"        # display label only, not used for grouping/features
DEFAULT_COL_CATEGORY = ""                  # not in the real file; leave blank unless you add one
DEFAULT_COL_UNITS = "Quantity"
DEFAULT_COL_PRICE = "Base Price"           # Revenue = Base Price x Quantity, computed automatically
DEFAULT_COL_DATE = "Sales Order Date"
DEFAULT_COL_STATUS = "Sales Order Status"  # used to exclude cancelled/pending/etc. orders
DEFAULT_FISCAL_START_MONTH = 4  # April
COMPANY_NAME = "MITSUBISHI ELECTRIC"

# Build Forecast -> "Feature set" expander. Each row is (toggle key, label,
# default-on). The key must match either a FEATURE_BASE column, an
# extra-feature column, or the special "Category" flag that build_everything()
# checks with _use(). "month_sin"/"month_cos" are toggled together via the
# "seasonality" pseudo-key.
# The experimental B2B rows are ordinary _use()-gated toggles that default OFF
# -- not a separate boolean. The ablation found them flat on the headline WAPE,
# so they are opt-in, but they are inspectable and switchable like everything
# else. Their column names must stay in step with REORDER_CADENCE_COLS /
# CUSTOMER_MIX_COLS in forecast_core (the labels are duplicated here only
# because PRETTY_COLS is defined further down the module).
FEATURE_TOGGLE_GROUPS = {
    "Recent demand": [
        ("lag_1", "Units 1 month ago", True),
        ("lag_2", "Units 2 months ago", True),
        ("lag_3", "Units 3 months ago", True),
        ("lag_12", "Units same month last year", True),
        ("roll_mean_3", "Trailing 3-month average", True),
        ("roll_mean_6", "Trailing 6-month average", True),
    ],
    "Item behaviour": [
        ("avg_monthly_units", "Average units per month (all history)", True),
        ("months_since_last_sale", "Months since last sale", True),
        ("sparsity_coverage", "Sparsity (% of months active)", True),
    ],
    "Seasonality": [
        ("seasonality", "Calendar month (sin/cos encoding)", True),
    ],
    "Product & customer": [
        ("Category", "Product category (Category 1 & 2)", True),
        ("ListPrice", "List price", True),
        ("distinct_customers_trailing_12m", "Distinct customers (trailing 12m)", True),
    ],
    "Experimental B2B, off by default": [
        ("customers_due_reorder", "Customers due to reorder", False),
        ("reorder_pressure", "Reorder pressure", False),
        ("reorder_readiness", "Reorder readiness", False),
        ("vip_demand_share_12m", "VIP demand share", False),
        ("wholesale_demand_share_12m", "Wholesale demand share", False),
        ("buyer_industry_count_12m", "Buyer industries", False),
        ("top_industry_share_12m", "Top industry share", False),
    ],
}
# Flattened views for the Select all / Deselect all / Reset buttons.
ALL_FEATURE_KEYS = [k for rows in FEATURE_TOGGLE_GROUPS.values() for k, _lbl, _d in rows]
FEATURE_DEFAULTS = {k: d for rows in FEATURE_TOGGLE_GROUPS.values() for k, _lbl, d in rows}
# The toggle labels double as display names for the model's feature columns (importance chart).
# Only snake_case keys: "Category" and "seasonality" are toggles, not columns.
FEATURE_LABELS = {k: lbl for rows in FEATURE_TOGGLE_GROUPS.values() for k, lbl, _d in rows if "_" in k}

# Human-readable labels for every internal (snake_case / suffixed) column
# name that gets shown in a table or CSV. prettify() applies this right
# before st.dataframe / to_csv so the code keeps its terse names but the
# operator never sees an underscore.
PRETTY_COLS = {
    "Item Internal ID": "Item ID", "Customer Internal ID": "Customer ID",
    "forecast_monthly_units": "Forecast / month", "forecast_horizon_units": "Forecast / year",
    "expected_demand_protection": "Demand over lead + review",
    "protection_period_months": "Protection period (months)",
    "safety_stock": "Safety stock", "legacy_safety_stock": "Current-rule safety stock",
    "safety_stock_delta_vs_legacy": "Safety stock vs current rule",
    "delta_value_thb": "Value of change (฿)",
    "reorder_point": "Reorder point", "order_up_to_level": "Order-up-to level",
    "recommended_action": "Action", "on_hand": "On hand",
    "suggested_order_qty": "Suggested order qty", "months_of_cover_on_hand": "Months of cover",
    "list_price": "List price (฿)", "safety_stock_value_thb": "Safety stock value (฿)",
    "suggested_order_value_thb": "Suggested order value (฿)",
    "reasons": "Flagged for", "trailing_actual_units": "Units sold (last 12m)",
    "forecast_next_units": "Forecast next 12m", "value_at_risk_thb": "Value at risk (฿)",
    "value_basis": "Valued from",
    "Baseline_Forecast": "Baseline", "ML_Forecast": "Model", "Final_Forecast": "Forecast",
    "q10": "Low (p10)", "q50": "Expected (p50)", "q90": "High (p90)",
    "p50_monthly_units": "p50 / month", "p90_monthly_units": "p90 / month",
    "safety_stock_p90": "Safety stock (p90 method)", "p90_vs_formula_ratio": "p90 vs formula",
    "n_rows": "Rows", "n_features": "Features", "n_items": "Items", "N items": "Items",
    "WAPE (SKU-month)": "WAPE (item × month)", "WAPE (Category-month)": "WAPE (category × month)",
    "ValidationYear": "Validated on FY", "MASE": "MASE (vs naive)",
    "MAPE (row-average %)": "MAPE", "WAPE (volume-weighted %)": "WAPE", "MAE (avg units off by)": "MAE",
    "RMSE (big misses weigh more)": "RMSE",
    "vip_demand_share_12m": "VIP demand share", "wholesale_demand_share_12m": "Wholesale demand share",
    "buyer_industry_count_12m": "Buyer industries", "top_industry_share_12m": "Top industry share",
    "customers_due_reorder": "Customers due to reorder", "reorder_pressure": "Reorder pressure",
    "reorder_readiness": "Reorder readiness", "distinct_customers_trailing_12m": "Distinct customers (12m)",
    "ProductionStatus": "Production status", "ListPrice": "List price",
    "Category1": "Category 1", "Category2": "Category 2", "AbsError": "Absolute error",
    "Sparsity Tier": "Sparsity tier",
}


NAVY = "#17233F"; NAVY_DARK = "#24345C"; ORANGE = "#B85400"  # redesign: action navy and forecast orange from the Forecast Console design system


GREEN = "#27AE60"; RED = "#EB5757"

CHART_ACTUAL = NAVY  # dark navy reads clearly against a light chart background (this is the light-theme counterpart of the CHART_ACTUAL workaround the dark theme needed)
BG ="#F4F4F1"; CARD = "#FFFFFF"; TEXT_MAIN = "#16181D"; TEXT_MUTED = "#4D535F"



BORDER_SUBTLE = "#DEDCD5"
BORDER_CONTROL = "#737884"  # design system --border-control: secondary buttons, inputs, segmented track
NAV_ACTIVE = "#EBEAE5"  # grey pill behind the current page in the sidebar
CHART_GRID = "#E3E1DA"  # design system --chart-grid
CARD_RADIUS = "12px"
# Relative to the working directory, like the item / customer masters.
LOGO_PATH = os.path.join("assets", "logo.png")
# Sidebar page -> Material Symbols icon. Only the label changes (format_func);
# the radio's values stay the plain page names that every `page ==` test and
# the _redirect_to hand-off compare against. build / home / inventory /
# table_chart / fact_check are the glyphs in the design mockup ("assets").
# Ordered as the board's three nav groups; NAV_GROUP_STARTS gives the 1-based
# position of each group's first page, where the CSS draws the group label.
NAV_ICONS = {
    "Overview": "home", "Inventory Plan": "inventory_2",
    "Category Plan": "category", "Product Detail": "inventory", "Customer Detail": "groups",
    "Forecast Table": "table_chart",
    "Model Accuracy": "fact_check", "Build Forecast": "build",
}
NAV_GROUP_STARTS = {1: "Plan", 3: "Explore", 7: "Model"}


FRONTIER_SERVICE_LEVELS = (0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99)
FRONTIER_FLAT_MONTHS = (1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12)


ACC_SECTIONS = ["Business impact", "Model accuracy", "Diagnostics"]


# Vega's category20c, spelled out so a chart that shows only some categories
# plus an "Other" stack can give each one the colour it has everywhere else.
CATEGORY20C = ["#3182bd", "#6baed6", "#9ecae1", "#c6dbef", "#e6550d", "#fd8d3c", "#fdae6b", "#fdd0a2",
               "#31a354", "#74c476", "#a1d99b", "#c7e9c0", "#756bb1", "#9e9ac8", "#bcbddc", "#dadaeb",
               "#636363", "#969696", "#bdbdbd", "#d9d9d9"]
