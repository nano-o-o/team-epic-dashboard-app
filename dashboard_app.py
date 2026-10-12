"""MELFT demand forecast dashboard -- the Streamlit UI over forecast_core.py,
inventory.py and classical.py. Runs locally; nothing leaves the machine.

    streamlit run dashboard_app.py

Upload the sales export on the Build Forecast page and click Build Forecast.
"""

import contextvars
import hashlib
import html
import io
import os
import pickle
import re
import sys
import threading
import time
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from forecast_core import (
    LIGHTGBM_AVAILABLE, XGBOOST_AVAILABLE, to_fiscal, fiscal_month_labels,
    aggregate_to_fiscal_monthly, add_sparsity_feature, run_backtest,
    train_final_model, forecast_future, compute_revenue, filter_by_status,
    clean_dates, drop_duplicate_lines, drop_unpriced_items, available_model_types, merge_item_master, compute_distinct_customers_trailing,
    compute_feature_importance, group_importance_by_category, evaluate_backtest,
    compute_reorder_cadence_feature, compute_customer_mix_features, walk_forward_backtest,
    feature_ablation, FEATURE_BASE, QUANTILES, quantile_backtest,
    REORDER_CADENCE_COLS, CUSTOMER_MIX_COLS, order_rhythm,
    compute_bias_correction, apply_bias_correction, _bias_factor_series,
    wape as _wape_fn, forecast_bias as _bias_fn,
    UNKNOWN_LEVEL, level_map_from, rollup_backtest_to_level, level_accuracy_breakdown,
    level_method_comparison, level_direct_forecast,
    HORIZON_LABELS, horizon_backtest, horizon_comparison,
)
from inventory import (forecast_error_sigma, build_inventory_plan, plan_for_rules,
                       service_level_comparison, policy_frontier, fill_rate_at_budget,
                       FAMILY_MODEL, FAMILY_FLAT)
from classical import aggregate_crosscheck, per_level_crosscheck

# Stocking rules the build plans with. The Inventory Plan page changes them
# live (session key "inv_rules") and every page reads the re-made plan.
INV_DEFAULTS = dict(lead_time=2.0, review=3.0, service=0.95, legacy=1.5)


def _set_inv_rule(name, key, scale=1):
    """on_change for the Inventory Plan rule inputs. Copied into a plain key
    because a widget's own key is dropped on any page that doesn't draw it.
    `scale` undoes a widget that shows the rule in other units (percent)."""
    st.session_state["inv_rules"] = {**st.session_state.get("inv_rules", INV_DEFAULTS),
                                     name: float(st.session_state[key]) / scale}
    remember_result("inv_rules")   # so a reload or restart keeps them


def live_inventory(results):
    """(inv_plan, obs_flags, build_meta) under the live stocking rules. The
    build plans with INV_DEFAULTS; when the rules set on Inventory Plan differ,
    re-plan (a second or so, no model involved). Only the latest re-plan is
    kept, keyed on build and rules; build_meta is copied so the saved build
    stays as built."""
    fc, inv_plan, obs_flags, meta = results[1], results[10], results[11], results[12]
    rules = st.session_state.get("inv_rules", INV_DEFAULTS)
    if meta.get("inv_inputs") is None or rules == meta.get("inv_params"):
        return inv_plan, obs_flags, meta
    rk = (st.session_state.get("build_version", 0), tuple(rules.items()))
    if st.session_state.get("_inv_replan", (None,))[0] != rk:
        st.session_state["_inv_replan"] = (rk, plan_for_rules(
            fc, st.session_state["build_config"]["col_product"], meta["inv_inputs"], rules))
    return (*st.session_state["_inv_replan"][1], {**meta, "inv_params": dict(rules)})

# Build progress. build_everything is cached, and Streamlit replays every element a
# cached function drew -- which fails when the element sits in a container made
# outside the function (the Build Forecast status box). So stage messages are
# written with the cache's recorder switched off, in a copied context. If that
# internal flag ever moves, stages are silently skipped; the build is unaffected.
try:
    from streamlit.runtime.scriptrunner_utils.script_run_context import in_cached_function as _in_cached_fn
except ImportError:
    _in_cached_fn = None
_build_progress = threading.local()


def _report_stage(msg):
    cb = getattr(_build_progress, "cb", None)
    if cb is None or _in_cached_fn is None:
        return

    def _untracked():
        assert _in_cached_fn is not None
        _in_cached_fn.set(False)
        cb(msg)
    contextvars.copy_context().run(_untracked)

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


def _pretty_label(c):
    if c in PRETTY_COLS:
        return PRETTY_COLS[c]
    if c in FEATURE_LABELS:
        return FEATURE_LABELS[c]
    if isinstance(c, str) and "_" in c:
        return c.replace("_", " ").strip().capitalize()
    return c


# Obsolescence reason tokens come out of inventory.py as machine-friendly
# snake_case ("declining_demand"). prettify() only renames COLUMNS, so these
# cell VALUES were the last place an underscore still reached the screen.
_OBS_REASONS = {
    "status_discontinued": "Discontinued",
    "declining_demand": "Declining demand",
    "overstocked_>3yr": "Over 3 years of cover",
}


def pretty_reasons(txt):
    """'status_discontinued, dormant_24m' -> 'Discontinued, No sales in 24 months'."""
    out = []
    for tok in str(txt).split(", "):
        if tok in _OBS_REASONS:
            out.append(_OBS_REASONS[tok])
        elif tok.startswith("dormant_") and tok.endswith("m"):
            out.append(f"No sales in {tok[len('dormant_'):-1]} months")
        else:
            out.append(tok.replace("_", " ").strip().capitalize())
    return ", ".join(out)


def prettify(df):
    """Rename any known internal column names to their operator-facing labels.
    Also snake_case -> Title Case for anything not in the map."""
    out = df.rename(columns=_pretty_label)
    if isinstance(out.index.name, str) and (out.index.name in PRETTY_COLS or "_" in out.index.name):
        out.index.name = _pretty_label(out.index.name)
    return out


def pretty_grain(name):
    """evaluate_backtest's grain keys ('SKU-month', 'Category1 x month') in the
    wording the rest of the app uses ('Item × month', 'Category 1 × month')."""
    return (str(name).replace("SKU-month", "Item × month").replace("SKU x", "Item x")
            .replace("Category1", "Category 1").replace("Category2", "Category 2").replace(" x ", " × "))


def localized_numbers(df):
    """column_config giving every numeric column thousands separators -- cheaper
    than a Styler on a table of thousands of rows."""
    return {c: st.column_config.NumberColumn(format="localized") for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])}


def item_pins(col_product):
    """column_config pinning Item ID and Item code on the left of an item table, as
    the design system's DataTable does. Keys absent from a table are ignored."""
    return {_pretty_label(col_product): st.column_config.Column(pinned=True),
            "Item Code": st.column_config.Column("Item code", pinned=True)}


# Table pill fills, the design system's StatusPill tones. Streamlit draws the text black
# on these light fills; it cannot take the tone's own text colour.
PILL = {"good": "#E0F0F3", "warn": "#FBEED3", "bad": "#FBE3E0", "neutral": "#EBEAE5"}
# Action pills: amber = act on it, grey = informational, teal = healthy.
ACTION_COLORS = {"Reorder": PILL["warn"], "Review - forecast zero, sold recently": PILL["neutral"],
                 "Review - no demand": PILL["neutral"], "Reduce - overstocked": PILL["warn"], "OK - hold": PILL["good"]}


def action_pills(label):
    """Recommended action as one coloured pill (the cell holds a one-item list)."""
    return st.column_config.MultiselectColumn(label, options=list(ACTION_COLORS),
                                              color=list(ACTION_COLORS.values()), width="medium")


def reason_pills(label, values):
    """Obsolescence reasons, one pill each; discontinued red, the rest amber."""
    opts = sorted({r for rs in values for r in rs})
    return st.column_config.MultiselectColumn(label, options=opts, width="medium",
                                              color=[PILL["bad"] if o == "Discontinued" else PILL["warn"] for o in opts])


def change_pills(label, pct_values, warn_below=0.0):
    """A column of percentage changes as Category Plan's "vs last FY" pills: "▲ 8%" /
    "▼ 40%", amber where the fall is bigger than `warn_below` (a positive number of
    points), grey otherwise. Returns (one-item lists for the grid, column config);
    a NaN change is an empty cell. Keep the plain text in any CSV."""
    cells, warn = [], {}
    for v in pct_values:
        if v is None or pd.isna(v):
            cells.append([])
            continue
        v = round(float(v))
        t = f"{'▼' if v < 0 else '▲'} {abs(v)}%"
        warn[t] = v < -warn_below
        cells.append([t])
    opts = sorted(warn)
    return cells, st.column_config.MultiselectColumn(
        label, options=opts, color=[PILL["warn"] if warn[o] else PILL["neutral"] for o in opts])


def fit_height(n_rows, max_rows=20):
    """st.dataframe height: every row for a short table, Streamlit's scrolling
    ten-row default for a long one."""
    return "content" if n_rows <= max_rows else "auto"


# Short model names for tight spaces (the per-model cards on Model Accuracy).
# Tables, captions, tooltips and exports keep the full name; these match the
# shorthand the deck and the numbers cheat-sheet already use.
MODEL_SHORT = {
    "LightGBM (Tweedie)": "LightGBM Tweedie",
    "LightGBM (standard)": "LightGBM std",
    "XGBoost (Tweedie)": "XGBoost",
    "HistGradientBoosting": "HGB",
    "Hurdle (2-stage)": "Hurdle",
    "Croston (SBA)": "Croston",
}


def short_model(name):
    return MODEL_SHORT.get(str(name), str(name))


def prettify_fmt(fmt):
    """Remap a {column: format} dict through the same labels prettify() uses,
    so a style.format() call still lines up after the columns are renamed."""
    return {_pretty_label(k): v for k, v in fmt.items()}


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

st.set_page_config(page_title=f"{COMPANY_NAME} Forecast Dashboard", page_icon=":material/insights:",
                   layout="wide", initial_sidebar_state="expanded")

# Full light theme: .streamlit/config.toml sets Streamlit's own base theme
# to light and matches these same colors, so native components (tables,
# tooltips, inputs) already render with correctly-contrasted text
# automatically -- this CSS only adds custom card/sidebar styling on top,
# it does NOT need to fight or re-assert Streamlit's own text color choices.
# (This is the same principle that fixed the earlier dark-theme migration:
# align with Streamlit's theme system instead of overriding it piecemeal.)
# Build Forecast shows one step at a time. The steps that are shut still
# RENDER their widgets -- Streamlit drops the state of a widget a run does not
# instantiate, and the build reads all three steps' values in one run -- so they
# are hidden here rather than skipped. This runs before the stylesheet because an
# Edit or Back/Next click is handled as a callback, i.e. before the script body,
# so the rule below is already right for the step that click opened.
st.session_state["_bf_step"] = min(st.session_state.get("_bf_step", 1), 3)
_STEP_HIDE_CSS = "".join(
    f'[class*="st-key-plain_body{_n}"] {{ display: none !important; }}'
    for _n in (1, 2, 3) if _n != st.session_state.get("_bf_step", 1))

st.markdown(f"""
<style>
    {_STEP_HIDE_CSS}
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
    .stApp {{ background-color: {BG}; }}
    .stApp, .stApp p, .stApp label, .stApp input, .stApp textarea, .stApp button p {{ font-family: "IBM Plex Sans", system-ui, sans-serif; }}
    header[data-testid="stHeader"] {{ background: transparent; }}
    /* Title close to the top of the page, level with the sidebar logo. */
    div[data-testid="stMainBlockContainer"] {{ padding-top: 2.5rem; }}

    /* Sidebar as a floating white card on the grey page. The section itself
       goes transparent and gets an inset; its content box carries the white
       fill and the rounded corners. */
    section[data-testid="stSidebar"] {{
        background-color: transparent !important; border-right: none; box-shadow: none;
        padding: 20px 0 20px 20px;
    }}
    section[data-testid="stSidebar"] > div[data-testid="stSidebarContent"] {{
        background-color: {CARD}; border-radius: 20px;
    }}
    section[data-testid="stSidebar"] div[data-testid="stSidebarHeader"] {{ padding-top: 22px; }}
    /* Streamlit leaves 96px of empty padding under the sidebar content, enough
       to push the 8 nav rows into a scrollbar on a 900px-tall screen. */
    section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {{ padding-bottom: 24px; }}
    .nav-divider {{ border-top: 3px solid {NAV_ACTIVE}; margin: 0 0 24px 0; }}
    .sidebar-wordmark {{ font-size: 22px; font-weight: 800; letter-spacing: -0.3px; color: {TEXT_MAIN}; margin: 4px 0 18px 0; }}
    /* Sidebar nav: plain rows with an icon, a grey pill on the current page,
       and the radio circle hidden -- it reads as a menu, not a form control. */
    section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 6px; width: 100%; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {{
        width: 100%; background-color: transparent; border: none; border-radius: 10px;
        padding: 3px 14px; margin: 0; transition: background-color 0.15s ease;
    }}
    /* Streamlit 1.63 option markup: label > span(hidden input) > div > div >
       [div(circle), div(stMarkdownContainer)]. The icon is a role="img" span. */
    section[data-testid="stSidebar"] div[role="radiogroup"] > label > div > div > div:first-child:not([data-testid]) {{ display: none; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label p {{ font-size: 16px; font-weight: 500; color: {TEXT_MAIN}; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label p [role="img"] {{ font-size: 22px; margin-right: 12px; vertical-align: middle !important; color: {TEXT_MAIN}; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{ background-color: #F5F5F5; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-selected="true"] {{ background-color: {NAV_ACTIVE}; }}

    /* Page title -- the large bold heading at the top of every page. */
    .page-title {{ font-size: 28px; font-weight: 600; line-height: 34px; letter-spacing: -0.01em; color: {TEXT_MAIN}; }}
    .page-subtitle {{ color: {TEXT_MUTED}; font-size: 13px; margin-top: 4px; }}

    /* Cards -- white on the grey page, borderless, rounded. */
    /* The card is a size container so the number can scale to the card's own
       width: full 30px in a 3-card row, smaller in Inventory Plan's 4-card row,
       where "฿588,091,558" at 30px used to break in the middle of the number.
       word-break normal lets text values wrap at spaces, never inside a number. */
    .kpi-card {{ background-color: {CARD}; border-radius: {CARD_RADIUS}; padding: 22px 24px; min-height: 150px;
        border: 1px solid {BORDER_SUBTLE}; box-shadow: 0 1px 2px rgba(22, 24, 29, 0.06);
        container-type: inline-size; }}
    .kpi-label {{ color: {TEXT_MUTED}; font-size: 13px; font-weight: 600; margin-bottom: 8px; }}
    .kpi-value {{ color: {TEXT_MAIN}; font-size: min(30px, 12.5cqi); font-weight: 700; line-height: 1.1;
        overflow-wrap: normal; word-break: normal; }}
    /* The design system's KpiCard delta: status tones, not the chart variance colours. */
    .kpi-delta-up {{ color: #0A6B7A; font-size: 13px; font-weight: 600; margin-top: 8px; }}
    .kpi-delta-down {{ color: #B42318; font-size: 13px; font-weight: 600; margin-top: 8px; }}
    /* A row of KPI cards shares one height: a card whose note wraps no longer stands
       taller than its neighbours. Only for columns holding just the card -- where a
       caption follows, stretching the card would push the captions out of line. */
    div[data-testid="stColumn"]:has(.kpi-card):not(:has([data-testid="stCaptionContainer"])) > div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] {{ flex: 1 1 auto; }}
    div[data-testid="stColumn"]:has(.kpi-card):not(:has([data-testid="stCaptionContainer"])) :is([data-testid="stMarkdown"], [data-testid="stMarkdown"] > div) {{ height: 100%; }}
    /* +1rem undoes Streamlit's -1rem bottom margin on Markdown, which otherwise leaves the
       stretched cards 1rem shorter than the tallest one. The target card gets the same
       margin so the Overview row measures the same way. */
    div[data-testid="stColumn"]:has(.kpi-card):not(:has([data-testid="stCaptionContainer"])) [data-testid="stMarkdownContainer"] {{ height: calc(100% + 1rem) !important; }}
    div[data-testid="stColumn"]:has(.kpi-card):not(:has([data-testid="stCaptionContainer"])) .kpi-card {{ min-height: max(150px, 100%); box-sizing: border-box; }}
    /* The comparison a KPI's arrow is measured against, or its exact value. */
    .kpi-note {{ color: {TEXT_MUTED}; font-size: 12px; font-weight: 500; margin-top: 4px; }}
    /* Digits of equal width, so numbers in a row of cards line up. */
    .kpi-value, .profile-value, .build-card-value {{ font-variant-numeric: tabular-nums; }}
    /* Sidebar build summary: what every page is showing. */
    .build-card {{ border-top: 3px solid {NAV_ACTIVE}; padding: 6px 14px 0 14px; margin-top: 8px; }}
    .build-card-label {{ color: {TEXT_MUTED}; font-size: 11px; font-weight: 700; letter-spacing: 0.6px;
        text-transform: uppercase; margin-top: 10px; }}
    .build-card-value {{ color: {TEXT_MAIN}; font-size: 13px; font-weight: 600; line-height: 1.35;
        overflow-wrap: anywhere; }}
    .build-card-flag {{ color: {ORANGE}; }}
    .build-card-note {{ color: {TEXT_MUTED}; font-size: 12px; line-height: 16px; margin-top: 2px; }}
    /* Info / success notes in the app's own navy and grey, not Streamlit's blue and
       green. Warnings and errors keep their colours -- those need attention. */
    div[data-testid="stAlertContentInfo"], div[data-testid="stAlertContentSuccess"] {{ color: {TEXT_MAIN}; }}
    div[data-testid="stAlert"]:has(div[data-testid="stAlertContentInfo"]) > div,
    div[data-testid="stAlert"]:has(div[data-testid="stAlertContentSuccess"]) > div {{
        background-color: {NAV_ACTIVE}; border: none; border-radius: 8px; }}
    .side-card {{ background-color: {CARD}; border-radius: {CARD_RADIUS};
        border: 1px solid {BORDER_SUBTLE}; box-shadow: 0 1px 2px rgba(22, 24, 29, 0.06);
        padding: 22px 24px; color: {TEXT_MAIN}; margin-bottom: 14px; }}
    .side-card-label {{ color: {TEXT_MUTED}; font-size: 11px; font-weight: 700; letter-spacing: 0.8px; margin-bottom: 10px; }}
    /* Compact profile card -- one fact (customer type, industry, category...).
       Same white fill, border and radius as the KPI cards, but auto height and
       a text-sized value: these hold words, not big numbers, and a long FA Sub
       Industry name must wrap rather than be clipped. min-height (not height:100%,
       which Streamlit columns do not stretch) keeps a row of cards level when one
       value wraps to a second line. */
    .profile-card {{ background-color: {CARD}; border-radius: {CARD_RADIUS};
        border: 1px solid {BORDER_SUBTLE};
        padding: 14px 18px; min-height: 88px; }}
    .profile-label {{ color: {TEXT_MUTED}; font-size: 11px; font-weight: 700; letter-spacing: 0.6px;
        text-transform: uppercase; margin-bottom: 4px; }}
    .profile-value {{ color: {TEXT_MAIN}; font-size: 16px; font-weight: 600; line-height: 1.3;
        overflow-wrap: anywhere; }}
    .profile-note {{ color: {TEXT_MUTED}; font-size: 12px; font-weight: 500; margin-top: 2px; }}
    /* Inset variant: the same tile INSIDE a white card (the per-model error row on
       Model Accuracy), where a white tile would be invisible. The accent value is
       orange -- the app's "this is the one we use" colour. */
    .profile-card-inset {{ background-color: {BG}; min-height: 74px; }}
    .profile-value-accent {{ color: {ORANGE}; }}
    /* st.container(border=True) card -- white fill, no visible border and the
       same radius as the KPI cards. Streamlit 1.63 renders a bordered container
       as a stVerticalBlock directly inside a stLayoutWrapper. The previous
       selector targeted the old stVerticalBlockBorderWrapper test id, which no
       longer exists -- so every chart card was silently left transparent on the
       grey page background. */
    /* Unbordered containers share this markup; the ones the app makes are keyed
       "plain_..." (the sidebar card slot, and the container inside each page fragment,
       whose unkeyable wrapper is recognised by that child) and stay transparent.
       Only the fragment bodies ("plain_..._detail") mark a wrapper: a real card whose
       first child is a plain_ container (Build Forecast's step bodies) stays white. */
    div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:not([class*="st-key-plain_"]):not(:has(> div[data-testid="stLayoutWrapper"] > [class*="st-key-plain_"]:where([class*="_detail"]))) {{
        background-color: {CARD} !important;
        border: 1px solid {BORDER_SUBTLE} !important;
        box-shadow: 0 1px 2px rgba(22, 24, 29, 0.06);
        border-radius: {CARD_RADIUS} !important;
        padding: 1.25rem 1.5rem !important;
    }}
    /* Every expander is the design system's Disclosure: a hairline above, a bold
       summary with +/- in place of the chevron, the body muted and indented. */
    div[data-testid="stExpander"] details {{
        background-color: transparent !important; border: none !important;
        border-top: 1px solid {BORDER_SUBTLE} !important; border-radius: 0 !important;
    }}
    div[data-testid="stExpander"] summary {{ padding: 12px 0 0 !important; }}
    div[data-testid="stExpander"] summary:hover {{ background: transparent !important; }}
    div[data-testid="stExpander"] summary p {{ font-weight: 600; }}
    div[data-testid="stExpander"] summary span:has(> :is([data-testid="stExpanderIcon"], [data-testid="stIconMaterial"])) {{ display: none; }}
    div[data-testid="stExpander"] summary::before {{ content: "+"; width: 16px; margin-right: 8px;
        text-align: center; font-weight: 600; color: {TEXT_MAIN}; }}
    div[data-testid="stExpander"] details[open] > summary::before {{ content: "−"; }}
    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {{ padding: 8px 0 4px 24px !important; }}
    /* The Overview FY-target card is a real container (it holds a number input),
       so it does not get .kpi-card's min-height from the HTML cards beside it.
       With a target set it holds a label, the input and two lines of gap text,
       which at the usual 150px ran into its bottom edge -- so this one KPI row is
       170px, the three HTML cards included (scoped with :has, so every other
       page's KPI row keeps 150px). st.container(key=...) renders the key as an
       "st-key-" class. */
    .st-key-ov_target_card [data-testid="stWidgetLabel"] p {{ color: {TEXT_MUTED};
        font-size: 13px; font-weight: 600; }}
    .target-gap-note {{ color: {TEXT_MUTED}; font-size: 12px; font-weight: 500; margin-top: 2px; }}

    /* --- Forecast Console: status pills, target bar, attention cards -----------
       Tones come from the design system's tokens (--status-*): teal for good, amber
       for warn, red for bad. They are deliberately NOT the app's GREEN/RED, which
       mark a variance being up or down, not a state needing attention. */
    .fc-pill {{ display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px;
        border-radius: 999px; font-size: 12px; line-height: 16px; font-weight: 600; white-space: nowrap; }}
    .fc-pill--good {{ background: #E0F0F3; color: #0A6B7A; }}
    .fc-pill--warn {{ background: #FBEED3; color: #7A4D00; }}
    .fc-pill--bad {{ background: #FBE3E0; color: #B42318; }}
    .fc-pill--neutral {{ background: {NAV_ACTIVE}; color: {TEXT_MUTED}; }}

    /* The board's identity line above a detail page's KPI row: the item or
       customer, its status pill and one muted line of facts. */
    .ident-row {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
        margin-bottom: 16px; }}
    .ident-name {{ font-size: 24px; line-height: 32px; font-weight: 600; color: {TEXT_MAIN};
        letter-spacing: -0.01em; }}
    .ident-meta {{ font-size: 14px; line-height: 20px; color: {TEXT_MUTED}; }}
    /* Model Accuracy, "Monthly miss by model": one small card per model, the chart's legend. */
    .miss-tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px;
        margin-bottom: 8px; }}
    .miss-tile {{ border: 1px solid {BORDER_SUBTLE}; border-radius: 8px; padding: 12px 16px; background: {CARD}; }}
    .miss-tile--on {{ border-color: {ORANGE}; }}
    .miss-name {{ display: flex; align-items: center; gap: 8px; font-size: 12px; line-height: 16px;
        font-weight: 600; color: {TEXT_MUTED}; }}
    .miss-sw {{ flex-shrink: 0; width: 18px; height: 3px; border-radius: 2px; }}
    .miss-val {{ font-size: 24px; line-height: 28px; font-weight: 600; color: {TEXT_MAIN}; margin-top: 8px;
        font-variant-numeric: tabular-nums; }}
    .miss-tile--on .miss-val {{ color: {ORANGE}; }}
    /* Customer Detail: each category's all-years total and share, the chart's legend. */
    /* No 1rem pull-up under the list, so the stretched chart beside it ends where it does. */
    [data-testid="stMarkdownContainer"]:has(> .cd-share) {{ margin-bottom: 0; }}
    .cd-share-h {{ font-size: 12px; line-height: 16px; font-weight: 600; color: {TEXT_MUTED}; margin-bottom: 4px; }}
    .cd-share-row {{ display: flex; align-items: center; gap: 8px; padding: 6px 0;
        border-top: 1px solid {BORDER_SUBTLE}; font-size: 13px; line-height: 18px; color: {TEXT_MAIN}; }}
    .cd-sw {{ flex-shrink: 0; width: 12px; height: 12px; border-radius: 2px; }}
    .cd-share-name {{ flex-grow: 1; min-width: 0; overflow-wrap: anywhere; }}
    .cd-share-val {{ font-weight: 600; font-variant-numeric: tabular-nums; white-space: nowrap; }}
    .cd-share-val em {{ font-style: normal; font-weight: 500; color: {TEXT_MUTED}; display: inline-block;
        min-width: 38px; text-align: right; }}

    /* Target bar: the forecast as a filled track, its p10-p90 range as a band on top
       of it, and the target as a dashed tick. The 40px/28px margins make room for the
       tick's label above the track. */
    .tgt-value {{ color: {TEXT_MAIN}; font-size: 36px; line-height: 40px; font-weight: 600;
        letter-spacing: -0.01em; font-variant-numeric: tabular-nums; }}
    .tgt-status {{ display: flex; align-items: baseline; gap: 10px; margin-top: 8px; flex-wrap: wrap; }}
    .tgt-status .kpi-note {{ margin-top: 0; }}
    .tgt-track {{ position: relative; height: 16px; margin: 40px 0 28px; background: {NAV_ACTIVE};
        border-radius: 4px; }}
    .tgt-fill {{ position: absolute; left: 0; top: 0; bottom: 0; background: {NAVY};
        border-radius: 4px 0 0 4px; }}
    .tgt-range {{ position: absolute; top: 4px; bottom: 4px; background: rgba(184, 84, 0, 0.15);
        border: 1px solid {ORANGE}; border-radius: 2px; }}
    .tgt-tick {{ position: absolute; top: -10px; bottom: -10px; width: 0;
        border-left: 2px dashed {TEXT_MAIN}; }}
    .tgt-tick span {{ position: absolute; top: -20px; transform: translateX(-50%); font-size: 12px;
        font-weight: 600; white-space: nowrap; color: {TEXT_MAIN}; }}

    /* Attention cards: pill, big count, title, body, then the button. space-between
       pins the button to the bottom of the card (the board's .fc-attn rule), so three
       cards of different text lengths still line their buttons up. */
    div[data-testid="stVerticalBlock"][class*="st-key-attn_card_"] {{ justify-content: space-between; }}
    /* Secondary button, as on the board: white with a hairline, not the navy of a
       primary action. Scoped to these cards so no other button changes. */
    div[class*="st-key-attn_card_"] .stButton button, .st-key-_ov_biz .stButton button {{
        background-color: {CARD} !important; color: {TEXT_MAIN} !important;
        border: 1px solid {BORDER_CONTROL} !important; width: auto !important; }}
    div[class*="st-key-attn_card_"] .stButton button:hover, .st-key-_ov_biz .stButton button:hover {{
        background-color: {NAV_ACTIVE} !important; color: {TEXT_MAIN} !important; }}
    /* The section's own link sits at the right of its heading row. */
    .st-key-_ov_biz .stButton {{ text-align: right; }}
    .attn-count {{ color: {TEXT_MAIN}; font-size: 24px; line-height: 28px; font-weight: 600;
        margin-top: 8px; font-variant-numeric: tabular-nums; }}
    .attn-title {{ color: {TEXT_MAIN}; font-size: 16px; line-height: 24px; font-weight: 600; }}
    .attn-body {{ color: {TEXT_MUTED}; font-size: 13px; line-height: 20px; margin-top: 8px; }}
    /* --- Build Forecast: the three-step strip and its step cards ---------------
       The strip sits on the page background as it does on the board; each step
       card repeats its numbered badge so a card still says which step it is when
       the strip has scrolled away. Teal = done, navy = the step you are on. */
    .fc-steps {{ display: flex; gap: 16px; margin: 0 0 36px; padding: 0; list-style: none; }}
    .fc-step {{ flex: 1 1 0; display: flex; align-items: center; gap: 12px;
        padding: 4px 4px 18px !important; margin: 0 !important;
        border-bottom: 3px solid transparent; }}
    .fc-step--on {{ border-bottom-color: {NAVY}; }}
    .fc-num {{ flex-shrink: 0; width: 32px; height: 32px; border-radius: 50%; display: flex;
        align-items: center; justify-content: center; font-size: 14px; font-weight: 600;
        background: {NAV_ACTIVE}; color: {TEXT_MUTED}; }}
    .fc-num--done {{ background: #0A6B7A; color: #FFFFFF; }}
    .fc-num--on {{ background: {NAVY}; color: #FFFFFF; }}
    .fc-step-name {{ display: block; font-size: 14px; font-weight: 600; color: {TEXT_MAIN}; }}
    .fc-step-sub {{ display: block; font-size: 12px; line-height: 16px; color: {TEXT_MUTED}; }}
    /* A step card's header: badge, title, one-line state, and its pill at the right. */
    .fc-hdr {{ display: flex; align-items: center; gap: 14px; margin-bottom: 14px; }}
    /* A section inside a step card. Streamlit renders ##### at 20px, the same
       size as the step title, which flattens the hierarchy -- the board sets
       these one level down at 16px. */
    .fc-sec {{ font-size: 16px; line-height: 24px; font-weight: 600; color: {TEXT_MAIN};
        margin: 8px 0 12px; }}
    .fc-hdr-title {{ font-size: 20px; line-height: 28px; font-weight: 600; color: {TEXT_MAIN}; }}
    .fc-hdr-title--sm {{ font-size: 16px; line-height: 24px; }}
    .fc-hdr-sub {{ font-size: 13px; line-height: 18px; color: {TEXT_MUTED}; }}
    /* Small uppercase group heading, inside a card (feature groups, master-file
       sections). Same treatment as .profile-label, which is tied to that tile. */
    .fc-grp {{ font-size: 11px; line-height: 16px; font-weight: 600; letter-spacing: 0.06em;
        text-transform: uppercase; color: {TEXT_MUTED}; margin: 20px 0 6px; }}
    /* A step's name and state are one line each on the board; the main column is
       narrower here, and "Inventory rules" was wrapping the strip out of line. */
    .fc-step-name, .fc-step-sub {{ white-space: nowrap; }}
    /* Fiscal-year tiles: the board draws them white with a hairline, not the grey
       inset the model tiles on Model Accuracy use. */
    div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"][class*="st-key-fy_tiles"][class*="st-key-fy_tiles"][class*="st-key-fy_tiles"][class*="st-key-fy_tiles"] {{
        padding: 0 !important; margin-bottom: 16px !important;
        background-color: transparent !important; border: 0 !important; box-shadow: none !important; }}
    .st-key-fy_tiles .profile-card-inset {{ background-color: {CARD};
        border: 1px solid {BORDER_SUBTLE}; }}
    /* One bordered row per model, navy once it is ticked, its one-line note inside
       at 12px and indented under the label -- the board's "Models to compare". */
    div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"][class*="st-key-mdlopt_"][class*="st-key-mdlopt_"][class*="st-key-mdlopt_"][class*="st-key-mdlopt_"] {{
        padding: 10px 16px !important; border: 1px solid {BORDER_SUBTLE} !important;
        border-radius: 8px !important; gap: 0 !important; }}
    div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"][class*="st-key-mdlopt_"][class*="st-key-mdlopt_"][class*="st-key-mdlopt_"][class*="st-key-mdlopt_"]:has(label[data-selected="true"]) {{
        border-color: {NAVY} !important; }}
    div[class*="st-key-mdlopt_"] [data-testid="stCaptionContainer"] {{ margin-left: 28px; }}
    div[class*="st-key-mdlopt_"] [data-testid="stCaptionContainer"] p {{ font-size: 12px;
        line-height: 16px; }}
    /* Select all / Deselect all are secondary buttons and Reset a ghost, each at its
       own width -- not three navy blocks stretched across the card. */
    div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"].st-key-feat_btns {{
        padding: 0 !important; }}
    .st-key-feat_btns .stButton button {{ background-color: {CARD} !important;
        color: {TEXT_MAIN} !important; border: 1px solid {BORDER_CONTROL} !important; }}
    .st-key-feat_btns .stButton button:hover {{ background-color: {NAV_ACTIVE} !important; }}
    .st-key-feat_btns .st-key-_feat_btn_reset button {{ border-color: transparent !important; }}
    /* Step 2's footer: back and next as the board draws them. They are jump links to
       the other step cards, which are all on the page at once here. */
    /* The open step's footer: a hairline, then a quiet Back at the left and the
       navy Next at the right. A shut step's Edit button gets the same quiet
       treatment, and neither stretches across the card, as on the board. */
    /* go_to()'s scroll-to-top script: runs, takes no space. */
    div[data-testid="stLayoutWrapper"]:has(> .st-key-plain_scroll_top) {{ display: none; }}
    .st-key-plain_step_nav {{ margin-top: 24px; padding: 18px 0 16px;
        border-top: 1px solid {BORDER_SUBTLE}; }}
    /* The block is a flex column whose children shrink to their content, so the
       Next button is pushed over by aligning the column, not its child. */
    [class*="st-key-plain_nav_right"] {{ align-items: flex-end; }}
    /* Two classes, not one: the page's own ".stButton button" rule comes later in
       this sheet, so a single-class selector loses to it and these stayed navy. */
    [class*="st-key-_bf_back"] .stButton button, [class*="st-key-_bf_edit"] .stButton button {{
        background-color: {CARD} !important; color: {TEXT_MAIN} !important;
        border: 1px solid {BORDER_CONTROL} !important; }}
    [class*="st-key-_bf_back"] .stButton button:hover, [class*="st-key-_bf_edit"] .stButton button:hover {{
        background-color: {NAV_ACTIVE} !important; color: {TEXT_MAIN} !important; }}
    /* Master files: one small bordered card per file inside the Data step. */
    div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"][class*="st-key-mastercard_"][class*="st-key-mastercard_"][class*="st-key-mastercard_"][class*="st-key-mastercard_"] {{
        padding: 16px !important; border: 1px solid {BORDER_SUBTLE} !important; box-shadow: none;
        border-radius: 8px !important; gap: 12px !important; }}
    /* Both master cards keep their buttons on the card's bottom edge, level with each other. */
    [data-testid="stLayoutWrapper"]:has(> [class*="st-key-plain_mst_btns_"]) {{ margin-top: auto; }}
    /* Forecast chart legend (chart_legend), as the design system's ForecastChart draws it. */
    .fc-legend {{ display: flex; gap: 24px; flex-wrap: wrap; font-size: 13px; line-height: 20px; color: {TEXT_MAIN};
                  margin-bottom: 8px; padding-left: 86px; }}  /* 86px = the y axis (fixed in trend_chart) */
    .fc-legend span {{ display: inline-flex; align-items: center; gap: 8px; }}
    .fc-legend svg {{ width: 28px; height: 10px; }}
    .fc-mst-head {{ display: flex; align-items: center; gap: 8px; }}
    .fc-mst-title {{ flex-grow: 1; font-size: 15px; line-height: 20px; font-weight: 600; color: {TEXT_MAIN}; }}
    .fc-mst-src {{ font-size: 13px; line-height: 18px; color: {TEXT_MUTED}; margin-top: 12px; }}
    .fc-mst-what {{ font-size: 13px; line-height: 18px; color: {TEXT_MAIN}; margin-top: 12px; }}
    /* Streamlit pulls a markdown block up by 1rem, which ran the text into the checkbox. */
    [class*="st-key-mastercard_"] [data-testid="stMarkdownContainer"]:has(.fc-mst-head, .fc-mst-note) {{
        margin-bottom: 0; }}
    [class*="st-key-mastercard_"] [data-testid="stCheckbox"] p {{ font-size: 14px; line-height: 20px; }}
    /* The checkbox's note, under its label as on the board. */
    .fc-mst-note {{ font-size: 12px; line-height: 16px; color: {TEXT_MUTED}; margin: -12px 0 0 24px; }}
    /* The design system's small buttons: Replace file secondary, Column names ghost. */
    div[class*="st-key-mastercard_"] [data-testid="stPopover"] button {{ min-height: 30px; height: 30px;
        padding: 0 12px; border-radius: 8px; }}
    div[class*="st-key-mastercard_"] [data-testid="stPopover"] button p {{ font-size: 13px; font-weight: 600; }}
    div[class*="st-key-mastercard_"] [data-testid="stPopover"] button[data-testid="stBaseButton-secondary"] {{
        background: {CARD}; border: 1px solid {BORDER_CONTROL}; color: {TEXT_MAIN}; }}
    div[class*="st-key-mastercard_"] [data-testid="stPopover"] button:hover {{ background: {NAV_ACTIVE}; color: {TEXT_MAIN}; }}
    /* The footer's right-hand note on the Run step. */
    .fc-nav-note {{ font-size: 13px; line-height: 18px; color: {TEXT_MUTED}; text-align: right; }}
    /* Inventory Plan, "Stocking rules": one small bordered card per rule with
       its input kept to a short width. */
    div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"][class*="st-key-rulecard_"][class*="st-key-rulecard_"][class*="st-key-rulecard_"][class*="st-key-rulecard_"] {{
        padding: 16px !important; border: 1px solid {BORDER_SUBTLE} !important;
        border-radius: 8px !important; gap: 6px !important; }}
    div[class*="st-key-rulecard_"] [data-testid="stNumberInput"] {{ max-width: 200px; }}
    div[class*="st-key-rulecard_"] [data-testid="stWidgetLabel"] p {{ font-size: 13px; font-weight: 600; }}
    div[class*="st-key-rulecard_"] [data-testid="stCaptionContainer"] p {{ font-size: 12px; line-height: 16px; }}
    /* Run step, "Forecast output options": one bordered row per option, navy once
       ticked, the title at 15px, the explanation under it and a pill at the right. */
    /* A horizontal container is a stHorizontalBlock, which the card rule never matches. */
    div[data-testid="stHorizontalBlock"][class*="st-key-runopt_"] {{
        padding: 16px; border: 1px solid {BORDER_SUBTLE}; border-radius: 8px; }}
    div[data-testid="stHorizontalBlock"][class*="st-key-runopt_"]:has(label[data-selected="true"]) {{
        border-color: {NAVY}; }}
    div[class*="st-key-runopt_"] [data-testid="stCheckbox"] label p {{ font-size: 15px; font-weight: 600; }}
    div[class*="st-key-runopt_"] [data-testid="stCaptionContainer"] {{ margin-left: 28px; }}
    div[class*="st-key-runopt_"] [data-testid="stCaptionContainer"] p {{ font-size: 13px; line-height: 18px; }}
    /* "Review your settings": label, value and a ghost Edit per row, hairlines between. */
    .st-key-plain_review {{ gap: 0 !important; border-bottom: 1px solid {BORDER_SUBTLE}; }}
    [class*="st-key-plain_rv_"] {{ border-top: 1px solid {BORDER_SUBTLE}; padding: 5px 0; gap: 12px !important;
        min-height: 41px; }}
    .fc-rv-k {{ display: block; width: 120px; font-size: 14px; line-height: 20px; color: {TEXT_MUTED}; }}
    .fc-rv-v {{ font-size: 14px; line-height: 20px; font-weight: 600; color: {TEXT_MAIN}; }}
    .fc-rv-was {{ font-weight: 500; color: {TEXT_MUTED}; }}
    [class*="st-key-_bf_rv_"] .stButton button {{ background-color: transparent !important;
        color: {TEXT_MAIN} !important; min-height: 30px; height: 30px; padding: 0 12px;
        border-radius: 8px !important; }}
    [class*="st-key-_bf_rv_"] .stButton button p {{ font-size: 13px; }}
    [class*="st-key-_bf_rv_"] .stButton button:hover {{ background-color: {NAV_ACTIVE} !important;
        color: {TEXT_MAIN} !important; }}
    /* Nobody enters a sales target by stepping 1,000 units or 1M baht at a time,
       and the two step buttons took 64px from a narrow input: a 10-digit baht
       figure was clipped to "2600000C" at a 1440px-wide window. */
    .st-key-ov_target_card [data-testid="stNumberInputStepDown"],
    .st-key-ov_target_card [data-testid="stNumberInputStepUp"] {{ display: none; }}
    /* Streamlit's "Press Enter to apply" hint is drawn INSIDE the input and sat on
       top of the typed number in this narrow box. The empty-state caption below
       says the same thing, and Tab / clicking away also applies the value. */
    .st-key-ov_target_card [data-testid="InputInstructions"] {{ display: none; }}
    /* The design system's small input and button under the bar: 36px tall, a control border. */
    .st-key-ov_target_card [data-testid="stNumberInputContainer"] {{ height: 36px; }}
    .st-key-_ov_set_target button {{ min-height: 36px; height: 36px; padding: 0 16px; }}
    /* Streamlit's chart theme writes the page background (config.toml
       backgroundColor, #EEEEEE) as an INLINE style on every Vega <svg>. An
       !important stylesheet rule beats an inline style, so paint it the card
       white rather than leaving a grey rectangle inside a white card. */
    [data-testid="stVegaLiteChart"] svg.marks {{ background-color: {CARD} !important; }}
    /* Dividers -- Streamlit's default hr carries a big vertical margin that reads
       as dead space (e.g. above the "experimental features" toggle). Tighten it. */
    hr, div[data-testid="stDivider"] {{ margin: 0.4rem 0 !important; }}
    /* Except the sidebar's one rule, between the build summary and the build notes:
       tightened to 0.4rem the notes read as the card's last line. !important because
       the rule above has it too -- without it this is silently overridden. */
    section[data-testid="stSidebar"] hr {{ margin: 22px 0 18px !important;
        border: 0 !important; border-top: 3px solid {NAV_ACTIVE} !important; background: none !important; }}
    h1, h2, h3, h4, h5, h6 {{ color: {TEXT_MAIN} !important; }}
    /* Streamlit gives h4/h5 no bottom margin, so a section or card title sat
       directly on its first line of content. One rule, so every page's "####
       page section" and "##### card title" get the same breathing room. */
    [data-testid="stMarkdownContainer"] :is(h4, h5) {{ margin-bottom: 12px !important; }}
    /* Buttons and download buttons -- dark navy instead of orange, guaranteed
       contrast regardless of any theme quirk (this has been a repeated
       trouble spot in earlier passes). */
    .stButton button, .stDownloadButton button {{
        background-color: {NAVY} !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }}
    /* Download and run buttons (keyed act_...): one padding and height, and a long
       label wraps inside the button instead of spilling out of it. */
    [data-testid="stMain"] :is(.stDownloadButton, [class*="st-key-act_"]) button {{
        min-height: 40px; height: auto !important; padding: 8px 16px !important; }}
    [data-testid="stMain"] :is(.stDownloadButton, [class*="st-key-act_"]) button p {{
        white-space: normal !important; overflow-wrap: anywhere; text-align: center; }}
    .stButton button:hover, .stDownloadButton button:hover {{
        background-color: {NAVY_DARK} !important;
        color: #FFFFFF !important;
    }}
    /* The sidebar's Glossary button reads as one more menu row, not an action. */
    section[data-testid="stSidebar"] .stButton button {{
        background-color: transparent !important; color: {TEXT_MAIN} !important; font-weight: 500 !important;
    }}
    /* Icon and label lined up with the nav rows above (measured in the browser). */
    section[data-testid="stSidebar"] .stButton button > div {{ width: 100%; justify-content: flex-start; }}
    section[data-testid="stSidebar"] .stButton button p {{ font-size: 16px; }}
    section[data-testid="stSidebar"] .stButton button [data-testid="stIconMaterial"] {{ font-size: 22px; }}
    section[data-testid="stSidebar"] .stButton button span:has(> [data-testid="stIconMaterial"]) {{ margin-right: 14px; }}
    section[data-testid="stSidebar"] .stButton button:hover {{ background-color: #F5F5F5 !important; }}
    /* SegmentedControl (design system): a sunken track with a control-grey
       outline, 28px segments, the selected one navy with white text. Every view
       switch (Units / Revenue, Monthly / Quarterly, grain) is an st.segmented_control. */
    [data-testid="stButtonGroup"] > [role="radiogroup"]:has(> button[data-variant="segmented_control"]) {{
        display: inline-flex; flex-wrap: nowrap; gap: 2px; padding: 2px; width: auto;
        background: {NAV_ACTIVE}; border: 1px solid #737884; border-radius: 8px; }}
    button[data-variant="segmented_control"] {{ height: 28px; min-height: 28px; padding: 0 12px;
        margin: 0 !important; border: 0 !important; border-radius: 6px !important;
        background: transparent !important; color: {TEXT_MAIN} !important; box-shadow: none !important; }}
    button[data-variant="segmented_control"] p {{ font-size: 14px; font-weight: 500; }}
    button[data-variant="segmented_control"]:hover {{ background: {CARD} !important; }}
    button[data-variant="segmented_control"][aria-checked="true"] {{ background: {NAVY} !important; }}
    button[data-variant="segmented_control"][aria-checked="true"] p {{ color: #FFFFFF !important; font-weight: 600; }}
    button[data-variant="segmented_control"]:focus-visible {{ outline: 2px solid #1A56DB; outline-offset: 2px; }}
    /* Redesign: the current page is marked by its pill and bold text (no red square: the user's call). */
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {{ position: relative; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label p {{ font-weight: 400; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-selected="true"] p {{ font-weight: 600; }}
    /* Glossary is a pop-up, not a page: a footer item under a hairline, outside the nav groups. */
    section[data-testid="stSidebar"] .st-key-glossary_btn {{ border-top: 3px solid {NAV_ACTIVE}; padding-top: 16px; margin-top: 8px; }}
    /* Redesign: the board's nav group labels (Plan / Explore / Model), drawn above each
       group's first row so the radio -- and nav_page -- stay one widget. */
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:is({", ".join(f":nth-of-type({n})" for n in NAV_GROUP_STARTS)}) {{ margin-top: 34px; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(1) {{ margin-top: 16px; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label::after {{
        position: absolute; left: 14px; bottom: 100%; margin-bottom: 2px; font-size: 11px; line-height: 16px;
        font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: {TEXT_MUTED}; }}
    {" ".join(f'section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type({n})::after {{ content: "{g}"; }}' for n, g in NAV_GROUP_STARTS.items())}
    /* Glossary pop-up (the board's Glossary page): grey body so its white cards read as on a page,
       term tables as grids, jump links as the board's "On this page" rows. */
    div[role="dialog"]:has(.gl-table) {{ background: {BG}; }}
    /* Title and subtitle as page_title() draws them on the pages: 28/34 semibold, subtitle 4px below. */
    [data-testid="stDialog"]:has(.gl-table) h2, [data-testid="stDialog"]:has(.gl-table) h2 p {{ font-size: 28px; line-height: 34px; }}
    [data-testid="stDialog"] .gl-subtitle {{ margin-top: -28px; }}
    .gl-title {{ font-size: 20px; line-height: 28px; font-weight: 600; color: {TEXT_MAIN}; margin-bottom: 16px; }}
    .gl-table {{ font-size: 14px; line-height: 20px; color: {TEXT_MAIN}; }}
    .gl-row {{ display: grid; grid-template-columns: var(--gl-cols); column-gap: 24px; padding: 12px 0;
        border-top: 1px solid {BORDER_SUBTLE}; }}
    .gl-row.gl-head {{ padding: 0 0 8px; border-top: 0; font-size: 11px; line-height: 16px; font-weight: 600;
        letter-spacing: .06em; text-transform: uppercase; color: {TEXT_MUTED}; }}
    .gl-row > div:nth-child(1) {{ font-weight: 600; }}
    .gl-row > div:nth-child(3) {{ color: {TEXT_MUTED}; }}
    .gl-head > div {{ font-weight: 600 !important; }}
    .gl-table code {{ font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12px; color: {TEXT_MAIN};
        background: {NAV_ACTIVE}; padding: 1px 4px; border-radius: 4px; }}
    .gl-kicker {{ font-size: 11px; line-height: 16px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
        color: {TEXT_MUTED}; padding: 0 12px 4px; }}
    a.gl-link {{ display: flex; justify-content: space-between; gap: 8px; padding: 8px 12px; border-radius: 8px;
        font-size: 14px; color: {TEXT_MAIN} !important; text-decoration: none !important; }}
    a.gl-link:hover {{ background: {NAV_ACTIVE}; }}
    a.gl-link span:last-child {{ color: {TEXT_MUTED}; }}
</style>
""", unsafe_allow_html=True)


def generate_fake_raw_data():
    """Synthetic demo data shaped like the real MELFT export (10 columns,
    no Revenue column -- Revenue gets computed from Base Price x Quantity
    just like it will for the real file). Each item has a FIXED set of
    active calendar months (consistent every year) rather than pure random
    noise, so sparsity and backtest numbers behave sensibly."""
    rng = np.random.default_rng(42)
    n_products = 60
    products = [f"ITEM-{i:04d}" for i in range(1, n_products + 1)]
    item_codes = [f"CODE-{i:04d}" for i in range(1, n_products + 1)]
    statuses = ["Completed"] * 9 + ["Cancelled"]  # ~10% cancelled, like real messy data

    item_active_months = {}
    for p_idx, prod in enumerate(products):
        is_sparse = p_idx % 4 != 0
        if is_sparse:
            n_active = rng.integers(1, 4)
            item_active_months[prod] = set(rng.choice(range(1, 13), size=n_active, replace=False))
        else:
            item_active_months[prod] = set(range(1, 13))

    rows = []
    so_id = 1
    for year in [2021, 2022, 2023, 2024]:
        growth = {2021: 0.95, 2022: 1.0, 2023: 1.08, 2024: 1.15}[year]
        for month in range(1, 13):
            for p_idx, prod in enumerate(products):
                if month not in item_active_months[prod]:
                    continue
                base = 10 + (p_idx % 10) * 5
                units = max(0, int(rng.normal(base * growth, base * 0.15)))
                price = round((30 + (p_idx % 10) * 15) * rng.uniform(0.9, 1.1), 2)
                rows.append([
                    f"SO{so_id:07d}", pd.Timestamp(year, month, min(28, rng.integers(1, 28))),
                    f"SON-{so_id:06d}", rng.choice(statuses), f"CUST-{p_idx % 15:03d}", f"Company {p_idx % 15}",
                    prod, item_codes[p_idx], price, units,
                ])
                so_id += 1
    return pd.DataFrame(rows, columns=[
        "SO Internal ID", DEFAULT_COL_DATE, "Sales Order Number", DEFAULT_COL_STATUS,
        "Customer Internal ID", "Company Name", DEFAULT_COL_PRODUCT, DEFAULT_COL_ITEM_CODE,
        DEFAULT_COL_PRICE, DEFAULT_COL_UNITS,
    ])


@st.cache_data(show_spinner=False)
def read_uploaded_file(file_bytes, filename, sheet_name):
    """Reads the uploaded file exactly ONCE per (file, sheet_name) combo --
    cached so peeking at years/statuses and building the forecast don't
    each independently re-parse a large file from scratch.

    CSV is dramatically faster to parse than .xlsx at this row count (no
    engine reads formatting/styles) -- if you're hitting slow load times,
    exporting your cleaned data as .csv from Excel (File -> Save As -> CSV)
    is the single biggest speed win available.

    For .xlsx, tries the much faster 'calamine' engine first
    (pip install python-calamine), and silently falls back to the standard
    engine if it isn't installed.

    Column headers are stripped of leading/trailing whitespace on load --
    a stray space in a header (e.g. ' Item Internal ID' from an Excel
    export) is invisible in the file but breaks an exact-name match against
    whatever's typed in the column settings. Stripping here means that class of bug
    can't happen regardless of which file (sales data, item master) it's in."""
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        try:
            df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, engine="calamine")
        except (ImportError, ValueError):
            df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    return df


@st.cache_data(show_spinner=False)
def peek_file_settings(file_bytes, filename, sheet_name, col_date, col_status, fiscal_start_month):
    """Computes the fiscal years and status options available in the file,
    ONCE per (file, column settings) combination. This used to re-run the
    date parsing and fiscal-year computation on EVERY widget interaction
    (Streamlit reruns the whole script top-to-bottom on every click), which
    was a real source of the 'still slow' complaint even after switching to
    CSV -- caching it here means it only runs when the file or these
    settings actually change. Returns (sorted_fiscal_years, status_options,
    n_rows_with_bad_dates, n_rows, column_names)."""
    peek = read_uploaded_file(file_bytes, filename, sheet_name)

    if col_date not in peek.columns:
        raise ValueError(f"Date column '{col_date}' not found. Columns in your file: {list(peek.columns)}")

    status_options = []
    if col_status and col_status in peek.columns:
        status_options = sorted(peek[col_status].dropna().unique().tolist())

    cleaned, n_dropped = clean_dates(peek, col_date)
    fy, _ = to_fiscal(cleaned[col_date], fiscal_start_month)
    years = sorted(set(int(y) for y in fy))  # NaNs are already excluded by clean_dates

    return years, status_options, n_dropped, len(peek), list(peek.columns)


@st.cache_data(show_spinner=False)
def build_everything(file_bytes, filename, sheet_name, col_product, col_item_code, col_category, col_units, col_price, col_date,
                      col_status, valid_statuses_tuple, fiscal_start_month, train_years_tuple, validation_year, forecast_year,
                      model_types_tuple, final_model_choice, col_customer,
                      item_master_bytes, item_master_filename, item_master_join_col, item_master_cat1_col, item_master_cat2_col,
                      item_master_status_col="Production Status", item_master_price_col="List price",
                      customer_master_bytes=None, customer_master_filename=None, customer_master_join_col="Customer Internal ID",
                      customer_master_type_col="Customer Type", customer_master_industry_col="FA Sub Industries",
                      exclude_discontinued=True, feature_toggles=None,
                      add_prediction_intervals=True, apply_bias_corr=False):
    train_years = list(train_years_tuple)
    valid_statuses = list(valid_statuses_tuple) if valid_statuses_tuple else None
    model_types = list(model_types_tuple)

    # feature_toggles: a set of feature names the user left ENABLED. None
    # means "use everything" (the default). `_use(name)` is the gate every
    # optional feature checks before it's added to the model. The UI toggles
    # the two calendar columns together under the single key "seasonality".
    _enabled = set(feature_toggles) if feature_toggles is not None else None
    if _enabled is not None and "seasonality" in _enabled:
        _enabled |= {"month_sin", "month_cos"}
    def _use(name):
        return _enabled is None or name in _enabled

    _report_stage("Reading and cleaning the sales data")
    if file_bytes is None:
        raw = generate_fake_raw_data()
        n_dropped_dates = 0
    else:
        raw = read_uploaded_file(file_bytes, filename, sheet_name).copy()

        required = {"Product ID column": col_product, "Units column": col_units, "Base Price column": col_price, "Date column": col_date}
        optional = {"Item Code column": col_item_code, "Category column": col_category, "Order Status column": col_status, "Customer ID column": col_customer}
        missing_required = [f"{label} ('{col}')" for label, col in required.items() if col and col not in raw.columns]
        missing_optional = [f"{label} ('{col}')" for label, col in optional.items() if col and col not in raw.columns]

        if missing_required:
            raise ValueError(
                f"These columns aren't in your uploaded file: {'; '.join(missing_required)}.\n\n"
                f"Columns actually found in your file: {list(raw.columns)}\n\n"
                f"Check the 'Column settings' section on the Build Forecast page for typos, extra spaces, or renamed columns."
            )
        if missing_optional:
            raise ValueError(
                f"These optional columns are set but not found in your file: {'; '.join(missing_optional)}.\n\n"
                f"Columns actually found in your file: {list(raw.columns)}\n\n"
                f"Either fix the name in 'Column settings', or clear the field if you don't have that column."
            )

        raw, n_dropped_dates = clean_dates(raw, col_date)

    # Product IDs are cast to string right here, unconditionally, so every
    # downstream piece (dataframes, dropdown options, session-state
    # selections) is consistently typed. Without this, a numeric ID column
    # (e.g. plain integers like 6199) can collide with leftover string
    # selections from a previous run (e.g. demo data's "ITEM-0001") and
    # crash sorting with a numpy int/str comparison error.
    raw[col_product] = raw[col_product].astype(str)
    raw, n_duplicate_lines = drop_duplicate_lines(raw)
    n_unpriced_items = 0

    raw = filter_by_status(raw, col_status, valid_statuses)
    raw = compute_revenue(raw, col_price, col_units, revenue_col="Revenue")

    # --- Optional: join Category 1/2 from a separate item master file ---
    category_cols = []
    passthrough = {}
    if col_item_code:
        passthrough["Item Code"] = col_item_code
    if col_category:
        passthrough["Category"] = col_category
        category_cols.append("Category")

    if item_master_bytes is not None:
        item_master_sheet = 0 if not item_master_filename.lower().endswith(".csv") else None
        item_master = read_uploaded_file(item_master_bytes, item_master_filename, item_master_sheet)
        cat_map = {}
        if item_master_cat1_col and item_master_cat1_col in item_master.columns:
            cat_map["Category1"] = item_master_cat1_col
        if item_master_cat2_col and item_master_cat2_col in item_master.columns:
            cat_map["Category2"] = item_master_cat2_col
        # Non-category attributes from the same file: Production Status
        # (active/discontinued -- a filter AND a feature) and List price
        # (official THB price -- a demand feature and the basis for the
        # revenue forecast and the inventory value calculations).
        attr_map = dict(cat_map)
        if item_master_status_col and item_master_status_col in item_master.columns:
            attr_map["ProductionStatus"] = item_master_status_col
        if item_master_price_col and item_master_price_col in item_master.columns:
            attr_map["ListPrice"] = item_master_price_col
        if attr_map:
            if item_master_join_col not in item_master.columns:
                raise ValueError(
                    f"'{item_master_join_col}' isn't a column in your item master file.\n\n"
                    f"Columns found: {list(item_master.columns)}"
                )
            raw = merge_item_master(raw, col_product, item_master, item_master_join_col, attr_map)
            for out_name in cat_map:
                passthrough[out_name] = out_name
                category_cols.append(out_name)
            if "ProductionStatus" in attr_map:
                # passthrough only -- used for the obsolescence flags and
                # display, NOT as a model feature (the ablation showed it adds
                # nothing to accuracy, and it's 3 extra one-hot columns).
                passthrough["ProductionStatus"] = "ProductionStatus"
            if "ListPrice" in attr_map:
                raw, n_unpriced_items = drop_unpriced_items(raw, col_product, item_master,
                                                            item_master_join_col, item_master_price_col)
                raw["ListPrice"] = pd.to_numeric(raw["ListPrice"], errors="coerce")
                passthrough["ListPrice"] = "ListPrice"

    df = aggregate_to_fiscal_monthly(raw, col_product, col_units, "Revenue", col_date, fiscal_start_month, passthrough_cols=passthrough or None)

    # --- Customer x category monthly history (for the Customer Detail page) --
    # Descriptive only -- includes every product, discontinued or not.
    cust_cat_history = None
    _catcols = [c for c in ("Category", "Category1", "Category2") if c in raw.columns]
    if col_customer and col_customer in raw.columns and _catcols:
        _cc = raw[[col_customer, col_units, "Revenue", col_date] + _catcols].copy()
        _fy, _fm = to_fiscal(_cc[col_date], fiscal_start_month)
        _cc["FiscalYear"] = np.asarray(_fy)
        _cc["FiscalMonth"] = np.asarray(_fm)
        _cc[col_customer] = _cc[col_customer].astype(str).str.strip()
        for c in _catcols:
            _cc[c] = _cc[c].astype(str)
        cust_cat_history = _cc.groupby([col_customer] + _catcols + ["FiscalYear", "FiscalMonth"], as_index=False).agg(
            Units=(col_units, "sum"), Revenue=("Revenue", "sum"))
        cust_name_map = raw.assign(**{col_customer: raw[col_customer].astype(str).str.strip()}) \
            .drop_duplicates(col_customer).set_index(col_customer).get("Company Name")
        cust_name_map = cust_name_map.to_dict() if cust_name_map is not None else {}
    else:
        cust_name_map = {}

    # --- Discontinued products: out of the forward forecast only ------------
    # The data dictionary says "Discontinue" items should generally be
    # excluded from forecasting -- you won't reorder them. But that status is
    # TODAY's: many items were discontinued because their demand died. Dropping
    # them before the backtests scored every past year on survivors only
    # (survivorship look-ahead: FY2024 WAPE read ~2 pts better than it was).
    # So training, every backtest and the fill-rate replay keep all items, and
    # only the FY forecast drops them (below) -- status IS known at that point.
    # They still show up on the obsolescence page (status from the item master).
    # "To Be Advised" items are kept -- that status is ambiguous.
    df_full = df.copy()   # every item (incl. discontinued), for the obsolescence page
    _discontinued = set()
    if exclude_discontinued and "ProductionStatus" in df.columns:
        _status = df["ProductionStatus"].astype(str).str.strip().str.lower()
        _discontinued = set(df.loc[_status == "discontinue", col_product])
    n_discontinued_excluded = len(_discontinued)

    # --- List price as a numeric demand feature (from the item master) ---
    # Kept on df regardless (used for revenue + inventory value); only added
    # as a *model feature* when the user leaves its toggle on.
    extra_features = []
    # Features the user ticked that this data genuinely cannot supply (no
    # Customer ID column, no customer master, no item master). Recorded and
    # shown in the sidebar rather than silently dropped -- the old single
    # "experimental" checkbox quietly delivered 3 features instead of 7
    # whenever the customer master was missing.
    unavailable_features = []
    if "ListPrice" in df.columns:
        df["ListPrice"] = pd.to_numeric(df["ListPrice"], errors="coerce").fillna(0)
        if _use("ListPrice"):
            extra_features.append("ListPrice")
    elif _use("ListPrice"):
        unavailable_features.append("ListPrice")

    # --- Optional customer-master maps (Customer Type + FA Sub Industry) ---
    cust_type_map = cust_industry_map = None
    if customer_master_bytes is not None and col_customer and col_customer in raw.columns:
        cm_sheet = 0 if not str(customer_master_filename).lower().endswith(".csv") else None
        cmaster = read_uploaded_file(customer_master_bytes, customer_master_filename, cm_sheet)
        if customer_master_join_col in cmaster.columns:
            cmaster = cmaster.copy()
            cmaster[customer_master_join_col] = cmaster[customer_master_join_col].astype(str).str.strip()
            raw[col_customer] = raw[col_customer].astype(str).str.strip()
            key = cmaster.drop_duplicates(subset=[customer_master_join_col]).set_index(customer_master_join_col)
            if customer_master_type_col in cmaster.columns:
                cust_type_map = key[customer_master_type_col].to_dict()
            if customer_master_industry_col in cmaster.columns:
                cust_industry_map = key[customer_master_industry_col].to_dict()

    # --- Customer-behaviour features (all need a Customer ID column) -------
    # The ablation found ONE customer feature carries the signal:
    # distinct_customers_trailing_12m. Reorder cadence (3 cols) and customer
    # mix (4 cols) were flat on the headline metric, so their toggles default
    # OFF -- but they are plain _use()-gated toggles, and they are NOT nested
    # under distinct_customers_trailing_12m any more. (They used to be: with
    # that one unrelated toggle off, all seven silently disappeared.)
    #
    # Each block is ONE expensive computation, so it runs if any of its columns
    # is wanted. The merge is then subset to exactly the wanted columns:
    # feature_ablation() on Model Accuracy infers its steps from df.columns, so
    # a column left on df that the model never saw would make that table report
    # a feature set this build did not actually use.
    _have_customer = bool(col_customer) and col_customer in raw.columns
    _keys = [col_product, "FiscalYear", "FiscalMonth"]
    _report_stage("Building customer features")

    if _have_customer and _use("distinct_customers_trailing_12m"):
        cust_trailing = compute_distinct_customers_trailing(raw, col_product, col_customer, col_date, fiscal_start_month)
        df = df.merge(cust_trailing, on=_keys, how="left")
        df["distinct_customers_trailing_12m"] = df["distinct_customers_trailing_12m"].fillna(0)
        extra_features.append("distinct_customers_trailing_12m")
    elif _use("distinct_customers_trailing_12m"):
        unavailable_features.append("distinct_customers_trailing_12m")

    _cadence_on = [c for c in REORDER_CADENCE_COLS if _use(c)]
    if _cadence_on and _have_customer:
        cadence = compute_reorder_cadence_feature(raw, col_product, col_customer, col_date, fiscal_start_month)
        df = df.merge(cadence[_keys + _cadence_on], on=_keys, how="left")
        for c in _cadence_on:
            df[c] = df[c].fillna(0)
        extra_features.extend(_cadence_on)
    elif _cadence_on:
        unavailable_features.extend(_cadence_on)

    # The mix features also need the customer master's type / industry maps.
    # compute_customer_mix_features tolerates empty maps, but would then emit
    # four all-zero constant columns -- hence this guard.
    _mix_on = [c for c in CUSTOMER_MIX_COLS if _use(c)]
    if _mix_on and _have_customer and (cust_type_map or cust_industry_map):
        mix = compute_customer_mix_features(raw, col_product, col_customer, col_date, col_units, fiscal_start_month,
                                            customer_type_map=cust_type_map, industry_map=cust_industry_map)
        df = df.merge(mix[_keys + _mix_on], on=_keys, how="left")
        for c in _mix_on:
            df[c] = df[c].fillna(0)
        extra_features.extend(_mix_on)
    elif _mix_on:
        unavailable_features.extend(_mix_on)

    available_years = set(df["FiscalYear"].unique())
    missing = [y for y in train_years if y not in available_years] + ([validation_year] if validation_year not in available_years else [])
    if missing:
        raise ValueError(f"Your data doesn't cover fiscal year(s): {sorted(set(missing))}. Fiscal years found: {sorted(available_years)}")

    # Product category is a feature too -- drop it if the user toggled it off.
    category_flag = (category_cols or None) if _use("Category") else None
    if _use("Category") and not category_cols:
        unavailable_features.append("Category")

    # The time-series base features the user left on (lags / rolling means /
    # calendar / recency / sparsity). None of these are ever missing from
    # the frame -- add_features computes them all -- so the toggle just
    # controls which ones the model is allowed to use.
    base_features = [f for f in FEATURE_BASE if _use(f)]
    if not base_features and not extra_features and not category_flag:
        _msg = ("Every feature is turned off -- the model needs at least one. "
                "Re-enable some in Build Forecast -> Feature set.")
        if unavailable_features:
            _msg = ("The only feature(s) selected -- " + ", ".join(sorted(set(unavailable_features)))
                    + " -- aren't available in this data (they need a Customer ID column, the customer "
                      "master, or the item master). " + _msg)
        raise ValueError(_msg)

    _report_stage(f"Backtesting {len(model_types)} model{'s' if len(model_types) != 1 else ''} "
                  f"on FY{int(validation_year)} (the longest step)")
    res_df, metrics, feature_cols, trained_models, skipped, (test_X, test_y) = run_backtest(
        df, col_product, category_flag, train_years, validation_year, fiscal_start_month,
        model_types, extra_features=extra_features, feature_base=base_features)

    if metrics is None or not trained_models:
        raise ValueError("Not enough history to train any model with the current train/validation year settings.")

    # Pick which model actually generates the FY forecast: either the
    # user's explicit choice, or (if "Auto (best WAPE)") whichever trained
    # model had the lowest WAPE in the backtest above. Only genuinely trained
    # feature models are eligible -- pure-baseline columns (Baseline, Croston)
    # appear in the metrics table for comparison but can't drive the forecast.
    model_cols = [c for c in metrics.columns if c in trained_models]
    if final_model_choice == "Auto (best WAPE)" or final_model_choice not in model_cols:
        wape_row = metrics[metrics["Metric"] == "WAPE (volume-weighted %)"].iloc[0]
        chosen_model_name = wape_row[model_cols].astype(float).idxmin()
    else:
        chosen_model_name = final_model_choice

    # Feature importance for the chosen model type, computed from the
    # BACKTEST model (trained on train_years only) rather than the final
    # model below (which is retrained on everything, leaving no genuinely
    # held-out data to evaluate importance against).
    importance_df, importance_method = compute_feature_importance(trained_models[chosen_model_name], feature_cols, test_X, test_y)

    # Richer, decision-relevant accuracy report (MASE, forecast bias,
    # WAPE at category/quarter/total grain, active-SKU-only, sparsity tiers).
    # Computed from the same backtest split, on a sparsity-tagged copy that
    # uses train_years only (leakage-free), matching run_backtest above.
    eval_report = evaluate_backtest(
        res_df, add_sparsity_feature(df, col_product, train_years),
        col_product, train_years, category_cols=category_flag, fiscal_start_month=fiscal_start_month,
    ) if res_df is not None else None

    # Point model + (optionally) one model per quantile, all fit on identical
    # rows/features so they line up at predict time.
    _report_stage(f"Training {chosen_model_name} and forecasting FY{int(forecast_year)}")
    final_model, quantile_models, feature_cols_final = train_final_model(
        df, col_product, category_flag, train_years, validation_year, fiscal_start_month,
        chosen_model_name, extra_features=extra_features, feature_base=base_features,
        quantiles=QUANTILES if add_prediction_intervals else None)

    df_with_sparsity = add_sparsity_feature(df, col_product, train_years + [validation_year])
    fc_df = forecast_future(df_with_sparsity, final_model, col_product, category_flag, feature_cols_final,
                            forecast_year, fiscal_start_month, quantile_models=quantile_models or None)
    fc_df = fc_df[~fc_df[col_product].isin(_discontinued)].reset_index(drop=True)
    quantile_cols = [c for c in fc_df.columns if c.startswith("q") and c[1:].isdigit()]

    # Residual-bias correction: a mean-objective model on mostly-zero data
    # under-forecasts the spikes. Measure the per-sparsity-tier multiplier on the
    # backtest; apply it to every forecast column (point + quantiles together, so
    # the interval shifts with the median). Always computed for the before/after
    # panel on Model Accuracy; only applied when the toggle is on.
    bias_correction = compute_bias_correction(res_df, chosen_model_name, df_with_sparsity, col_product)
    if apply_bias_corr and bias_correction:
        fc_df = apply_bias_correction(fc_df, bias_correction, df_with_sparsity, col_product,
                                      ["ML_Forecast", "Final_Forecast"] + quantile_cols)

    # You can't order a fraction of a PLC -- round the unit forecast to whole
    # numbers. Sub-0.5 forecasts round to 0, which is the right call (don't
    # stock an item the model expects ~no demand for).
    for _c in ("Baseline_Forecast", "ML_Forecast", "Final_Forecast", *quantile_cols):
        if _c in fc_df.columns:
            fc_df[_c] = fc_df[_c].round()

    # Interval calibration on the held-out year (did p10-p90 actually contain
    # ~80% of the real values?). Independent of the forecast above.
    coverage_report = None
    if add_prediction_intervals:
        _report_stage("Checking the p10-p90 range on the held-out year")
        try:
            coverage_report = quantile_backtest(
                df, col_product, category_flag, train_years, validation_year, fiscal_start_month,
                quantiles=QUANTILES, extra_features=extra_features, feature_base=base_features)
        except Exception:
            coverage_report = None

    # --- Honest backtest: scored the way the forecast is actually made --------
    # run_backtest predicts each validation month from the ACTUAL months before
    # it -- a forecast made 1 month ahead. The forecast MELFT receives is made
    # once, at the start of the year, feeding its own predictions forward -- up
    # to 12 months ahead. That second score costs one extra fit. A second fit one
    # origin earlier gives every item a sigma measured on a year the fill-rate
    # replay has NOT seen: the in-sample sigma made the old 95.9% fill rate
    # optimistic (it had already seen the year it was scored on). Failures are
    # recorded, not raised -- the rest of the build is still valid without this.
    honest = None
    _report_stage("Scoring the forecast 12 months ahead")
    try:
        _hz = horizon_backtest(df, col_product, category_flag, train_years, validation_year, fiscal_start_month,
                               chosen_model_name, extra_features=extra_features, feature_base=base_features,
                               quantiles=QUANTILES if add_prediction_intervals else None)
        if _hz is not None:
            _pc = f"{chosen_model_name}_Pred"
            # p90 hit rate with the range forecast 12 months ahead -- the side
            # that sizes safety stock, at the horizon it is actually used at.
            _upper_hit = None
            if "q90" in _hz.columns:
                _u = res_df[[col_product, "FiscalMonth"]].merge(  # pyright: ignore[reportOptionalSubscript]
                    _hz[[col_product, "FiscalMonth", "Actual", "q90"]], on=[col_product, "FiscalMonth"], how="left")
                _upper_hit = float((_u["Actual"] <= _u["q90"].fillna(0.0)).mean())
            hz_table, hz_scored = horizon_comparison(
                res_df, chosen_model_name, _hz, df, col_product, train_years, category_cols=category_flag,
                fiscal_start_month=fiscal_start_month)
            # res_df-shaped (Actual + '<model>_Pred') so the inventory functions take it unchanged.
            hz_res = hz_scored[[col_product, "FiscalMonth", "Actual"]].assign(
                **{_pc: hz_scored[f"{HORIZON_LABELS[1]}_Pred"].to_numpy()})
            _prior_year = int(validation_year) - 1
            _prior_train = [y for y in train_years if int(y) < _prior_year]
            sigma_honest = sigma_basis = None
            if _prior_train and _prior_year in available_years:
                _hp = horizon_backtest(df, col_product, category_flag, _prior_train, _prior_year, fiscal_start_month,
                                       chosen_model_name, extra_features=extra_features, feature_base=base_features)
                if _hp is not None:
                    sigma_honest = forecast_error_sigma(_hp.rename(columns={"Pred": _pc}), chosen_model_name,
                                                        col_product).to_dict()
                    sigma_basis = f"FY{_prior_year}"
            if sigma_honest is None:
                sigma_honest = forecast_error_sigma(hz_res, chosen_model_name, col_product).to_dict()
                sigma_basis = f"FY{int(validation_year)} (no earlier year to measure it on)"
            # Recent-sales window ending the March before the replayed year.
            _pre = df_with_sparsity[df_with_sparsity["FiscalYear"] < int(validation_year)]
            _pt = _pre["FiscalYear"] * 12 + _pre["FiscalMonth"]
            _trail_pre = (_pre[_pt > _pt.max() - 12].groupby(col_product)["Units"].sum().to_dict()
                          if len(_pre) else {})
            honest = dict(table=hz_table, res=hz_res, sigma=sigma_honest,
                          sigma_basis=sigma_basis, trailing=_trail_pre, upper_hit=_upper_hit)
    except Exception as e:
        honest = {"error": f"{type(e).__name__}: {e}"}

    # --- Inventory-planning layer: forecast -> stocking parameters ---------
    # Turns the demand forecast into what the Purchasing team reviews each
    # quarter (safety stock / reorder point / order-up-to level), sized from
    # the model's OWN error on each item rather than a flat formula.
    _report_stage("Sizing the inventory plan and obsolescence flags")
    price_map = {}
    if "ListPrice" in df_full.columns:
        price_map = df_full.drop_duplicates(col_product).set_index(col_product)["ListPrice"].to_dict()
    sigma_map = forecast_error_sigma(res_df, chosen_model_name, col_product).to_dict()
    horizon_n = fc_df["FiscalMonth"].nunique() or 12
    # Average monthly p50 / p90 per product -- lets the inventory layer size a
    # safety stock straight from the model's own 90th-percentile demand and
    # cross-check it against the sigma-based formula.
    p50_map = p90_map = None
    if "q50" in fc_df.columns and "q90" in fc_df.columns:
        _qg = fc_df.groupby(col_product).agg(_p50=("q50", "mean"), _p90=("q90", "mean"))
        p50_map = _qg["_p50"].to_dict()
        p90_map = _qg["_p90"].to_dict()
    # Units actually sold in the last 12 fiscal months. Only feeds the
    # recommended_action label, so a zero forecast can't call an item that
    # demonstrably sold "no demand" (same rule as the obsolescence flags:
    # accuse on actuals, never on the forecast alone).
    _t = df_with_sparsity.assign(_t=df_with_sparsity["FiscalYear"] * 12 + df_with_sparsity["FiscalMonth"])
    _trailing_map = (_t[_t["_t"] > _t["_t"].max() - horizon_n]
                     .groupby(col_product)["Units"].sum().to_dict())
    # Obsolescence runs against the FULL catalog (incl. the discontinued
    # items we dropped from the forecast) so they still get flagged here.
    obs_status_map = {}
    if "ProductionStatus" in df_full.columns:
        obs_status_map = df_full.drop_duplicates(col_product).set_index(col_product)["ProductionStatus"].to_dict()
    # Everything the plan needs besides the stocking rules. Kept in build_meta
    # so the Inventory Plan page can re-plan for new rules without a rebuild;
    # the plan below uses the default rules.
    inv_inputs = dict(horizon=horizon_n, sigma=sigma_map, price=price_map, p50=p50_map, p90=p90_map,
                      trailing=_trailing_map, status=obs_status_map,
                      history=df_full[[col_product, "FiscalYear", "FiscalMonth", "Units"]])
    inv_plan, obs_flags = plan_for_rules(fc_df, col_product, inv_inputs, INV_DEFAULTS)

    return (df_with_sparsity, fc_df, res_df, metrics, chosen_model_name, n_dropped_dates, skipped,
            importance_df, importance_method, eval_report, inv_plan, obs_flags,
            dict(extra_features=extra_features, category_flag=category_flag,
                 base_features=base_features, n_discontinued_excluded=n_discontinued_excluded,
                 n_duplicate_lines=n_duplicate_lines, n_unpriced_items=n_unpriced_items,
                 # Features the user selected that this data could not supply.
                 unavailable_features=sorted(set(unavailable_features)),
                 cust_cat_history=cust_cat_history, cust_name_map=cust_name_map,
                 # Every item's code, discontinued ones included -- `forecast` excludes
                 # them, but the obsolescence list shows them.
                 item_code_map=(df_full.drop_duplicates(col_product).set_index(col_product)["Item Code"].to_dict()
                                if "Item Code" in df_full.columns else {}),
                 cust_type_map=cust_type_map or {}, cust_industry_map=cust_industry_map or {},
                 col_customer=col_customer,
                 quantile_cols=quantile_cols, bias_correction=bias_correction,
                 bias_applied=bool(apply_bias_corr and bias_correction),
                 coverage_report=coverage_report,
                 honest=honest,
                 inv_params=dict(INV_DEFAULTS), inv_inputs=inv_inputs))


@st.cache_data(show_spinner=False)
def run_feature_ablation(history_df, col_product, train_years_tuple, validation_year, fiscal_start_month, model_type):
    return feature_ablation(history_df, col_product, list(train_years_tuple), validation_year,
                            fiscal_start_month, model_type=model_type)


@st.cache_data(show_spinner=False)
def run_classical_crosscheck(history_df, forecast_df, fiscal_start_month, value_col, ml_backtest_monthly_tuple):
    return aggregate_crosscheck(history_df, forecast_df, fiscal_start_month, value_col=value_col,
                                ml_backtest_monthly=ml_backtest_monthly_tuple)


@st.cache_data(show_spinner=False)
def run_walk_forward(history_df, col_product, category_flag, fiscal_start_month, model_type,
                     extra_features_tuple, all_years_tuple, base_features_tuple=None):
    """Rolling-origin backtest, cached separately from the main build because
    it retrains the model once per origin year (slower)."""
    return walk_forward_backtest(
        history_df, col_product, list(category_flag) if category_flag else category_flag,
        fiscal_start_month, [model_type],
        all_years=list(all_years_tuple), min_train_years=2,
        extra_features=list(extra_features_tuple),
        feature_base=list(base_features_tuple) if base_features_tuple else None,
    )


@st.cache_data(show_spinner=False)
def run_year_backtest(history_df, col_product, category_flag, fiscal_start_month, model_types_tuple,
                      extra_features_tuple, train_years_tuple, validation_year, base_features_tuple=None):
    """Every model backtested on one earlier year, for Model Accuracy's "Monthly miss
    by model" year switch. The same run_backtest the build uses, on another split."""
    res, *_ = run_backtest(
        history_df, col_product, list(category_flag) if category_flag else category_flag,
        list(train_years_tuple), validation_year, fiscal_start_month, list(model_types_tuple),
        extra_features=list(extra_features_tuple),
        feature_base=list(base_features_tuple) if base_features_tuple else None)
    return res


@st.cache_data(show_spinner=False)
def run_level_crosscheck(history_df, sku_res_df, forecast_df, col_product, level_col,
                         train_years_tuple, validation_year, fiscal_start_month, model_type,
                         extra_features_tuple, base_features_tuple, category_flag_tuple, forecast_year):
    """Bottom-up (sum of item forecasts) vs forecasting the category series
    directly, scored at level x month on TWO origin years.

    Two origins matter here: at the category grain the methods are close
    enough that a single held-out year can crown any of them by luck, and
    they genuinely do swap places. Cached separately from the main build
    because the earlier origin needs its own item-level backtest.

    The most recent origin reuses the build's own `sku_res_df`, so the
    bottom-up row is the exact number the rest of the app reports rather than
    a near-miss recomputation.

    Returns {'comparison': [Method, Approach, WAPE, Bias, Forecast, Actual, Origin],
             'verdict': str, 'forward': [level, Bottom-up, Direct] or None}."""
    tys = [int(y) for y in train_years_tuple]
    extra = list(extra_features_tuple)
    base = list(base_features_tuple) if base_features_tuple else None
    cats = list(category_flag_tuple) if category_flag_tuple else None

    origins = [(tys, int(validation_year), sku_res_df)]
    # Only step the origin back if doing so still leaves two training years --
    # one year of history can't produce a lag_12 feature at all.
    if len(tys) >= 3:
        origins.append((tys[:-1], tys[-1], None))

    parts = []
    for o_tys, o_vy, reuse in origins:
        tbl, _ = level_method_comparison(
            history_df, col_product, level_col, o_tys, o_vy, fiscal_start_month, model_type,
            extra_features=extra, feature_base=base, category_cols=cats, sku_res_df=reuse)
        try:
            ctbl, _ = per_level_crosscheck(history_df, level_col, o_tys, o_vy, fiscal_start_month)
        except Exception:
            ctbl = pd.DataFrame(columns=tbl.columns)   # statsmodels missing or unfittable
        part = pd.concat([tbl, ctbl], ignore_index=True)
        part["Origin"] = f"Validated on FY{int(o_vy)}"
        parts.append(part)
    comparison = pd.concat(parts, ignore_index=True)

    # --- verdict, computed from the table rather than asserted ------------
    lines, top2_everywhere, ever_best = [], True, False
    for origin in sorted(comparison["Origin"].unique()):
        g = comparison[comparison["Origin"] == origin].sort_values("WAPE").reset_index(drop=True)
        pos = g.index[g["Approach"] == "Bottom-up"]
        rank = int(pos[0]) + 1 if len(pos) else None
        best = g.iloc[0]
        lines.append(f"**{origin.replace('Validated on ', '')}**: bottom-up ranks "
                     f"{rank} of {len(g)} (best here was {best['Method']} at {best['WAPE']:.1%})")
        if rank is None or rank > 2:
            top2_everywhere = False
        if rank == 1:
            ever_best = True
    verdict = "  ·  ".join(lines) + ".  "
    if len(origins) < 2:
        verdict += ("Only one origin year is available with the current train/validation split, so read "
                    "this as a single sample -- widen the training years to get a second.")
    elif top2_everywhere and not ever_best:
        verdict += ("No method wins at both origins -- the ranking flips, which is exactly why one "
                    "held-out year can't settle this. Bottom-up is the only approach in the top two at "
                    "both, and the only one that also produces the per-item forecast the inventory plan "
                    "is built from, so it stays the plan of record.")
    elif top2_everywhere:
        verdict += ("Bottom-up is at or near the top at both origins, and it is the only approach that "
                    "also produces the per-item forecast the inventory plan is built from.")
    else:
        verdict += ("The ranking moves between origins, so read the two columns together rather than "
                    "either alone. Bottom-up stays the plan of record because it is the only approach "
                    "that also produces the per-item forecast the inventory plan needs.")

    # --- forward: the direct model's FY forecast next to the plan ----------
    forward = None
    try:
        direct = level_direct_forecast(history_df, level_col, tys, int(validation_year),
                                       fiscal_start_month, model_type, int(forecast_year),
                                       feature_base=base)
        if direct is not None and forecast_df is not None:
            lm = level_map_from(history_df, col_product, level_col)
            f = forecast_df.copy()
            f[level_col] = f[col_product].map(lm).fillna(UNKNOWN_LEVEL)
            bu = (f.groupby(level_col, as_index=False)["Final_Forecast"].sum()
                  .rename(columns={"Final_Forecast": "Bottom-up"}))
            forward = (bu.merge(direct, on=level_col, how="outer").fillna(0)
                       .sort_values("Bottom-up", ascending=False).reset_index(drop=True))
    except Exception:
        forward = None

    return {"comparison": comparison, "verdict": verdict, "forward": forward}


@st.cache_data(show_spinner=False)
def run_service_simulation(res_df, history_df, col_product, model_name, train_years_tuple,
                           validation_year, lead, review, service, legacy_months):
    """Fill-rate backtest of the stocking policy itself.

    Sizes two order-up-to policies from the SAME held-out-year forecast --
    one with safety stock from each item's own measured error, one with the
    flat '{legacy_months} months for everything' rule -- then replays both
    against that year's real monthly demand. Because both are handed the
    identical forecast, the gap between them isolates the value of the error
    estimate rather than the value of the forecast.

    The policy is built by calling build_inventory_plan on the BACKTEST
    predictions (not the forward forecast), so the levels are the ones we
    would have set standing at the start of the validation year."""
    pred_col = f"{model_name}_Pred"
    # Reshape the backtest into what build_inventory_plan expects, so the
    # policy levels come from the same tested code the Inventory Plan page uses.
    bt_fc = res_df[[col_product, "FiscalMonth", pred_col]].rename(
        columns={pred_col: "Final_Forecast"})
    sigma = forecast_error_sigma(res_df, model_name, col_product).to_dict()
    _t = history_df.assign(_t=history_df["FiscalYear"] * 12 + history_df["FiscalMonth"])
    trailing = (_t[_t["_t"] > _t["_t"].max() - 12].groupby(col_product)["Units"].sum().to_dict())
    price = {}
    if "ListPrice" in history_df.columns:
        price = history_df.drop_duplicates(col_product).set_index(col_product)["ListPrice"].to_dict()

    bt_plan = build_inventory_plan(
        bt_fc, col_product, horizon_months=12, lead_time_months=lead,
        review_period_months=review, service_level=service, sigma_by_product=sigma,
        legacy_safety_months=legacy_months, list_price=price or None,
        trailing_actual_by_product=trailing)

    s_model = bt_plan.set_index(col_product)["order_up_to_level"].astype(float)
    # The flat-rule policy differs ONLY in the safety-stock term.
    s_flat = (bt_plan.set_index(col_product)["expected_demand_protection"].astype(float)
              + bt_plan.set_index(col_product)["legacy_safety_stock"].astype(float))
    summary, _ = service_level_comparison(
        res_df, col_product,
        {f"Forecast-driven safety stock (σ per item, {service:.0%} target)": s_model.to_dict(),
         f"Flat {legacy_months}-month rule (current practice)": s_flat.to_dict()},
        lead_time_months=lead, review_period_months=review,
        list_price=price or None)
    return summary


FRONTIER_SERVICE_LEVELS = (0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99)
FRONTIER_FLAT_MONTHS = (1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12)


@st.cache_data(show_spinner=False, max_entries=64)
def run_policy_frontier(horizon_res, col_product, model_name, sigma, trailing, price,
                        lead, review, service_levels, flat_months):
    """Same-budget comparison of the two stocking policies, replayed honestly: a
    12-month-ahead forecast, sigma from the previous year's errors, and a
    recent-sales window that ends before the replayed year (all from
    build_meta['honest']). Cached, so a rerun of the page doesn't redo the
    ~0.5s replay."""
    pred_col = f"{model_name}_Pred"
    fc_like = horizon_res[[col_product, "FiscalMonth", pred_col]].rename(columns={pred_col: "Final_Forecast"})
    return policy_frontier(horizon_res, fc_like, col_product, sigma, lead_time_months=lead,
                           review_period_months=review, service_levels=service_levels, flat_months=flat_months,
                           list_price=price or None, trailing_actual_by_product=trailing or None)


def build_full_workbook(history_df, res_df, metrics_df, eval_report, forecast_df, inv_df, obs_df):
    """Serialises every result frame into one multi-sheet .xlsx and returns the
    bytes. Deliberately NOT cached: the caller holds the result in session
    state keyed on build_version, which avoids re-hashing a 250k-row frame
    just to look up a value we already have. ~1 minute on the real data --
    openpyxl is the bottleneck, not the data prep."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        history_df.to_excel(writer, sheet_name="Aggregated_History", index=False)
        if res_df is not None:
            res_df.to_excel(writer, sheet_name="Backtest_Accuracy", index=False)
        if metrics_df is not None:
            metrics_df.to_excel(writer, sheet_name="Accuracy_Summary", index=False)
        if eval_report is not None:
            eval_report["summary"].to_excel(writer, sheet_name="Accuracy_AllSKUs")
            eval_report["active_only"].to_excel(writer, sheet_name="Accuracy_ActiveSKUs")
            eval_report["by_grain"].to_excel(writer, sheet_name="Accuracy_ByGrain")
            if eval_report.get("by_sparsity") is not None:
                eval_report["by_sparsity"].to_excel(writer, sheet_name="Accuracy_BySparsity", index=False)
        forecast_df.to_excel(writer, sheet_name="Forecast", index=False)
        if inv_df is not None:
            inv_df.to_excel(writer, sheet_name="Inventory_Plan", index=False)
        if obs_df is not None and len(obs_df):
            obs_df.to_excel(writer, sheet_name="Obsolescence_Flags", index=False)
    return buf.getvalue()


def revenue_axis():
    """Y axis for a baht chart. Vega's "~s" format uses SI prefixes, so a billion
    baht read "1G" (giga) -- meaningless to a sales audience. Label with the
    bn / M / K abbreviations the rest of the app already uses for money
    (e.g. "฿1.05bn", "฿537M"). abs() so the Model Accuracy error chart, which
    goes below zero, keeps its sign ("-200M")."""
    return alt.Axis(labelExpr=(
        "abs(datum.value) >= 1e9 ? format(datum.value / 1e9, '.2~f') + 'bn' : "
        "abs(datum.value) >= 1e6 ? format(datum.value / 1e6, '.2~f') + 'M' : "
        "abs(datum.value) >= 1e3 ? format(datum.value / 1e3, '.2~f') + 'K' : "
        "format(datum.value, '.2~f')"))


def price_per_item(history_df, col_product):
    """Average realised THB price per unit for each product (total past
    Revenue / total past Units), falling back to the item-master list price,
    then the global average. Used to turn a unit forecast into a revenue
    forecast."""
    g = history_df.groupby(col_product).agg(rev=("Revenue", "sum"), un=("Units", "sum"))
    price = g["rev"] / g["un"].replace(0, np.nan)
    if "ListPrice" in history_df.columns:
        lp = history_df.drop_duplicates(col_product).set_index(col_product)["ListPrice"]
        price = price.fillna(lp.where(lp > 0))
    return price.fillna(price.mean())


def top_products(hist, fc, col_product, measure, fy_act, n=10):
    """The n products with the biggest forecast-year forecast on `measure`
    (revenue at each item's realised price, as on Overview's chart), with the
    `fy_act` actual. Overview's #1 and Product Detail's list both come from here.
    Returns (frame indexed by product with Forecast / Actual, forecast total)."""
    f = fc.groupby(col_product)["Final_Forecast"].sum()
    if measure == "Revenue":
        f = f * price_per_item(hist, col_product).reindex(f.index)
    a = hist[hist["FiscalYear"] == fy_act].groupby(col_product)[measure].sum()
    out = pd.DataFrame({"Forecast": f, "Actual": a.reindex(f.index).fillna(0.0)})
    return out.sort_values(["Forecast", "Actual"], ascending=False, kind="stable").head(n), float(f.sum())


def top_customers(cch, ccustcol, val, fy, custs=None, n=10):
    """The n customers with the most `val` in `fy`, lifetime beside it; `custs`
    limits the pool (Customer Detail's filters). Returns (frame indexed by
    customer with FY / Lifetime, FY total of the pool)."""
    life = cch.groupby(ccustcol)[val].sum()
    last = cch[cch["FiscalYear"] == fy].groupby(ccustcol)[val].sum()
    pool = life.index if custs is None else custs
    out = pd.DataFrame({"FY": last.reindex(pool).fillna(0.0), "Lifetime": life.reindex(pool).fillna(0.0)})
    return (out.sort_values(["FY", "Lifetime"], ascending=False, kind="stable").head(n),
            float(out["FY"].sum()))


def open_item(page_name, focus_key, item):
    """on_click callback: open a detail page on one item. The page's picker is
    keyed `focus_key`; `_clicked` lets it add the item back if a filter hid it."""
    go_to(page_name)
    st.session_state[focus_key] = st.session_state[focus_key + "_clicked"] = item


def ranking_table(df, items, page_name, focus_key, table_key, **kw):
    """A top-N table whose row click opens that item on `page_name`. Each click
    swaps in a fresh table key, so no row stays selected and the same row can
    be clicked again after picking something else."""
    n = st.session_state.get(table_key + "_n", 0)
    key = f"{table_key}_{n}"

    def _pick():
        # Cell selection too: a click on any cell of the row opens it, not only its checkbox.
        sel = st.session_state[key].selection
        rows = list(sel.rows) + [c[0] for c in sel.cells]
        if rows:
            open_item(page_name, focus_key, items[rows[0]])
            st.session_state[table_key + "_n"] = n + 1

    st.dataframe(df, key=key, on_select=_pick, selection_mode=["single-row", "single-cell"], hide_index=True, **kw)


def with_clicked(options, focus_key):
    """A picker's options, with a just-clicked item put first if a filter or
    search had left it out (until the picker is changed by hand)."""
    c = st.session_state.get(focus_key + "_clicked")
    if c is not None and c == st.session_state.get(focus_key) and c not in options:
        return [c] + list(options)
    return options


def monthly_series(long_df, group_cols, value_col, price=None, col_product=None):
    """Sum `value_col` by `group_cols`; if `price` is given, first convert
    to revenue (value x per-item price) so the same code path serves both
    the Units and Revenue views."""
    d = long_df
    if price is not None:
        d = d.copy()
        d["_v"] = d[value_col] * d[col_product].map(price).fillna(price.mean())
        value_col = "_v"
    return d.groupby(group_cols, as_index=False)[value_col].sum().rename(columns={value_col: "Value"})


def bucket_periods(df, year_col, month_col, granularity):
    """Adds a Period label + a numeric `t` sort key for either a Monthly or a
    Quarterly view of a fiscal-month series. Quarterly buckets fiscal months
    1-3 -> Q1, 4-6 -> Q2, ... (Q1 is the start of the fiscal year, so for an
    April start Q1 = Apr-Jun). Callers then group by [t, Period, Series] and
    sum, so the same chart code serves both granularities."""
    d = df.copy()
    y = d[year_col].astype(int)
    m = d[month_col].astype(int)
    if granularity == "Quarterly":
        q = (m - 1) // 3 + 1
        d["Period"] = "FY" + y.astype(str) + "-Q" + q.astype(str)
        d["t"] = y * 4 + q
    else:
        # Calendar month and year ("Jan '26", as the design system labels months): "FY2025-Jan"
        # read as January 2025 when it is January 2026.
        cal_y = y + (m + int(fiscal_start_month) - 1 > 12)
        d["Period"] = m.map(lambda mm: fmonth_labels[mm - 1]) + " '" + (cal_y % 100).map("{:02d}".format)
        d["t"] = y * 12 + m
    d["FY"] = "FY" + y.astype(str)
    return d


def period_axis(period_order, granularity):
    """x axis for bucket_periods() output, labels level (never rotated, as the design
    system asks). Monthly: each fiscal quarter's first month, or every half-year past
    three years. Quarterly: every quarter, or Q1 only past three years."""
    long = len(period_order) > (36 if granularity == "Monthly" else 12)
    if granularity == "Monthly":
        keep = {fmonth_labels[i] for i in ((0, 6) if long else (0, 3, 6, 9))}
        values = [p for p in period_order if p.split(" ")[0] in keep]
    else:
        values = [p for p in period_order if not long or p.endswith("Q1")]
    return alt.Axis(values=values, labelAngle=0)


def kpi_card(label, value, delta=None, delta_positive_is_good=True, delta_text=None, note=None, exact=None):
    """delta sets the arrow direction and colour. delta_text replaces the default
    "{abs(delta):.1%}" -- pass it whenever delta is a gap in percentage POINTS
    (a WAPE or coverage difference), which a % sign would misstate.
    note: a quiet line under the value naming what it is compared against.
    exact: the unrounded value, shown on hover when `value` is abbreviated."""
    delta_html = ""
    if delta is not None:
        is_up = delta >= 0
        good = is_up if delta_positive_is_good else not is_up
        cls = "kpi-delta-up" if good else "kpi-delta-down"
        arrow = "&#9650;" if is_up else "&#9660;"
        delta_html = f'<div class="{cls}">{arrow} {delta_text or f"{abs(delta):.1%}"}</div>'
    note_html = f'<div class="kpi-note">{html.escape(note)}</div>' if note else ""
    title = f' title="{html.escape(exact)}"' if exact else ""
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value"{title}>{value}</div>{delta_html}{note_html}</div>', unsafe_allow_html=True)


def short_baht(v):
    """฿ with three significant figures: ฿1.05bn, ฿588M, ฿36.6M, ฿4.21M. For KPI cards
    and prose; tables and exports keep the exact figure."""
    v = float(v)
    if abs(v) >= 1e9:
        return f"฿{v / 1e9:,.2f}bn"
    if abs(v) >= 1e6:
        m = v / 1e6
        return f"฿{m:,.0f}M" if abs(m) >= 100 else (f"฿{m:,.1f}M" if abs(m) >= 10 else f"฿{m:,.2f}M")
    return f"฿{v:,.0f}"


ACC_SECTIONS = ["Business impact", "Model accuracy", "Diagnostics"]


def go_to(page_name, tab=None):
    """on_click callback: open another page (and Model Accuracy tab). Callbacks run
    before the sidebar radio is drawn, so its key can be written here. A real page
    change also asks for a scroll to the top: the page otherwise opens at the old
    page's scroll position (a button low on Overview landed at the bottom)."""
    if st.session_state.get("nav_page") != page_name:
        st.session_state["_scroll_top"] = True
    st.session_state["nav_page"] = page_name
    if tab:
        st.session_state["acc_tab"] = tab


def empty_state(message, button=None, page_name="Build Forecast"):
    """A card in place of a bare notice when a page has nothing to show."""
    with st.container(border=True):
        st.markdown(message)
        if button:
            st.button(button, on_click=go_to, args=(page_name,), key=f"_empty_{page_name}")


@st.cache_data(show_spinner=False)
def glossary_sections(path="GLOSSARY.md"):
    """Part A of GLOSSARY.md (the forecasting / inventory vocabulary; Part B is about
    working on the code) as [(heading, header cells, rows of cells)], one per '###' table.
    A heading's '(...)' aside is dropped, as on the board."""
    try:
        with open(path, encoding="utf-8") as f:
            part_a = f.read().split("\n## Part B")[0]
    except OSError:
        return []
    sections = []
    for block in part_a.split("\n### ")[1:]:
        lines = block.splitlines()
        table = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines if ln.startswith("|")]
        if len(table) > 2:
            sections.append((lines[0].split(" (")[0].strip(), table[0], table[2:]))
    return sections


def _md_inline(s):
    """GLOSSARY.md's inline Markdown (**bold**, *italic*, `code`) as escaped HTML."""
    s = html.escape(s, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"\*(.+?)\*", r"<em>\1</em>", s)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", s)


def _heading_id(title):
    """A glossary card's anchor id: 'Model types' -> 'gl-model-types'."""
    return "gl-" + "-".join(re.findall(r"[a-z0-9]+", title.lower()))


@st.dialog("Glossary", width="large")
def show_glossary(sections):
    """The board's Glossary page as a pop-up: search and "On this page" jump links on
    the left, one card per GLOSSARY.md section on the right."""
    n_terms = sum(len(rows) for _, _, rows in sections)
    st.markdown('<div class="page-subtitle gl-subtitle">Words that make the forecast and the stocking plan easier to talk about</div>',
                unsafe_allow_html=True)
    aside, body = st.container(key="plain_glossary_detail").columns([300, 900], gap="medium")
    with aside, st.container(border=True):
        q = st.text_input(f"Search {n_terms} terms", placeholder="For example WAPE or safety stock",
                          key="glossary_q").strip().lower()
        shown = [(h, head, [r for r in rows if q in re.sub(r"[*`]", "", " ".join(r)).lower()])
                 for h, head, rows in sections]
        shown = [s for s in shown if s[2]]
        st.markdown('<div class="gl-kicker">On this page</div>' + "".join(
            f'<a class="gl-link" href="#{_heading_id(h)}"><span>{html.escape(h)}</span><span>{len(rows)}</span></a>'
            for h, _, rows in shown), unsafe_allow_html=True)
    with body:
        st.caption("The forecasting and inventory vocabulary this project runs on. Use these terms in the deck and with mentors.")
        if not shown:
            empty_state(f"No term matches “{html.escape(q)}”.")
        for h, head, rows in shown:
            cols = "220px minmax(0, 1fr)" + (" 300px" if len(head) > 2 else "")
            lines = [f'<div class="gl-row gl-head">{"".join(f"<div>{html.escape(c)}</div>" for c in head)}</div>']
            lines += [f'<div class="gl-row">{"".join(f"<div>{_md_inline(c)}</div>" for c in r)}</div>' for r in rows]
            with st.container(border=True):
                st.markdown(f'<div class="gl-title" id="{_heading_id(h)}">{html.escape(h)}</div>'
                            f'<div class="gl-table" style="--gl-cols: {cols};">{"".join(lines)}</div>',
                            unsafe_allow_html=True)


def profile_card(label, value, note=None, accent=False, inset=False):
    """One fact as a compact card, for the Customer / Product Detail profile
    rows. Unlike kpi_card, which only ever receives formatted numbers, the
    value here is raw data -- company names, FA Sub Industry strings -- so it
    is HTML-escaped before it goes into unsafe_allow_html markup.
    inset: grey fill, for a tile that sits inside a white card rather than on the
    grey page. accent: the value in orange, i.e. the selected model or plan."""
    note_html = f'<div class="profile-note">{html.escape(str(note))}</div>' if note else ""
    card_cls = "profile-card profile-card-inset" if inset else "profile-card"
    value_cls = "profile-value profile-value-accent" if accent else "profile-value"
    st.markdown(f'<div class="{card_cls}"><div class="profile-label">{html.escape(str(label))}</div>'
                f'<div class="{value_cls}">{html.escape(str(value))}</div>{note_html}</div>',
                unsafe_allow_html=True)


# The Forecast Console scale (design system tokens) for the redesigned pages:
# 16px between cards (space-4), 24px card padding (space-5), 48px from a band's
# cards to the next "#### section" (space-7: the <br> before it is sized to make
# that), a 12px step from a title to its content; headings 20/28, captions 12/16
# muted, KPI values 36/40. Emitted with the page title, so it lives only while
# one of these pages renders and takes no block of its own (a lone <style>
# element would still take a 16px gap). The card rule repeats the global
# selector so it wins by order, and still loses to the keyed special cards.
FC_SCALE_PAGES = ("Build Forecast", "Overview", "Inventory Plan", "Model Accuracy",
                  "Product Detail", "Customer Detail", "Category Plan", "Forecast Table")
FC_SCALE_CSS = f"""<style>
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] :is(h4, h5) {{ font-size: 20px !important;
        line-height: 28px !important; padding: 0 !important; margin: 0 0 11px !important; }}
    [data-testid="stMain"] [data-testid="stCaptionContainer"] p {{ font-size: 12px; line-height: 16px;
        color: {TEXT_MUTED}; }}
    div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:not([class*="st-key-plain_"]):not(:has(> div[data-testid="stLayoutWrapper"] > [class*="st-key-plain_"]:where([class*="_detail"]))) {{
        padding: 24px !important; }}
    div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"][class*="st-key-rulecard_"][class*="st-key-rulecard_"][class*="st-key-rulecard_"][class*="st-key-rulecard_"] {{
        gap: 8px !important; }}
    [data-testid="stMain"] [data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] [data-testid="stMarkdownContainer"] > br:only-child) {{
        height: 15px; }}
    /* A title beside controls (a 40px button or radio row): centred on them, its
       12px step taken by the row's own height, and a section row pulled up by the
       centring offset so the band still measures 48px to the title. */
    /* Only a title placed straight in a column: a title inside a card in a column
       must not centre the whole page row. */
    [data-testid="stMain"] [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] [data-testid="stMarkdownContainer"] :is(h4, h5) {{
        margin-bottom: 0 !important; }}
    [data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] :is(h4, h5)) {{
        align-items: center !important; }}
    [data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] h4) {{
        margin-top: -15px; }}
    /* A segmented control (34px) is shorter than a button, so its row adds the
       4px that keeps the title 12px off what follows. */
    [data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] :is(h4, h5)):has([data-testid="stButtonGroup"]) {{
        margin-bottom: 4px; }}
    /* A card title alone in its column beside a switch or button: Markdown's -1rem
       bottom margin left a 12px box for the 28px title, so centring that box set the
       title 8px low. (A title with a caption under it is not centred; left as is.) */
    [data-testid="stMain"] [data-testid="stHorizontalBlock"]:has([data-testid="stButtonGroup"], [data-testid="stButton"]) > [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:only-child [data-testid="stMarkdownContainer"]:has(h5) {{
        margin-bottom: 0 !important; }}
    /* The KPI cards stretch 1rem past their row to undo Markdown's -1rem margin,
       which ran them into the card below; the row reserves that 1rem. */
    [data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.kpi-card) {{ margin-bottom: 1rem; }}
    [data-testid="stMain"] .kpi-card {{ padding: 24px; }}
    [data-testid="stMain"] .kpi-label {{ font-size: 12px; line-height: 16px; margin-bottom: 8px; }}
    [data-testid="stMain"] .kpi-value {{ font-size: min(36px, 20cqi); line-height: 40px; font-weight: 600;
        letter-spacing: -0.01em; }}
    [data-testid="stMain"] .kpi-note {{ font-size: 12px; line-height: 16px; margin-top: 8px; }}
    [data-testid="stMain"] :is(.stButton, .stDownloadButton) button {{ border-radius: 8px !important; }}
    /* Product Detail: 24px between the page's sections, as the board's main column
       spaces them (16px block gap + 8px). */
    :is(.st-key-pd_rhythm, .st-key-pd_chart, .st-key-pd_monthly, .st-key-cd_chart, .st-key-cd_projection,
        [class*="st-key-cp_"], [class*="st-key-ft_"]) {{ margin-top: 8px; }}
    /* Order rhythm tiles in the design system's terms: white with a hairline
       (sunken fill is for table headers and control tracks), a sentence-case
       12px label, the value at the 16px subheading size, a 12px muted note. */
    /* The tiles' Markdown runs 1rem past their row, as the KPI cards do; the row
       reserves it so the card keeps its 24px bottom padding. */
    .st-key-pd_rhythm [data-testid="stHorizontalBlock"] {{ margin-bottom: 1rem; }}
    .st-key-pd_rhythm .profile-card-inset {{ background-color: {CARD}; border: 1px solid {BORDER_SUBTLE};
        padding: 16px; min-height: 0; }}
    .st-key-pd_rhythm .profile-label {{ font-size: 12px; line-height: 16px; font-weight: 600;
        letter-spacing: 0; text-transform: none; margin-bottom: 8px; }}
    .st-key-pd_rhythm .profile-value {{ font-size: 16px; line-height: 24px; }}
    .st-key-pd_rhythm .profile-note {{ font-size: 12px; line-height: 16px; font-weight: 400; margin-top: 4px; }}
</style>"""


def page_title(title, subtitle=None):
    """The large bold heading at the top of a page. A plain div rather than a
    Markdown heading, so Streamlit adds no anchor-link icon beside it."""
    sub = f'<div class="page-subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
    css = FC_SCALE_CSS if title in FC_SCALE_PAGES else ""
    st.markdown(f'{css}<div class="page-title">{html.escape(title)}</div>{sub}', unsafe_allow_html=True)


def frontier_points(frontier, service_level, legacy_months):
    """The three fill rates the plan is judged on: the forecast-driven plan at the
    build's service level, the flat rule read off at that SAME stock budget, and
    today's rule at its own budget. Shared by Model Accuracy (which also draws the
    frontier chart) and Overview (which only quotes the numbers), so the two pages
    can never disagree."""
    use_value = bool(frontier["Avg inventory value"].notna().all()) and float(frontier["Avg inventory value"].max()) > 0
    vcol = "Avg inventory value" if use_value else "Avg inventory (units)"
    mine = frontier[(frontier["Family"] == FAMILY_MODEL) & np.isclose(frontier["Setting"], service_level)].iloc[0]
    cur = frontier[(frontier["Family"] == FAMILY_FLAT) & np.isclose(frontier["Setting"], legacy_months)].iloc[0]
    flat_same = fill_rate_at_budget(frontier, float(mine[vcol]), value_col=vcol)

    def money(v):
        return short_baht(v) if use_value else f"{v:,.0f} units"

    return dict(use_value=use_value, value_col=vcol, mine=mine, cur=cur,
                flat_same=flat_same, money=money)


def fill_rate_kpis(pts, service_level, legacy_months, replay_year):
    """The three same-budget fill-rate cards (Overview and Model Accuracy)."""
    mine, cur, flat, money, vc = pts["mine"], pts["cur"], pts["flat_same"], pts["money"], pts["value_col"]
    _f1, _f2, _f3 = st.columns(3)
    with _f1:
        _pts = (mine["Fill rate"] - flat) * 100 if pd.notna(flat) else None
        kpi_card(f"Forecast-driven plan · {service_level:.0%} target", f"{mine['Fill rate']:.1%}",
                 delta=_pts,
                 delta_text=(f"{abs(_pts):.1f} pts {'above' if _pts >= 0 else 'below'} the flat rule "
                             f"at this budget" if _pts is not None else None),
                 note=f"of demand shipped from stock · {money(mine[vc])} held · FY{replay_year} replay")
    with _f2:
        kpi_card("Flat rule, same budget", f"{flat:.1%}" if pd.notna(flat) else "--",
                 note=("read off the flat-rule line at the plan's stock level" if pd.notna(flat)
                       else "outside the range the flat rule was tested over"))
    with _f3:
        kpi_card(f"Current rule · {legacy_months:g} months", f"{cur['Fill rate']:.1%}",
                 note=f"of demand shipped from stock · {money(cur[vc])} held")


def render_frontier(frontier, service_level, legacy_months, title, replay_year):
    """KPI row + 'fill rate vs stock held' card for one policy_frontier result,
    drawn on Model Accuracy at the build's inventory parameters. The chart takes
    the card's left side; the returned `text` column on its right is for the
    explanation. Also returns the numbers that explanation quotes."""
    _pts = frontier_points(frontier, service_level, legacy_months)
    use_value, vcol = _pts["use_value"], _pts["value_col"]
    mine, cur, flat_same, money = _pts["mine"], _pts["cur"], _pts["flat_same"], _pts["money"]
    fill_rate_kpis(_pts, service_level, legacy_months, replay_year)

    d = frontier.copy()
    d["Stock"] = d[vcol] / (1e9 if use_value else 1.0)
    x_title = "Average stock held (฿ bn)" if use_value else "Average stock held (units)"
    color = alt.Color("Family:N", scale=alt.Scale(domain=[FAMILY_MODEL, FAMILY_FLAT], range=[ORANGE, TEXT_MUTED]),
                      legend=alt.Legend(title=None, orient="top", labelLimit=360, columnPadding=24))
    # Dashed flat rule: the two lines stay apart without relying on colour alone.
    dash = alt.StrokeDash("Family:N", scale=alt.Scale(domain=[FAMILY_MODEL, FAMILY_FLAT], range=[[1, 0], [6, 4]]),
                          legend=None)
    x = alt.X("Stock:Q", title=x_title, scale=alt.Scale(zero=False))
    y = alt.Y("Fill rate:Q", title="Demand shipped from stock", axis=alt.Axis(format=".0%"), scale=alt.Scale(zero=False))
    tip = [alt.Tooltip("Policy:N"), alt.Tooltip("Fill rate:Q", format=".1%"),
           alt.Tooltip("Stock:Q", title=x_title, format=",.2f" if use_value else ",.0f"),
           alt.Tooltip("Items stocked out:Q", format=",")]
    lines = alt.Chart(d).mark_line(point=True, strokeWidth=2.5).encode(
        x=x, y=y, color=color, strokeDash=dash, order=alt.Order("Stock:Q"), tooltip=tip)
    marks = d[d["Policy"].isin([mine["Policy"], cur["Policy"]])].copy()
    marks["Label"] = np.where(marks["Policy"] == mine["Policy"],
                              f"Plan, {service_level:.0%} target: {mine['Fill rate']:.1%}",
                              f"Current rule, {legacy_months:g} months: {cur['Fill rate']:.1%}")
    points = alt.Chart(marks).mark_point(size=220, filled=True, opacity=1).encode(x=x, y=y, color=color, tooltip=tip)
    # Both lines rise to the right, so the clear space is above-left of the plan
    # point and below-right of the current-rule point (which sits near the y axis,
    # where a left-aligned label would run into the axis title).
    _txt = dict(fontWeight="bold", color=TEXT_MAIN)
    labels = (alt.Chart(marks[marks["Policy"] == mine["Policy"]])
              .mark_text(align="right", dx=-12, dy=-12, **_txt).encode(x=x, y=y, text="Label:N")
              + alt.Chart(marks[marks["Policy"] == cur["Policy"]])
              .mark_text(align="left", dx=12, dy=14, **_txt).encode(x=x, y=y, text="Label:N"))
    layers = [lines, points, labels]
    # The flat rule read off at the plan's budget: a hollow point straight below
    # the plan's, labelled below-right, where both rising lines leave space.
    if pd.notna(flat_same):
        same = pd.DataFrame({"Stock": [d.loc[d["Policy"] == mine["Policy"], "Stock"].iloc[0]],
                             "Fill rate": [flat_same],
                             "Label": [f"Flat rule, same budget: {flat_same:.1%}"]})
        layers += [alt.Chart(same).mark_point(size=200, filled=False, strokeWidth=2.5, color=TEXT_MUTED, opacity=1)
                   .encode(x=x, y=y, tooltip=[alt.Tooltip("Label:N", title="At the plan's budget")]),
                   alt.Chart(same).mark_text(align="left", dx=12, dy=16, **_txt).encode(x=x, y=y, text="Label:N")]
    card = st.container(border=True)
    with card:
        st.markdown(f"##### {title}")
        chart_col, text = st.columns([1.7, 1], gap="large")
        with chart_col:
            st.altair_chart(alt.layer(*layers).properties(height=340), width="stretch")
    return dict(mine=mine, cur=cur, flat_same=flat_same, value_col=vcol, money=money, text=text)


def classical_verdict(bt_u):
    """(best classical method, its WAPE, our WAPE, one-clause verdict) on the catalog
    total, computed from the cross-check table -- on the real data SARIMA edges ours,
    so the wording is never fixed text. None when the table can't say."""
    _ml_name = "ML model (this project)"
    _bt_w = bt_u.set_index("Method")["WAPE"].dropna()
    if _ml_name not in _bt_w.index or len(_bt_w) < 2:
        return None
    _ml_w = float(_bt_w[_ml_name])
    _others = _bt_w.drop(_ml_name)
    _best, _best_w = _others.idxmin(), float(_others.min())
    _gap = (_ml_w - _best_w) * 100  # percentage points; positive = classical is better
    if abs(_gap) < 1:
        _lead, _lag = (_best, "ours") if _gap > 0 else ("ours", _best)
        _verdict = (f"{_lead} edges {_lag} by {abs(_gap):.1f} points ({min(_ml_w, _best_w):.1%} vs "
                    f"{max(_ml_w, _best_w):.1%}) -- effectively a tie")
    elif _gap < 0:
        _verdict = f"our model has the lowest WAPE ({_ml_w:.1%}, next best {_best} at {_best_w:.1%})"
    else:
        _verdict = f"{_best} beats our model ({_best_w:.1%} vs {_ml_w:.1%})"
    return _best, _best_w, _ml_w, _verdict


def trend_chart(actual, forecast, band, granularity, y_axis, y_title, height,
                target_pace=None, target_label="Target pace"):
    """History-vs-forecast chart drawn as the design system's ForecastChart (Overview
    and Product Detail): solid Actual line, dashed Forecast line with hollow points
    joined to the last actual, the p10-p90 band, a "Forecast →" divider, and the
    target pace as a thin dashed line.

    actual / forecast: [FiscalYear, FiscalMonth, Value]; band: [FiscalYear, FiscalMonth,
    lo, hi] or None. target_pace: the value each period has to reach; target_label is
    its hover text. Returns (chart, legend HTML to draw above it)."""
    both = bucket_periods(pd.concat([actual.assign(Series="Actual"), forecast.assign(Series="Forecast")],
                                    ignore_index=True), "FiscalYear", "FiscalMonth", granularity)
    combined = both.groupby(["t", "Period", "FY", "Series"], as_index=False)["Value"].sum().sort_values("t")
    # Explicit period order (history -> forecast), passed to EVERY layer's x encoding so
    # the axis order doesn't depend on which layer Altair reads first -- the band only
    # spans the forecast, so left to itself it would push the forecast to the left.
    order = combined.drop_duplicates("Period")["Period"].tolist()
    x = alt.X("Period:N", sort=order, title=None, axis=period_axis(order, granularity))
    y = alt.Y("Value:Q", title=y_title, axis=y_axis)
    _hist = combined[combined["Series"] == "Actual"]
    _fc = combined[combined["Series"] == "Forecast"]
    # The forecast line starts from the last actual, so the two read as one series.
    lines = pd.concat([combined, _hist.tail(1).assign(Series="Forecast")], ignore_index=True) \
        if len(_hist) and len(_fc) else combined
    series = dict(color=alt.Color("Series:N", scale=alt.Scale(domain=["Actual", "Forecast"], range=[CHART_ACTUAL, ORANGE]),
                                                  legend=None),
                  strokeDash=alt.StrokeDash("Series:N", scale=alt.Scale(domain=["Actual", "Forecast"], range=[[1, 0], [6, 4]]),
                                            legend=None))
    _when = "Quarter" if granularity == "Quarterly" else "Month"
    tip = [alt.Tooltip("Period:N", title=_when), alt.Tooltip("FY:N", title="Fiscal year"), "Series",
           alt.Tooltip("Value:Q", title=y_title, format=",.0f")]
    layers = [alt.Chart(lines).mark_line(strokeWidth=2).encode(x=x, y=y, **series)]
    # Actual dots only on a short series, as the board does; hollow forecast points always.
    if len(_hist) <= 30:
        layers.append(alt.Chart(_hist).mark_circle(size=24, color=CHART_ACTUAL, opacity=1).encode(x=x, y=y))
    layers.append(alt.Chart(_fc).mark_point(size=48, filled=True, fill="white", stroke=ORANGE, strokeWidth=2,
                                            opacity=1).encode(x=x, y=y))
    # Invisible hover targets on every period: a line mark's tooltip reports one point only.
    layers.append(alt.Chart(combined).mark_circle(size=160, opacity=0).encode(x=x, y=y, tooltip=tip))
    # A dashed rule where the forecast starts, labelled, so the boundary doesn't
    # depend on telling navy from orange.
    if len(_fc) and len(_hist):
        _start = pd.DataFrame({"Period": [_fc["Period"].iloc[0]], "Label": ["Forecast →"]})
        layers.append(alt.Chart(_start).mark_rule(color=BORDER_CONTROL, strokeDash=[3, 3]).encode(x=x))
        layers.append(alt.Chart(_start).mark_text(align="left", dx=6, fontSize=12, color=TEXT_MUTED)
                      .encode(x=x, y=alt.value(8), text="Label:N"))
    # The target, spread evenly over the forecast periods -- what each month (or
    # quarter) has to reach for the year to land on it. Named in the legend; the value is on hover.
    if target_pace:
        _pace = pd.DataFrame({"v": [float(target_pace)], "Label": [target_label]})
        layers.append(alt.Chart(_pace).mark_rule(color=TEXT_MAIN, strokeDash=[2, 4], strokeWidth=1).encode(
            y="v:Q", tooltip=[alt.Tooltip("Label:N", title="Target")]))
    has_band = band is not None
    if has_band:
        band = bucket_periods(band, "FiscalYear", "FiscalMonth", granularity)
        band = band.groupby(["t", "Period"], as_index=False)[["lo", "hi"]].sum().sort_values("t")
        layers.insert(0, alt.Chart(band).mark_area(color=ORANGE, opacity=0.15).encode(
            x=x, y="lo:Q", y2="hi:Q",
            tooltip=[alt.Tooltip("Period:N", title=_when), alt.Tooltip("lo:Q", title="p10", format=",.0f"),
                     alt.Tooltip("hi:Q", title="p90", format=",.0f")]))
    return (alt.layer(*layers).resolve_scale(y="shared").properties(height=height)
            .configure_axis(labelFontSize=12, labelColor=TEXT_MUTED, titleColor=TEXT_MUTED, domain=False,
                            ticks=False, gridColor=CHART_GRID)
            # A fixed y-axis width, so the plot starts at the same x on every chart and the
            # legend above (.fc-legend padding) lines up with it.
            .configure_axisY(minExtent=56, maxExtent=56)
            .configure_axisX(grid=False)), chart_legend(has_band, bool(target_pace))


def chart_legend(band, target):
    """The design system's ForecastChart legend as HTML: Altair's own legend can't mix a
    solid line, a dashed line and a shaded box in one row. Range and Target only when drawn."""
    def key(svg, label):
        return (f'<span><svg viewBox="0 0 28 10" aria-hidden="true">{svg}</svg>{label}</span>')
    keys = [key(f'<line x1="0" y1="5" x2="28" y2="5" stroke="{CHART_ACTUAL}" stroke-width="2"/>', "Actual"),
            key(f'<line x1="0" y1="5" x2="28" y2="5" stroke="{ORANGE}" stroke-width="2" stroke-dasharray="6 4"/>',
                "Forecast")]
    if band:
        keys.append(key(f'<rect x="0" y="1" width="28" height="8" fill="{ORANGE}" fill-opacity="0.15"/>',
                        "Range p10–p90"))
    if target:
        keys.append(key(f'<line x1="0" y1="5" x2="28" y2="5" stroke="{TEXT_MAIN}" stroke-width="1" '
                        f'stroke-dasharray="2 4"/>', "Target pace"))
    return f'<div class="fc-legend">{"".join(keys)}</div>'


# ============================= SAVED BUILD ========================================
# A browser reload starts a new Streamlit session, which used to wipe `results` and
# cost a 90-second rebuild mid-demo. The last build is kept in memory and in a file,
# so a reload -- or a server restart -- gets it straight back, on the same page.
# MELFT_LAST_BUILD moves the file; setting it to "" switches all of this off.
LAST_BUILD_PATH = os.environ.get("MELFT_LAST_BUILD", os.path.join(".streamlit", "last_build.pkl"))


def _code_fingerprint():
    """Changes whenever the app or pipeline code does, so an edit never serves a stale build."""
    h = hashlib.sha256()
    for path in [__file__] + [sys.modules[m].__file__ for m in ("forecast_core", "inventory", "classical")]:
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


@st.cache_resource(show_spinner=False)
def _saved_build():
    """The last build, shared by every session of this server ({} if none)."""
    # ponytail: one slot for the whole server -- right for one presenter on a local app;
    # key it per user before hosting the app for several people at once.
    try:
        with open(LAST_BUILD_PATH, "rb") as f:
            saved = pickle.load(f)  # only ever written by _write_saved_build below
        if saved.get("code") == _code_fingerprint():
            return saved
    except Exception:
        pass  # no file, unreadable, or from older code: the next build replaces it
    return {}


def _write_saved_build():
    try:
        os.makedirs(os.path.dirname(LAST_BUILD_PATH) or ".", exist_ok=True)
        with open(LAST_BUILD_PATH + ".tmp", "wb") as f:
            pickle.dump(_saved_build(), f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(LAST_BUILD_PATH + ".tmp", LAST_BUILD_PATH)
    except Exception as e:
        st.toast(f"Couldn't save the build for page reloads: {e}")


def remember_build():
    """After a successful build: this session's build becomes the saved one."""
    st.session_state.pop("_restored_at", None)
    if not LAST_BUILD_PATH:
        return
    saved = _saved_build()
    saved.clear()
    saved.update(
        code=_code_fingerprint(), token=time.time_ns(), built=time.strftime("%d %b %H:%M"),
        state={k: st.session_state[k] for k in ("results", "is_fake", "build_version", "build_config")},
        # The feature and model toggles the build used, the live stocking rules, and page results added
        # later by remember_result. Without the model ones a reloaded session
        # would show Build Forecast's "changes vs the last run" reporting a
        # model change nobody made.
        extras={k: v for k, v in st.session_state.items() if k.startswith(("feat_", "mdl_", "inv_rules"))},
        # The live dict, so targets typed later travel too (to disk on the next write).
        ov_targets=st.session_state.setdefault("_ov_targets", {}),
    )
    st.session_state["_saved_token"] = saved["token"]
    _write_saved_build()


def remember_result(key):
    """Saves a slow page result (walk-forward, ablation, cross-check, workbook) with the build."""
    saved = _saved_build() if LAST_BUILD_PATH else {}
    if saved and saved["token"] == st.session_state.get("_saved_token"):
        saved["extras"][key] = st.session_state[key]
        _write_saved_build()


def restore_build():
    """First run of a new session: pick up the saved build, if there is one."""
    saved = _saved_build() if LAST_BUILD_PATH else {}
    if "results" in st.session_state or not saved or saved["code"] != _code_fingerprint():
        return False
    st.session_state.update(saved["state"])
    st.session_state.update(saved["extras"])
    st.session_state.update(_ov_targets=saved["ov_targets"], _saved_token=saved["token"],
                            _restored_at=saved["built"])
    return True


# ============================= SIDEBAR: NAV ========================================
# st.logo draws into the sidebar header, beside the collapse button. Without the
# image file, fall back to the company name as a text wordmark.
if os.path.exists(LOGO_PATH):
    st.logo(LOGO_PATH, size="large")
else:
    st.sidebar.markdown(f'<div class="sidebar-wordmark">{html.escape(COMPANY_NAME)}</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)

# A restored session returns to the page in the URL, which the radio keeps current.
if restore_build() and st.query_params.get("page") in NAV_ICONS:
    st.session_state["nav_page"] = st.query_params["page"]
if "_redirect_to" in st.session_state:
    st.session_state["nav_page"] = st.session_state.pop("_redirect_to")
# Overview heads the list now, but a fresh session still opens on Build Forecast.
st.session_state.setdefault("nav_page", "Build Forecast")

page = st.sidebar.radio("View", list(NAV_ICONS), format_func=lambda p: f":material/{NAV_ICONS[p]}: {p}",
                        label_visibility="collapsed", key="nav_page", width="stretch")
st.query_params["page"] = page
if st.session_state.pop("_scroll_top", False):
    with st.container(key="plain_scroll_top"):
        st.html("<script>document.querySelector('section.stMain')?.scrollTo(0, 0)</script>",
                unsafe_allow_javascript=True)
if page != "Model Accuracy":
    st.query_params.pop("tab", None)

_glossary = glossary_sections()
if _glossary and st.sidebar.button("Glossary", icon=":material/menu_book:", key="glossary_btn", width="stretch"):
    show_glossary(_glossary)
# Filled once the page knows its product selection (see render_build_card).
_build_card_slot = st.sidebar.container(key="plain_build_card")


def render_build_card(products_line=None, filtered=False, note=None):
    """Sidebar summary of the build every page is showing. note: a muted line
    under the Products row."""
    res = st.session_state.get("results")
    if res is None:
        return
    cfg = st.session_state.get("build_config", {})
    ty = sorted(int(y) for y in cfg.get("train_years", []))
    rows = [
        ("Data", ("Sample data" if st.session_state.get("is_fake") else cfg.get("filename", "Uploaded file"))
         + f" · built {cfg.get('built_at', '--')}" + (" (restored)" if st.session_state.get("_restored_at") else "")),
        ("Model", res[4]),
        ("Years", (f"Trained FY{ty[0]}–{ty[-1]} · " if ty else "")
         + f"validated FY{cfg.get('validation_year', '?')} · forecast FY{cfg.get('forecast_year', '?')}"),
    ]
    if products_line:
        rows.append(("Products", products_line))
    body = "".join(
        f'<div class="build-card-label">{html.escape(k)}</div>'
        f'<div class="build-card-value{" build-card-flag" if k == "Products" and filtered else ""}">'
        f'{html.escape(str(v))}</div>' for k, v in rows)
    if note and products_line:
        body += f'<div class="build-card-note">{html.escape(note)}</div>'
    _build_card_slot.markdown(f'<div class="build-card">{body}</div>', unsafe_allow_html=True)

# ============================= PAGE: BUILD FORECAST ============================
# Laid out to the Forecast Console board: a three-step strip across the top, one
# card per step. The Run step's Review marks what a rebuild would change; the
# board's aside (the same settings again, plus a second Build button) is gone at
# the user's request. Every option the long form had is still here.
#
# The board is a wizard -- one step open, the rest collapsed to a line with an
# Edit button -- and a shut step's widgets are HIDDEN rather than skipped (see
# _STEP_HIDE_CSS with the stylesheet). Behind an `if` they would not run, and
# Streamlit discards the state of a widget a run does not instantiate, while the
# build below reads all three steps' values in this same run; step 1's column
# inputs have no keys at all, so they would not even exist. A header or summary
# that can only be written once its own widgets have run goes into an st.empty()
# slot reserved above them.
if page == "Build Forecast":
    page_title("Build Forecast",
               "Three short steps. The defaults work as they are, so you can go straight to Run.")
    st.markdown("<br>", unsafe_allow_html=True)

    DEFAULT_ITEM_MASTER_LOCAL_PATH = os.path.join("data", "Item_master.xlsx")
    DEFAULT_CUSTOMER_MASTER_LOCAL_PATH = os.path.join("data", "Customer_master.xlsx")

    # One line per model, so the choice is readable without opening the Glossary.
    MODEL_NOTES = {
        "LightGBM (Tweedie)": "Primary. The Tweedie objective fits zero-heavy, intermittent B2B demand.",
        "XGBoost (Tweedie)": "Primary. The Tweedie objective fits zero-heavy, intermittent B2B demand.",
        "HistGradientBoosting": "Fallback with no extra install. Acts as a control.",
        "LightGBM (standard)": "Control: shows that the Tweedie objective is what does the work.",
        "Hurdle (2-stage)": "Control: an explicit intermittent-demand architecture, which doesn't beat Tweedie.",
    }

    def _master_loaded(upload, local_path):
        """Whether a master file is in play, for the step 1 header line. The file
        it came from is named in the expander; spelling it out here wrapped the
        line onto two."""
        if upload is not None:
            return "uploaded"
        return "loaded" if os.path.exists(local_path) else "not loaded"

    def _badge(n, state):
        """The numbered badge: done = teal tick, on = navy number, else grey."""
        cls = {"done": "fc-num fc-num--done", "on": "fc-num fc-num--on"}.get(state, "fc-num")
        return f'<span class="{cls}" aria-hidden="true">{"&#10003;" if state == "done" else n}</span>'

    def _step_header(n, state, title, subtitle, pill=None, tone="neutral", small=False):
        """A step card's header row, returned as HTML for its reserved slot."""
        pill_html = f'<span class="fc-pill fc-pill--{tone}">{html.escape(pill)}</span>' if pill else ""
        tcls = "fc-hdr-title fc-hdr-title--sm" if small else "fc-hdr-title"
        return (f'<div class="fc-hdr">{_badge(n, state)}<div style="flex-grow:1;min-width:0">'
                f'<div class="{tcls}">{html.escape(title)}</div>'
                f'<div class="fc-hdr-sub">{html.escape(subtitle)}</div></div>{pill_html}</div>')

    # Which step is on show. A plain session key, not a widget's, so the Edit
    # and Back/Next callbacks can write it.
    STEP_NAMES = {1: "Data", 2: "Model", 3: "Run"}
    _step = st.session_state.setdefault("_bf_step", 1)

    def _go_step(n):
        st.session_state["_bf_step"] = n

    def _step_slot(n, is_open):
        """The slot a step card's header is written into once the widgets below it
        have run. A shut step puts its Edit button at the right of that one line,
        as on the board; an open one has the card to itself."""
        if is_open:
            return st.empty()
        _row, _edit = st.columns([8, 1.6], vertical_alignment="center")
        _edit.button("Edit", key=f"_bf_edit{n}", on_click=_go_step, args=(n,), width="stretch")
        return _row.empty()

    def _step_nav(prev=None, nxt=None, note=None, build=None):
        """The open step's footer. Real buttons, not the jump links this page had
        before: they move the wizard on rather than scrolling to a card. build:
        the Run step's own Build Forecast button, at the right where Next sits;
        pass whether it is disabled, and the click is returned."""
        with st.container(key="plain_step_nav"):
            _c1, _c2 = st.columns(2, vertical_alignment="center")
            if prev:
                _c1.button(f":material/arrow_back: Back to {STEP_NAMES[prev]}",
                           key=f"_bf_back{prev}", on_click=_go_step, args=(prev,))
            if note:
                _c2.markdown(f'<div class="fc-nav-note">{html.escape(note)}</div>', unsafe_allow_html=True)
            if nxt:
                with _c2.container(key="plain_nav_right"):
                    st.button(f"Next: {STEP_NAMES[nxt]} :material/arrow_forward:",
                              key=f"_bf_next{nxt}", on_click=_go_step, args=(nxt,))
            if build is not None:
                with _c2.container(key="plain_nav_right_build"):
                    return st.button("Build Forecast", type="primary", key="_bf_build_run", disabled=build)

    # Nothing real built yet: step 1 is the step you are on, and the sample-data
    # caption under the uploader still applies.
    _no_real_build = st.session_state.get("results") is None or st.session_state.get("is_fake")

    # The right third stays empty: the steps keep the width they had beside the
    # removed aside (full width read too wide), and a column, unlike a bare
    # container, doesn't turn into one white card around all three steps.
    main_col, _ = st.columns([2.25, 1])
    with main_col:
        _strip = st.empty()   # the strip's states depend on every step below it

        # --- 1. Data ----------------------------------------------------------
        with st.container(border=True):
            _open1 = _step == 1
            _hdr1 = _step_slot(1, _open1)
            with st.container(key="plain_body1"):
                uploaded = st.file_uploader("Upload your sales order data (any filename works -- .xlsx or .csv)", type=["xlsx", "csv"])
                # A build restored in a new session: the uploader is empty, but the
                # pages show that build's file, not sample data. Building now would
                # swap it for sample data, so the Run step asks for the file again.
                _kept_file = (None if uploaded is not None or _no_real_build
                              else st.session_state.get("build_config", {}).get("filename"))
                _data_name = uploaded.name if uploaded is not None else _kept_file or "Sample data"
                is_csv = uploaded is not None and uploaded.name.lower().endswith(".csv")
                if is_csv:
                    st.caption(":material/check_circle: CSV detected -- this loads much faster than .xlsx at this row count.")
                elif uploaded is not None:
                    st.caption(":material/info: If loading feels slow, export this sheet as .csv (File -> Save As -> CSV "
                               "in Excel) and upload that instead -- usually dramatically faster.")
                elif _kept_file:
                    st.caption(f"The pages show the saved build of {_kept_file}. Upload it again to rebuild.")
                elif _no_real_build:
                    st.caption("No file yet -- every page shows sample data until you upload one.")
                # Filled once the column settings below are known: rows, years and any
                # missing column, so a bad file shows up here rather than at build time.
                _file_check = st.container()

                # One expander, not four: the board collapses this step to a single
                # line, and Streamlit forbids expanders inside expanders anyway.
                with st.expander("Data settings: columns and order status", icon=":material/tune:"):
                    st.markdown('<div class="fc-grp">Sales file columns</div>', unsafe_allow_html=True)
                    if is_csv:
                        st.caption("Sheet name doesn't apply to CSV files.")
                        sheet_name = DEFAULT_SHEET_NAME
                    else:
                        st.caption("Sheet TAB name inside the workbook (not the filename).")
                        sheet_name = st.text_input("Excel sheet/tab name", value=DEFAULT_SHEET_NAME)
                    st.caption("The reliable grouping key -- use the internal ID, not the item code/name.")
                    col_product = st.text_input("Product ID column", value=DEFAULT_COL_PRODUCT)
                    col_item_code = st.text_input("Item Code column (display label only, optional)", value=DEFAULT_COL_ITEM_CODE)
                    col_category = st.text_input("Category column in the SALES file (leave blank if using the item master file, or if none)", value=DEFAULT_COL_CATEGORY)
                    col_units = st.text_input("Units column", value=DEFAULT_COL_UNITS)
                    st.caption("Revenue is computed automatically as Base Price x Quantity.")
                    col_price = st.text_input("Base Price column", value=DEFAULT_COL_PRICE)
                    col_date = st.text_input("Date column", value=DEFAULT_COL_DATE)
                    fiscal_start_month = st.number_input("Fiscal year start month (1=Jan, 4=Apr)", min_value=1, max_value=12, value=DEFAULT_FISCAL_START_MONTH)
                    col_status = st.text_input("Order Status column (leave blank to skip filtering)", value=DEFAULT_COL_STATUS)
                    st.caption("Optional: adds a 'distinct customers in the trailing 12 months' feature -- useful for B2B demand patterns.")
                    col_customer = st.text_input("Customer ID column (leave blank to skip)", value="Customer Internal ID")

                    valid_statuses = []
                    _peek_years = [2021, 2022, 2023, 2024]

                    if uploaded is not None:
                        file_bytes_peek = uploaded.getvalue()
                        with st.sidebar:
                            with st.spinner("Reading file -- this is the slow step for large .xlsx files..."):
                                try:
                                    _peek_years, _status_options, _n_bad_dates, _n_rows, _file_cols = peek_file_settings(
                                        file_bytes_peek, uploaded.name, sheet_name, col_date, col_status, int(fiscal_start_month)
                                    )
                                except Exception as e:
                                    _file_check.error(f"Couldn't read the file: {e}")
                                    _status_options, _n_rows = [], None
                        if _n_rows is not None:
                            with _file_check:
                                _missing = [c for c in (col_product, col_item_code, col_category, col_units, col_price,
                                                        col_status, col_customer) if c and c not in _file_cols]
                                if _missing:
                                    st.error(f"Column{'s' if len(_missing) > 1 else ''} not found: "
                                             f"{', '.join(_missing)}. Fix the name in Data settings. "
                                             f"Columns in your file: {', '.join(map(str, _file_cols))}")
                                if not _peek_years:
                                    st.error("No valid dates found in this column -- check the Date column in Data settings.")
                                    _peek_years = [2021, 2022, 2023, 2024]
                                elif not _missing:
                                    st.caption(f":material/check_circle: {_n_rows:,} rows, FY{_peek_years[0]} to "
                                               f"FY{_peek_years[-1]}, every column found.")
                                if _n_bad_dates:
                                    st.warning(f"{_n_bad_dates:,} row(s) have a missing/invalid date and will be excluded.")

                        if col_status and _status_options:
                            st.markdown('<div class="fc-grp">Order status filter</div>', unsafe_allow_html=True)
                            st.caption("Uncheck any statuses that shouldn't count as real sales (e.g. Cancelled, Pending).")
                            valid_statuses = st.multiselect("Include these statuses", _status_options, default=_status_options)

                # The board's "Master files": one small card per file, its state as a
                # pill, and the uploader and column names behind two buttons.
                # A popover's widgets still run while it is shut, so nothing here
                # loses state.
                st.markdown('<div class="fc-sec" style="margin-top:24px">Master files</div>', unsafe_allow_html=True)
                st.caption("Optional. A file saved next to the app under the name shown loads automatically. "
                           "Uploading one here overrides it.")

                def _master_head(slot, title, upload, local_path, what):
                    _state = _master_loaded(upload, local_path)
                    _pill = {"uploaded": ("good", "Uploaded"), "loaded": ("good", "Loaded")}.get(
                        _state, ("neutral", "Not loaded"))
                    _src = (f"{upload.name} · uploaded" if upload is not None
                            else f"{local_path} · loaded automatically" if _state == "loaded"
                            else f"No file. Save one as {local_path} next to the app, or upload it.")
                    slot.markdown(
                        f'<div class="fc-mst-head"><span class="fc-mst-title">{html.escape(title)}</span>'
                        f'<span class="fc-pill fc-pill--{_pill[0]}">{_pill[1]}</span></div>'
                        f'<div class="fc-mst-src">{html.escape(_src)}</div>'
                        f'<div class="fc-mst-what">{html.escape(what)}</div>', unsafe_allow_html=True)

                _mc1, _mc2 = st.columns(2)
                with _mc1, st.container(border=True, key="mastercard_item", height="stretch"):
                    _im_head = st.empty()
                    exclude_discontinued = st.checkbox(
                        "Exclude discontinued products from the forecast (recommended)", value=True)
                    st.markdown('<div class="fc-mst-note">They still appear on the Obsolescence list. '
                                '“To Be Advised” items are always kept.</div>', unsafe_allow_html=True)
                    with st.container(horizontal=True, gap="small", key="plain_mst_btns_item"):
                        with st.popover("Replace file"):
                            item_master_upload = st.file_uploader("Item master file", type=["xlsx", "csv"],
                                                                  key="item_master_upload")
                        with st.popover("Column names", type="tertiary"):
                            item_master_join_col = st.text_input("Product ID column in this file", value=DEFAULT_COL_PRODUCT, key="item_master_join_col")
                            item_master_cat1_col = st.text_input("Category 1 column", value="Product Category 1", key="item_master_cat1")
                            item_master_cat2_col = st.text_input("Category 2 column", value="Product Category 2", key="item_master_cat2")
                            item_master_status_col = st.text_input("Production Status column", value="Production Status", key="item_master_status",
                                                                   help="'Continue' / 'Discontinue'. Drives the obsolescence flags and the option above.")
                            item_master_price_col = st.text_input("List price column", value="List price", key="item_master_price")
                    _master_head(_im_head, "Item master", item_master_upload, DEFAULT_ITEM_MASTER_LOCAL_PATH,
                                 "Adds Category 1 and 2, production status and list price. List price feeds the "
                                 "revenue forecast and the inventory value figures.")
                with _mc2, st.container(border=True, key="mastercard_customer", height="stretch"):
                    _cm_head = st.empty()
                    with st.container(horizontal=True, gap="small", key="plain_mst_btns_customer"):
                        with st.popover("Replace file"):
                            customer_master_upload = st.file_uploader("Customer master file", type=["xlsx", "csv"],
                                                                      key="customer_master_upload")
                        with st.popover("Column names", type="tertiary"):
                            customer_master_join_col = st.text_input("Customer ID column in this file", value="Customer Internal ID", key="cm_join")
                            customer_master_type_col = st.text_input("Customer Type column", value="Customer Type", key="cm_type")
                            customer_master_industry_col = st.text_input("FA Sub Industry column", value="FA Sub Industries", key="cm_ind")
                    _master_head(_cm_head, "Customer master", customer_master_upload, DEFAULT_CUSTOMER_MASTER_LOCAL_PATH,
                                 "Customer type and industry for Customer Detail (and the experimental customer-mix "
                                 "features, off by default). Needs the Customer ID column in Data settings.")

            _hdr1.markdown(_step_header(
                1, "on" if _open1 else "done" if _step > 1 else "todo", "Data",
                "Your sales export and the master files. Saved files load automatically." if _open1
                else f"{uploaded.name if uploaded is not None else f'{_kept_file} (saved build)' if _kept_file else 'No sales file yet'}"
                f"  ·  Item master {_master_loaded(item_master_upload, DEFAULT_ITEM_MASTER_LOCAL_PATH)}"
                f"  ·  Customer master {_master_loaded(customer_master_upload, DEFAULT_CUSTOMER_MASTER_LOCAL_PATH)}",
                pill="File ready" if uploaded is not None else "Saved build" if _kept_file else "Sample data",
                tone="good" if uploaded is not None or _kept_file else "warn",
                small=not _open1), unsafe_allow_html=True)
            if _open1:
                _step_nav(nxt=2)

        # --- 2. Model ---------------------------------------------------------
        # The step the board draws open: years, models and features inline.
        with st.container(border=True):
            _open2 = _step == 2
            _hdr2 = _step_slot(2, _open2)
            with st.container(key="plain_body2"):

                st.markdown('<div class="fc-sec">Fiscal years</div>', unsafe_allow_html=True)
                with st.container(key="fy_tiles"):
                    _y1, _y2, _y3 = st.columns(3)
                _t1, _t2, _t3 = _y1.empty(), _y2.empty(), _y3.empty()
                with st.expander("Change the fiscal years", icon=":material/edit_calendar:"):
                    validation_year = st.selectbox("Validation fiscal year (held out)", _peek_years, index=len(_peek_years) - 1)
                    train_years = st.multiselect("Training fiscal years", [y for y in _peek_years if y != validation_year], default=[y for y in _peek_years if y != validation_year])
                    forecast_year = st.number_input("Forecast fiscal year", value=max(_peek_years) + 1, step=1)
                _ty = sorted(train_years)
                with _t1:
                    profile_card("Train on", f"FY{_ty[0]} to FY{_ty[-1]}" if len(_ty) > 1
                                 else (f"FY{_ty[0]}" if _ty else "No years"), inset=True)
                with _t2:
                    profile_card("Held out to validate", f"FY{validation_year}", inset=True)
                with _t3:
                    profile_card("Forecast", f"FY{int(forecast_year)}", inset=True)

                st.markdown('<div class="fc-sec">Models to compare</div>', unsafe_allow_html=True)
                _model_options = available_model_types()
                _available_model_names = [name for name, avail, _note in _model_options if avail]

                # The three we run by default when everything is installed. The first
                # two need optional libraries (LightGBM / XGBoost);
                # HistGradientBoosting ships with scikit-learn and always works.
                PREFERRED_DEFAULTS = ["LightGBM (Tweedie)", "XGBoost (Tweedie)", "HistGradientBoosting"]

                # Pre-tick the preferred three, but only the ones that are actually
                # importable on this machine. If that leaves fewer than two, top up
                # from whatever else is available so the backtest always has
                # something to compare against.
                default_models = [m for m in PREFERRED_DEFAULTS if m in _available_model_names]
                if len(default_models) < 2:
                    default_models += [m for m in _available_model_names if m not in default_models][:2 - len(default_models)]

                # One bordered row per model, with its note inside, as on the board.
                selected_model_types = []
                for _i, (_name, _avail, _note) in enumerate(_model_options):
                    with st.container(key=f"mdlopt_{_i}"):
                        _ticked = st.checkbox(_name, value=_avail and _name in default_models,
                                              key=f"mdl_{_name}", disabled=not _avail)
                        st.caption(MODEL_NOTES[_name] if _avail
                                   else f"Needs `pip install {_note}` to run." if _note
                                   else "Not available on this machine.")
                    if _ticked and _avail:
                        selected_model_types.append(_name)

                # Keyed, so untick the model it names and the stored value is no longer
                # an option: drop it before the widget is made rather than letting it
                # resolve to nothing.
                _final_options = ["Auto (best WAPE)"] + selected_model_types
                if st.session_state.get("_final_model") not in _final_options:
                    st.session_state.pop("_final_model", None)
                final_model_choice = st.segmented_control(
                    f"Model used for the FY{int(forecast_year)} forecast",
                    _final_options, default="Auto (best WAPE)", key="_final_model",
                    format_func=lambda n: n if n.startswith("Auto") else short_model(n),
                    help="Auto takes whichever of the models above scored the lowest WAPE on the held-out "
                         "year. Pick one by name to override that.",
                ) or "Auto (best WAPE)"

                st.markdown('<div class="fc-sec">Features</div>', unsafe_allow_html=True)
                _feat_note = st.empty()  # the count needs the checkboxes, which come below

                # These buttons MUST render above the checkboxes. A click is handled
                # during this same run, and at this point no feat_* checkbox has been
                # instantiated yet -- the only moment Streamlit still allows writing
                # its key. Placed below the checkboxes this would raise
                # StreamlitAPIException. No st.rerun() is needed.
                # Each at its own width on one line, as on the board.
                with st.container(horizontal=True, key="feat_btns"):
                    if st.button("Select all", key="_feat_btn_all"):
                        for _k in ALL_FEATURE_KEYS:
                            st.session_state[f"feat_{_k}"] = True
                    if st.button("Deselect all", key="_feat_btn_none"):
                        for _k in ALL_FEATURE_KEYS:
                            st.session_state[f"feat_{_k}"] = False
                    if st.button(
                            "Reset to defaults", key="_feat_btn_reset",
                            help="Restores the shipped selection. The experimental group (reorder cadence, "
                                 "customer mix) was flat in the ablation and adds ~27s to a build, so it is "
                                 "off there; customer mix also needs the customer master file."):
                        for _k, _d in FEATURE_DEFAULTS.items():
                            st.session_state[f"feat_{_k}"] = _d

                # The board's two columns, filled row-major: recent demand | item
                # behaviour + seasonality, then product & customer | experimental.
                # Membership is collected in a set and re-ordered through
                # ALL_FEATURE_KEYS, so the tuple handed to build_everything keeps its
                # canonical order whatever the layout does.
                _groups = list(FEATURE_TOGGLE_GROUPS.items())
                _on = set()
                for _col, _idxs in zip(st.columns(2), ([0, 3], [1, 2, 4])):
                    with _col:
                        for _i in _idxs:
                            _gname, _rows = _groups[_i]
                            st.markdown(f'<div class="fc-grp">{html.escape(_gname)}</div>', unsafe_allow_html=True)
                            for _key, _label, _default in _rows:
                                # value= is only the first-run default. Once
                                # feat_<key> exists in session state (a click, or the
                                # buttons above) that value wins and value= is ignored.
                                if st.checkbox(_label, value=_default, key=f"feat_{_key}"):
                                    _on.add(_key)
                feature_toggles = [_k for _k in ALL_FEATURE_KEYS if _k in _on]

                _feat_note.caption(
                    f"**{len(feature_toggles)} of {len(ALL_FEATURE_KEYS)} on.** Every model uses the same set. Each "
                    "feature only looks at earlier months, so nothing from the future leaks in.")
                if not feature_toggles:
                    st.warning("At least one feature must stay on -- a model with no inputs can only predict a "
                               "single constant. **Build Forecast** is disabled until you re-enable one.")

                _n_models = len(selected_model_types)
                _is_default = (selected_model_types == default_models
                               and final_model_choice == "Auto (best WAPE)"
                               and set(feature_toggles) == {_k for _k, _d in FEATURE_DEFAULTS.items() if _d})
            _hdr2.markdown(_step_header(
                2, "on" if _open2 else "done" if _step > 2 else "todo", "Model",
                "Years, models and features. The defaults work as they are." if _open2
                else (f"Trained FY{_ty[0]} to FY{_ty[-1]}" if len(_ty) > 1 else f"FY{_ty[0]}" if _ty else "No years")
                + f"  ·  validated FY{validation_year}  ·  forecast FY{int(forecast_year)}"
                + f"  ·  {_n_models} models compared  ·  {len(feature_toggles)} of {len(ALL_FEATURE_KEYS)} features on",
                pill="Defaults set" if _is_default else "Custom",
                small=not _open2), unsafe_allow_html=True)
            if _open2:
                _step_nav(prev=1, nxt=3)

        # --- 3. Run -----------------------------------------------------------
        with st.container(border=True):
            _open3 = _step == 3
            _hdr3 = _step_slot(3, _open3)
            with st.container(key="plain_body3"):
                # The board's Run card: output options, a recap of every setting with
                # a way back to its step, and the stages the build walks through.
                def _run_option(key, label, value, desc, pill):
                    with st.container(key=f"runopt_{key}", horizontal=True, gap="small"):
                        with st.container(key=f"plain_runopt_txt_{key}", gap=None):
                            _v = st.checkbox(label, value=value, key=key)
                            st.caption(desc)
                        st.markdown(f'<span class="fc-pill fc-pill--neutral">{pill}</span>',
                                    unsafe_allow_html=True, width="content")
                    return _v

                st.markdown('<div class="fc-sec">Forecast output options</div>', unsafe_allow_html=True)
                add_prediction_intervals = _run_option(
                    "_add_pi", "Add prediction intervals (p10 to p90)", True,
                    "Also trains one quantile model per level, so every forecast comes with a low, expected "
                    "and high range. Feeds the interval bands on the charts, the p10 and p90 table columns, "
                    "the p90-based safety stock and the calibration check on Model Accuracy.",
                    "Adds 15 to 20 s")
                apply_bias_corr = _run_option(
                    "_bias", "Correct residual forecast bias", False,
                    "A mean-objective model on mostly-zero data under-forecasts total demand. Rescaling per "
                    "sparsity tier fixes the total but costs some item-level WAPE (Model Accuracy, "
                    "Diagnostics shows how much, either way). Leave off for the most accurate per-item "
                    "forecast. Turn on if the aggregate or budget number matters more.",
                    "Off by default")

                # Stored with the build, so the next visit's Review can say what a
                # rebuild would change. Display strings only -- never read back into a build.
                _settings_now = {
                    "Data": _data_name,
                    "Years": (f"FY{_ty[0]}-{_ty[-1]}" if len(_ty) > 1 else (f"FY{_ty[0]}" if _ty else "none"))
                             + f" -> FY{int(forecast_year)}",
                    "Models": f"{_n_models} compared, "
                              + ("best WAPE" if final_model_choice == "Auto (best WAPE)"
                                 else short_model(final_model_choice)),
                    "Features": f"{len(feature_toggles)} of {len(ALL_FEATURE_KEYS)} on",
                    "Run options": ("p10-p90 on" if add_prediction_intervals else "p10-p90 off")
                                   + (", bias correction on" if apply_bias_corr else ""),
                }
                _prev_cfg = st.session_state.get("build_config", {})
                _prev_settings = _prev_cfg.get("settings") or {}
                _built_at = _prev_cfg.get("built_at", "earlier")
                _changed = {_k for _k, _v in _settings_now.items() if _prev_settings and _prev_settings.get(_k) != _v}
                st.markdown('<div class="fc-sec">Review your settings</div>', unsafe_allow_html=True)
                st.caption("Everything the build will use. Edit any line to go back to that step."
                           + ("" if not _prev_settings
                              else f" {len(_changed)} of {len(_settings_now)} differ from the saved build from "
                                   f"{_built_at}." if _changed
                              else f" All match the saved build from {_built_at}."))
                _im = _master_loaded(item_master_upload, DEFAULT_ITEM_MASTER_LOCAL_PATH)
                _cm = _master_loaded(customer_master_upload, DEFAULT_CUSTOMER_MASTER_LOCAL_PATH)
                _review = [
                    ("Data", 1, _data_name + " · "
                     + (f"Item master and Customer master {_im}" if _im == _cm
                        else f"Item master {_im} · Customer master {_cm}")),
                    ("Years", 2, (f"Train FY{_ty[0]} to FY{_ty[-1]}" if len(_ty) > 1
                                  else f"Train FY{_ty[0]}" if _ty else "No training years")
                     + f" · validate FY{validation_year} · forecast FY{int(forecast_year)}"),
                    ("Models", 2, f"{_n_models} compared · "
                     + ("best WAPE" if final_model_choice == "Auto (best WAPE)" else short_model(final_model_choice))
                     + " drives the forecast"),
                    ("Features", 2, f"{len(feature_toggles)} of {len(ALL_FEATURE_KEYS)} on"),
                    # No Edit here: the output options are the ones just above.
                    ("Output", None, "p10 to p90 " + ("on" if add_prediction_intervals else "off")
                     + " · bias correction " + ("on" if apply_bias_corr else "off")),
                ]
                with st.container(key="plain_review"):
                    for _k, _to, _v in _review:
                        _sk = "Run options" if _k == "Output" else _k
                        _was = ("" if _sk not in _changed else
                                ' <span class="fc-rv-was">&middot; was '
                                + html.escape(str(_prev_settings.get(_sk, "not recorded")).replace(" -> ", " → "))
                                + "</span>")
                        with st.container(key=f"plain_rv_{_k.split()[0]}", horizontal=True,
                                          vertical_alignment="center"):
                            st.markdown(f'<span class="fc-rv-k">{_k}</span>', unsafe_allow_html=True,
                                        width="content")
                            st.markdown(f'<span class="fc-rv-v">{html.escape(_v)}{_was}</span>',
                                        unsafe_allow_html=True, width="stretch")
                            if _to:
                                st.button("Edit", key=f"_bf_rv_{_k.split()[0]}", on_click=_go_step, args=(_to,))
                _use_sample = bool(_kept_file) and st.checkbox(
                    "Build on sample data instead", key="_bf_sample",
                    help=f"Replaces the saved build of {_kept_file} with the sample data on every page.")
            _hdr3.markdown(_step_header(
                3, "on" if _open3 else "todo", "Run",
                "Output options and a last look before you build." if _open3
                else "Prediction intervals p10 to p90 "
                + ("on (adds 15 to 20 seconds)" if add_prediction_intervals else "off")
                + "  ·  bias correction " + ("on" if apply_bias_corr else "off"),
                pill=("Defaults set" if add_prediction_intervals and not apply_bias_corr else "Custom")
                if _open3 else None,
                small=not _open3), unsafe_allow_html=True)
            _run_here = (False if not _open3
                         else _step_nav(prev=2, note=f"Upload {_kept_file} again in Data to rebuild.")
                         if _kept_file and not _use_sample
                         else _step_nav(prev=2, build=not feature_toggles))

        _strip.markdown(
            '<ol class="fc-steps" aria-label="Build steps">'
            + "".join(
                f'<li class="fc-step{" fc-step--on" if _sta == "on" else ""}">{_badge(_i, _sta)}'
                f'<span><span class="fc-step-name">{_nm}</span>'
                f'<span class="fc-step-sub">{_sub}</span></span></li>'
                for _i, (_nm, _sub) in enumerate([
                    ("Data", "File ready" if uploaded is not None else "Saved build" if _kept_file else "Upload your file"),
                    ("Model", "Years, models"),
                    ("Run", "Options, build"),
                ], start=1)
                for _sta in ["on" if _i == _step else "done" if _i < _step else "todo"])
            + "</ol>", unsafe_allow_html=True)

    run_clicked = _run_here

    if "results" not in st.session_state:
        st.session_state.results = None

    if run_clicked or (st.session_state.results is None and uploaded is None):
        file_bytes = uploaded.getvalue() if uploaded is not None else None
        filename = uploaded.name if uploaded is not None else "demo.xlsx"
        status_label = "Building forecast..." if file_bytes is not None else "Loading sample data (upload your file to build a real forecast)..."
        # In the steps column, under step 3, rather than full width beneath the page.
        status_box = main_col.status(status_label, expanded=True)
        with status_box:
            def _on_stage(msg):
                status_box.update(label=f"{msg}...")
                status_box.write(f"{msg}...")
            _build_progress.cb = _on_stage
            try:
                # Safe fallback if somehow nothing is selected: HistGradientBoosting
                # specifically (not just "first available"), since that's the
                # fast, reliable option regardless of what's installed or how
                # many CPU cores are available.
                models_for_build = selected_model_types or ["HistGradientBoosting"]
                item_master_bytes_arg = item_master_upload.getvalue() if item_master_upload is not None else None
                item_master_filename_arg = item_master_upload.name if item_master_upload is not None else None
                if item_master_bytes_arg is None and os.path.exists(DEFAULT_ITEM_MASTER_LOCAL_PATH):
                    with open(DEFAULT_ITEM_MASTER_LOCAL_PATH, "rb") as f:
                        item_master_bytes_arg = f.read()
                    item_master_filename_arg = DEFAULT_ITEM_MASTER_LOCAL_PATH

                cust_master_bytes_arg = customer_master_upload.getvalue() if customer_master_upload is not None else None
                cust_master_filename_arg = customer_master_upload.name if customer_master_upload is not None else None
                if cust_master_bytes_arg is None and os.path.exists(DEFAULT_CUSTOMER_MASTER_LOCAL_PATH):
                    with open(DEFAULT_CUSTOMER_MASTER_LOCAL_PATH, "rb") as f:
                        cust_master_bytes_arg = f.read()
                    cust_master_filename_arg = DEFAULT_CUSTOMER_MASTER_LOCAL_PATH

                st.session_state.results = build_everything(
                    file_bytes, filename, sheet_name, col_product, col_item_code or None, col_category or None, col_units, col_price, col_date,
                    col_status or None, tuple(valid_statuses) if valid_statuses else None,
                    int(fiscal_start_month), tuple(sorted(train_years)), validation_year, int(forecast_year),
                    tuple(models_for_build), final_model_choice, col_customer or None,
                    item_master_bytes_arg, item_master_filename_arg, item_master_join_col, item_master_cat1_col, item_master_cat2_col,
                    item_master_status_col or "Production Status", item_master_price_col or "List price",
                    cust_master_bytes_arg, cust_master_filename_arg, customer_master_join_col or "Customer Internal ID",
                    customer_master_type_col or "Customer Type", customer_master_industry_col or "FA Sub Industries",
                    bool(exclude_discontinued),
                    # Always a tuple: an empty selection must stay empty. It used
                    # to collapse to None, which _use() reads as "use EVERYTHING".
                    tuple(feature_toggles),
                    bool(add_prediction_intervals), bool(apply_bias_corr),
                )
                _build_progress.cb = None
                st.session_state.is_fake = file_bytes is None
                # Bump a version counter so the product-selector widget below gets
                # a brand-new key on this run. Just popping the old key's state
                # isn't always enough -- Streamlit's frontend widget component can
                # retain its own state within the same run, so old selections
                # (e.g. demo "ITEM-0001") could still show even after the pop.
                # A new key forces Streamlit to treat it as an entirely new
                # widget, guaranteeing the fresh default is used.
                st.session_state["build_version"] = st.session_state.get("build_version", 0) + 1
                st.session_state["build_config"] = {
                    "col_product": col_product, "fiscal_start_month": int(fiscal_start_month),
                    "forecast_year": int(forecast_year), "validation_year": validation_year, "train_years": train_years,
                    "filename": filename if file_bytes is not None else "Sample data",
                    "built_at": time.strftime("%d %b %H:%M"),
                    # What the next visit's Review diffs against.
                    "settings": dict(_settings_now),
                }
                remember_build()
                if file_bytes is not None:
                    status_box.update(label="Forecast built successfully.", state="complete", expanded=False)
                    # Signal a redirect via a SEPARATE flag rather than
                    # setting st.session_state["nav_page"] directly here --
                    # that key's widget (the sidebar radio) has already
                    # rendered earlier in this same run, and Streamlit
                    # disallows modifying a widget's own state after it's
                    # been instantiated in the current run. The flag is
                    # picked up and applied at the very top of the next run,
                    # before the radio widget renders.
                    st.session_state["_redirect_to"] = "Overview"
                    st.rerun()
                else:
                    status_box.update(label="Sample data loaded -- upload your file above to build a real forecast.", state="complete", expanded=False)
            except Exception as e:
                _build_progress.cb = None
                status_box.update(label="Failed.", state="error", expanded=True)
                st.error(f"Couldn't build the forecast: {e}")
                st.stop()


    if st.session_state.get("results") is not None:
        _restored = st.session_state.get("_restored_at")
        main_col.success("Forecast ready -- pick any page in the sidebar to explore it."
                        + (f" Restored from the build made {_restored}; upload a file and click "
                           "**Build Forecast** to replace it." if _restored else ""))
        render_build_card()
    st.stop()

if st.session_state.get("results") is None:
    empty_state("**No forecast yet.** Upload your sales data and build a forecast first -- every page "
                "reads from that build.", button="Go to Build Forecast")
    st.stop()

(history, forecast, res_df, metrics, model_name, n_dropped_dates, skipped_models, importance_df,
 importance_method, eval_report, inv_plan, obs_flags, build_meta) = st.session_state.results
_cfg = st.session_state.get("build_config", {})
col_product = _cfg.get("col_product")
fiscal_start_month = _cfg.get("fiscal_start_month")
forecast_year = _cfg.get("forecast_year")
validation_year = _cfg.get("validation_year")
train_years = _cfg.get("train_years")
# The stocking rules are set live on Inventory Plan, not in the build: every
# page below reads the plan and rules for the live ones.
inv_plan, obs_flags, build_meta = live_inventory(st.session_state.results)
# A rule between the build summary card and these build notes -- without one the
# "discontinued products excluded" caption reads as the card's last line.
if (skipped_models or n_dropped_dates or build_meta.get("n_discontinued_excluded")
        or build_meta.get("n_duplicate_lines") or build_meta.get("n_unpriced_items")
        or build_meta.get("unavailable_features") or st.session_state.get("is_fake")):
    st.sidebar.divider()
if skipped_models:
    skip_msg = "; ".join(f"{name} ({reason})" for name, reason in skipped_models)
    st.sidebar.caption(f"Skipped (not installed): {skip_msg}")
if n_dropped_dates:
    st.sidebar.warning(f"{n_dropped_dates:,} row(s) had a missing/invalid date and were excluded from the forecast.")
if build_meta.get("n_duplicate_lines"):
    st.sidebar.caption(f"{build_meta['n_duplicate_lines']:,} exact duplicate sales line(s) removed.")
if build_meta.get("n_unpriced_items"):
    st.sidebar.caption(f"{build_meta['n_unpriced_items']:,} product(s) with no list price in the item master removed.")
if build_meta.get("n_discontinued_excluded"):
    st.sidebar.caption(f"{build_meta['n_discontinued_excluded']:,} discontinued product(s) excluded from the forecast (still on the Obsolescence page).")
if build_meta.get("unavailable_features"):
    st.sidebar.caption("Selected but not available in this data, so skipped: "
                       + ", ".join(_pretty_label(c) for c in build_meta["unavailable_features"]))
is_fake = st.session_state.get("is_fake", False)
if is_fake:
    st.sidebar.warning("Showing sample data -- upload your real file above.")

all_products = sorted(forecast[col_product].unique())
# Item Code comes from `history`, NOT `forecast`: forecast_future builds its
# output frame from an explicit column list (product, year, month, the forecast
# columns and the quantiles) and carries no display attributes, so the old
# `if "Item Code" in forecast.columns` test never fired and code_lookup was
# silently empty on real data. That quietly disabled the product filter's
# search-products-by-code, the Product Detail focus search, and the Item Code
# column on the Category Plan error table.
code_lookup = {}
for _src in (forecast, history):
    if "Item Code" in _src.columns:
        code_lookup = _src.drop_duplicates(col_product).set_index(col_product)["Item Code"].to_dict()
        break

# Vega's category20c, spelled out so a chart that shows only some categories
# plus an "Other" stack can give each one the colour it has everywhere else.
CATEGORY20C = ["#3182bd", "#6baed6", "#9ecae1", "#c6dbef", "#e6550d", "#fd8d3c", "#fdae6b", "#fdd0a2",
               "#31a354", "#74c476", "#a1d99b", "#c7e9c0", "#756bb1", "#9e9ac8", "#bcbddc", "#dadaeb",
               "#636363", "#969696", "#bdbdbd", "#d9d9d9"]


def category_color_scale(col, keep=None, other=None):
    """One fixed colour per category value, shared by every chart that colours by
    `col`. Without a pinned domain Vega assigns colours in the order each chart's
    own data happens to list them, so PLC was green on Category Plan and orange
    on Customer Detail. With `keep`, the scale holds only those categories (each
    in its usual colour) plus an `other` label in a neutral grey."""
    values = set(level_map_from(history, col_product, col).values())
    cch = build_meta.get("cust_cat_history")
    if cch is not None and col in cch.columns:
        values |= set(cch[col].astype(str))
    dom = sorted(values)
    if keep is None:
        return alt.Scale(domain=dom, scheme="category20c")
    rng = [CATEGORY20C[dom.index(c) % 20] if c in dom else CATEGORY20C[16] for c in keep]
    return alt.Scale(domain=list(keep) + ([other] if other else []), range=rng + (["#B8B5AC"] if other else []))


PRODUCT_KEY = f"products_multiselect_v{st.session_state.get('build_version', 0)}"
default_selection = all_products  # all products shown/selected by default
fmonth_labels = fiscal_month_labels(int(fiscal_start_month))  # needed on every page

# The product selector (and everything derived from it) is skipped on pages
# that don't filter by product selection -- Model Accuracy works from the
# full-catalog backtest, Customer Detail is keyed on customer, and Category
# Plan is a whole-catalog S&OP roll-up (a category total filtered to a
# product subset isn't a plan). Rendering a multiselect of thousands of chips
# on those pages was wasted cost.
uses_product_filter = page not in ("Model Accuracy", "Customer Detail", "Category Plan")
# On the whole-catalog pages the Products row says why a Scope set elsewhere is
# not applied here, so it doesn't read as a lost selection.
_kept_n = st.session_state.get(PRODUCT_KEY + "_n", len(all_products))
_catalog_note = (f"Your Scope ({_kept_n:,} of {len(all_products):,}) still applies on Overview, "
                 "Product Detail, Forecast Table and Inventory Plan."
                 if _kept_n < len(all_products)
                 else "This page doesn't use the Scope filter.")
_subtitle = f"Mitsubishi Electric Factory Automation (Thailand)  ·  FY{int(forecast_year)} forecast  ·  {model_name}"
if uses_product_filter:
    # The filter sits in a popover at the right of the title row, keeping the
    # sidebar to navigation only. Widgets inside a popover are rendered on every
    # run (merely hidden while it is closed), so the selection persists exactly
    # as it did in the sidebar.
    title_col, filter_col = st.columns([3, 1], vertical_alignment="bottom")
    with title_col:
        page_title(page, _subtitle)
    # The Scope is global: Streamlit drops the multiselect's state on pages that
    # don't draw it, so its last value is kept under a plain key and put back.
    _scope_kept = PRODUCT_KEY + "_kept"
    # Seeded here rather than via default=, which would clash with the Session State API.
    if PRODUCT_KEY not in st.session_state:
        st.session_state[PRODUCT_KEY] = list(st.session_state.get(_scope_kept, default_selection))
    already_selected = st.session_state[PRODUCT_KEY]
    # Category 1 / Category 2 narrow the Scope on top of the product list (empty =
    # all). Kept like the products, and a stale value (a rebuild on another file)
    # is dropped before the widget sees it.
    _scope_cats = [c for c in ("Category1", "Category2") if c in history.columns]
    _cat_maps = {c: level_map_from(history, col_product, c) for c in _scope_cats}
    _cat_opts = {c: sorted(set(_cat_maps[c].values())) for c in _scope_cats}
    for c in _scope_cats:
        _k = f"scope_{c}"
        if _k not in st.session_state:
            st.session_state[_k] = [v for v in st.session_state.get(_k + "_kept", []) if v in _cat_opts[c]]
    if "Category1" in _cat_maps and "Category2" in _cat_maps and st.session_state["scope_Category1"]:
        _c1 = set(st.session_state["scope_Category1"])
        _cat_opts["Category2"] = sorted({v for p, v in _cat_maps["Category2"].items() if _cat_maps["Category1"][p] in _c1})
        st.session_state["scope_Category2"] = [v for v in st.session_state["scope_Category2"] if v in _cat_opts["Category2"]]

    def _in_cats(prods):
        for c in _scope_cats:
            if st.session_state[f"scope_{c}"]:
                keep = set(st.session_state[f"scope_{c}"])
                prods = [p for p in prods if _cat_maps[c].get(p, UNKNOWN_LEVEL) in keep]
        return prods

    _n_scope = len(_in_cats(already_selected))
    _scope_label = (f"Scope: all {len(all_products):,} products" if _n_scope >= len(all_products)
                    else f"Scope: {_n_scope:,} of {len(all_products):,} products")
    with filter_col, st.popover(_scope_label, icon=":material/filter_list:", width="stretch",
                                key="product_filter_popover"):
        for c in _scope_cats:
            st.multiselect(_pretty_label(c), _cat_opts[c], key=f"scope_{c}", placeholder="All")
        search_term = st.text_input("Search products (ID or code)", placeholder="Type to filter, e.g. 6199 or FGN")
        if search_term:
            s = search_term.lower()
            matches = [p for p in all_products if s in p.lower() or s in str(code_lookup.get(p, "")).lower()]
            if not matches:
                st.caption("No matches -- showing your current selection only.")
        else:
            matches = all_products

        options_for_widget = sorted(set(matches) | set(already_selected))
        selected_products = st.multiselect("Products to display", options_for_widget, key=PRODUCT_KEY)
    st.session_state[_scope_kept] = selected_products
    for c in _scope_cats:
        st.session_state[f"scope_{c}_kept"] = st.session_state[f"scope_{c}"]
    selected_products = _in_cats(selected_products)
    _n_sel = len(selected_products)
    st.session_state[PRODUCT_KEY + "_n"] = _n_sel
    render_build_card(f"All {_n_sel:,}" if _n_sel >= len(all_products)
                      else f"{_n_sel:,} of {len(all_products):,} selected (Scope)",
                      filtered=_n_sel < len(all_products))
    if not selected_products:
        empty_state("**No products selected.** Pick at least one product, or a category that has some, in the "
                    "**Scope** control at the top right.")
        st.stop()

    hist_f = history[history[col_product].isin(selected_products)]
    fc_f = forecast[forecast[col_product].isin(selected_products)]

    total_forecast_units = fc_f["Final_Forecast"].sum()
    latest_year = hist_f["FiscalYear"].max()
    last_year_units = hist_f[hist_f["FiscalYear"] == latest_year]["Units"].sum()
    yoy_growth = (total_forecast_units / last_year_units - 1) if last_year_units else 0

    ml_wape = baseline_wape = None
    if metrics is not None:
        wape_row = metrics[metrics["Metric"].str.contains("WAPE")]
        if len(wape_row) and model_name in metrics.columns:
            ml_wape, baseline_wape = wape_row[model_name].values[0], wape_row["Baseline"].values[0]
elif page == "Model Accuracy":
    # The board's Section switch at the right of the title picks which section is
    # drawn (only that one is). Streamlit drops the widget's state when the page
    # changes, so returning here opens the section named in the URL (?tab=), or
    # the first. go_to() writes the key from other pages' buttons.
    if st.session_state.get("acc_tab") not in ACC_SECTIONS:
        _qt = st.query_params.get("tab")
        st.session_state["acc_tab"] = _qt if _qt in ACC_SECTIONS else ACC_SECTIONS[0]
    title_col, sec_col = st.columns([1.6, 1], vertical_alignment="bottom")
    with title_col:
        # Six cards with no order of their own, so this section's subtitle says how
        # to read them (a bare caption would sit on the grey page). The build's
        # model and year are on the sidebar card.
        page_title(page, "Read top to bottom: the headline scores, which model won and how each missed, then "
                         "how it holds up 12 months ahead, at coarser grains and year to year."
                   if st.session_state["acc_tab"] == "Model accuracy" else _subtitle)
    with sec_col:
        st.segmented_control("Section", ACC_SECTIONS, required=True, key="acc_tab",
                             label_visibility="collapsed", width="stretch")
    st.query_params["tab"] = st.session_state["acc_tab"]
    render_build_card("Whole catalog on this page", note=_catalog_note)
elif page == "Category Plan":
    # The board's header slot holds the Scope button; this page is a whole-catalog
    # roll-up, so the slot holds the two switches that drive every number on it.
    # A rebuild can drop a category column, so a stale stored grain is dropped
    # (pop, not assign) and the control's default takes over.
    _cp_levels = [c for c in ("Category1", "Category2", "Category") if c in history.columns]
    if st.session_state.get("cat_level") not in _cp_levels:
        st.session_state.pop("cat_level", None)
    title_col, sw_col = st.columns([1.6, 1], vertical_alignment="bottom")
    with title_col:
        page_title(page, _subtitle)
    if _cp_levels:
        with sw_col, st.container(horizontal=True, horizontal_alignment="right"):
            st.segmented_control("Grain", _cp_levels, default=_cp_levels[0], required=True, key="cat_level",
                                 label_visibility="collapsed", format_func=_pretty_label)
            st.segmented_control("Show", ["Units", "Revenue"], default="Units", required=True, key="cat_metric",
                                 label_visibility="collapsed")
    render_build_card("Whole catalog on this page", note=_catalog_note)
else:
    page_title(page, _subtitle)
    render_build_card("Whole catalog on this page", note=_catalog_note)
st.markdown("<br>", unsafe_allow_html=True)

# ============================= PAGE: OVERVIEW =================================
# Laid out to the Forecast Console board: the year against its target, the trend
# that produced it, what needs acting on, and the decision the whole project is
# judged by. The Summary-by-Product table lives on Forecast Table, the build
# details in the sidebar card, and the ETS/SARIMA cross-check on Model Accuracy
# -> Diagnostics, so this page carries only what a planner opens it for.
if page == "Overview":
    # --- FY target bar ----------------------------------------------------------
    # The card has its own Units/Revenue switch. It is drawn inside the card, after
    # the totals below are worked out -- but session_state already holds the value
    # the user just switched to, so the bar is in sync from the same rerun.
    _tgt_rev = st.session_state.get("ov_target_measure", "Units") == "Revenue"
    _tgt_price = price_per_item(history, col_product) if _tgt_rev else None

    def _ov_total(col):
        """One forecast column's year total, in the measure the page is showing."""
        if col not in fc_f.columns:
            return None
        s = fc_f[col].astype(float)
        if _tgt_rev:
            s = s * fc_f[col_product].map(_tgt_price).fillna(0.0)
        return float(s.sum())

    def _ov_amt(v):
        return short_baht(v) if _tgt_rev else f"{v:,.0f}"

    _tgt_forecast = _ov_total("Final_Forecast") if _tgt_rev else float(total_forecast_units)
    _tgt_lo, _tgt_hi = _ov_total("q10"), _ov_total("q90")

    # A target is a planning input, not a build output, so nothing here is keyed
    # on build_version -- it survives a rebuild like the feat_* toggles. One value
    # is kept per measure so switching Units<->Revenue never reinterprets a number
    # typed in the other unit. The real copy lives in a plain dict because
    # Streamlit DISCARDS a widget's session-state entry on any run where the
    # widget isn't rendered, and a measure switch renders only the other input.
    _tgt_store = st.session_state.setdefault("_ov_targets", {})
    _tgt_measure = "revenue" if _tgt_rev else "units"
    _tgt_key = f"ov_target_{_tgt_measure}"
    # Default targets until the user types their own (a cleared input stays cleared).
    _tgt_store.setdefault(_tgt_measure, 1_500_000_000.0 if _tgt_rev else 150_000.0)
    if _tgt_key not in st.session_state and _tgt_store.get(_tgt_measure) is not None:
        st.session_state[_tgt_key] = _tgt_store[_tgt_measure]

    with st.container(border=True, key="ov_target_card"):
        _head_col, _set_col = st.columns([2.3, 1], vertical_alignment="top")
        with _set_col, st.container(horizontal=True, horizontal_alignment="right"):
            st.segmented_control("Target measure", ["Units", "Revenue"], default="Units", required=True,
                                 key="ov_target_measure", label_visibility="collapsed")
        # The board's TargetBar: the input and its Set target button sit under the bar.
        # The bar reads the target, so it is drawn into slots filled after the input.
        _head_slot, _bar_slot, _note_slot = _head_col.empty(), st.empty(), st.empty()
        with st.container(horizontal=True, vertical_alignment="bottom", key="plain_ov_target_set"):
            # The board hides this input once a target is set; here it stays, so the
            # figure can be changed in front of an audience without clearing it first.
            _target = st.number_input(
                f"FY{int(forecast_year)} target ({'฿' if _tgt_rev else 'units'})",
                min_value=0.0, value=None,
                step=1_000_000.0 if _tgt_rev else 1_000.0, format="%.0f",
                key=_tgt_key, placeholder=f"Enter FY{int(forecast_year)} target",
                label_visibility="collapsed", width=240)
            # The value is applied as the input loses focus; the click only reruns.
            st.button("Set target", key="_ov_set_target", help=(
                "MELFT's own sales plan for the year, if one exists. The bar then shows how far "
                "this forecast sits from it. Read a large gap against the model's known "
                "under-forecast bias (Model Accuracy) before treating it as a business signal."))
        _tgt_store[_tgt_measure] = _target

        _head = (f'<div class="kpi-label">FY{int(forecast_year)} forecast '
                 f'({"revenue" if _tgt_rev else "units"}), against your target</div>'
                 f'<div class="tgt-value">{_ov_amt(_tgt_forecast)}</div>')
        if _target:
            _gap = _tgt_forecast - float(_target)
            _gp = _gap / float(_target)
            # Tone thresholds from the design system's TargetBar: at or above the
            # target is good, within 5% of it a warning, further below is bad.
            _tone, _word = (("good", "On or above target") if _gp >= 0
                            else ("warn", "Near target") if _gp > -0.05
                            else ("bad", "Below target"))
            _head += (f'<div class="tgt-status">'
                      f'<span class="fc-pill fc-pill--{_tone}">{_word}: '
                      f'{"+" if _gap >= 0 else "&minus;"}{abs(_gp) * 100:.1f}%</span>'
                      f'<span class="kpi-note">Gap to target {_ov_amt(abs(_gap))}</span></div>')
        _head_slot.markdown(_head, unsafe_allow_html=True)

        # The track: the forecast as a filled bar, its p10-p90 range as a band over
        # it, the target as a dashed tick. Everything is a percentage of one scale.
        _scale = max(_tgt_hi or 0.0, float(_target or 0.0), _tgt_forecast) * 1.08 or 1.0

        def _tgt_pc(v):
            return f"{max(0.0, min(100.0, v / _scale * 100.0)):.3f}%"

        _bar = [f'<div class="tgt-fill" style="width:{_tgt_pc(_tgt_forecast)}"></div>']
        if _tgt_lo is not None and _tgt_hi is not None and _tgt_hi > _tgt_lo:
            _bar.append(f'<div class="tgt-range" style="left:{_tgt_pc(_tgt_lo)}; '
                        f'width:calc({_tgt_pc(_tgt_hi)} - {_tgt_pc(_tgt_lo)})"></div>')
        if _target:
            _bar.append(f'<div class="tgt-tick" style="left:{_tgt_pc(float(_target))}">'
                        f'<span>Target {_ov_amt(float(_target))}</span></div>')
        _bar_slot.markdown(f'<div class="tgt-track">{"".join(_bar)}</div>', unsafe_allow_html=True)

        if _tgt_lo is not None and _tgt_hi is not None:
            _note_slot.caption(f"Forecast range (p10-p90): {_ov_amt(_tgt_lo)} to {_ov_amt(_tgt_hi)}.")

    # --- The trend behind that number -------------------------------------------
    with st.container(border=True):
        hcol, ccol = st.columns([2.6, 2.15])
        # Both switches together at the right of the title, as on the board.
        with ccol, st.container(horizontal=True, horizontal_alignment="right"):
            trend_metric = st.segmented_control("Show", ["Units", "Revenue"], default="Units", required=True, key="trend_metric", label_visibility="collapsed",
                                                on_change=lambda: st.session_state.update(ov_target_measure=st.session_state["trend_metric"]))
            trend_gran = st.segmented_control("Granularity", ["Monthly", "Quarterly"], default="Monthly", required=True, key="trend_gran", label_visibility="collapsed")
        with hcol:
            # Titled by the measure on show, like the board, rather than a fixed
            # "History vs Forecast" that repeats what the legend already says.
            st.markdown(f"##### {trend_metric}: History and Forecast")

        try:
            as_revenue = trend_metric == "Revenue"
            price = price_per_item(history, col_product) if as_revenue else None
            y_title = "Revenue (฿)" if as_revenue else "Units"

            # History revenue uses the real Revenue column; forecast revenue is
            # forecast units x each item's average realised price. Both frames
            # carry FiscalYear + FiscalMonth so bucket_periods() can roll them
            # up to quarters when asked.
            h = monthly_series(hist_f, ["FiscalYear", "FiscalMonth"], "Revenue" if as_revenue else "Units")
            f = monthly_series(fc_f, ["FiscalMonth"], "Final_Forecast", price, col_product)
            f["FiscalYear"] = int(forecast_year)

            # p10-p90 band over the forecast horizon (shaded area under the line).
            band_df = None
            if {"q10", "q90"}.issubset(fc_f.columns):
                blo = monthly_series(fc_f, ["FiscalMonth"], "q10", price, col_product).rename(columns={"Value": "lo"})
                bhi = monthly_series(fc_f, ["FiscalMonth"], "q90", price, col_product).rename(columns={"Value": "hi"})
                band_df = blo.merge(bhi, on="FiscalMonth")
                band_df["FiscalYear"] = int(forecast_year)
            # The target spread evenly over the forecast year, so the chart shows what
            # each period has to reach. Only when the target is in the measure on show.
            _pace = (float(_target) / (4 if trend_gran == "Quarterly" else 12)
                     if _target and as_revenue == _tgt_rev else None)
            chart, legend = trend_chart(h, f, band_df, trend_gran,
                                        revenue_axis() if as_revenue else alt.Undefined, y_title, 340,
                                        target_pace=_pace, target_label=(
                                            f"{_ov_amt(_pace)} per "
                                            f"{'quarter' if trend_gran == 'Quarterly' else 'month'}" if _pace else ""))
            st.markdown(legend, unsafe_allow_html=True)
            st.altair_chart(chart, width="stretch")
            # Method out of the way, as the board's "How is this calculated?" does.
            with st.expander("How is this calculated?"):
                st.caption(
                    f"Forecast units come from {model_name}, trained on FY{', FY'.join(str(y) for y in train_years)} "
                    f"and validated on FY{int(validation_year)}. Revenue multiplies each item's forecast units by its "
                    f"average realised price (past Revenue / past Units), or its list price where an item has no "
                    f"sales history -- so it is revenue at today's prices, not a price forecast.")
        except Exception as e:
            st.error(f"Couldn't build the trend chart: {e}")

    # --- The #1 product and #1 customer, each a way into its detail page, where the
    # full top 10 sits in the controls card. Same measure as the chart above.
    _t_rev = trend_metric == "Revenue"
    _t_amt = (lambda v: short_baht(v)) if _t_rev else (lambda v: f"{v:,.0f} units")
    _t_cards = []
    _tp, _tp_tot = top_products(hist_f, fc_f, col_product, trend_metric, latest_year)
    if len(_tp) and _tp_tot > 0:
        _p1 = _tp.index[0]
        _p1_code = str(code_lookup.get(_p1, "") or _p1)
        _p1_chg = (f", {_tp['Forecast'].iloc[0] / _tp['Actual'].iloc[0] - 1:+.0%} vs FY{int(latest_year)}"
                   if _tp["Actual"].iloc[0] > 0 else "")
        _t_cards.append(("Top product", _p1_code, f"{_t_amt(_tp['Forecast'].iloc[0])} forecast for FY{int(forecast_year)}",
                         f"#1 of {fc_f[col_product].nunique():,} products in the Scope{_p1_chg}. The top 10 are "
                         f"{_tp['Forecast'].sum() / _tp_tot:.0%} of the forecast.",
                         "Open Product Detail", ("Product Detail", "pd_focus", _p1)))
    _ov_cch, _ov_ccol = build_meta.get("cust_cat_history"), build_meta.get("col_customer")
    if _ov_cch is not None and _ov_ccol is not None and len(_ov_cch):
        _ov_cfy = int(_ov_cch["FiscalYear"].max())
        _tc, _tc_tot = top_customers(_ov_cch, _ov_ccol, trend_metric, _ov_cfy)
        if len(_tc) and _tc_tot > 0:
            _c1 = _tc.index[0]
            _c1_nm = str((build_meta.get("cust_name_map") or {}).get(_c1, "")).strip()
            _c1_nm = _c1_nm if _c1_nm and _c1_nm.lower() != "nan" else str(_c1)
            _t_cards.append(("Top customer", _c1_nm, f"{_t_amt(_tc['FY'].iloc[0])} bought in FY{_ov_cfy}",
                             f"#1 of {_ov_cch[_ov_ccol].nunique():,} customers (all of them: the Scope doesn't apply "
                             f"to customers). The top 10 are {_tc['FY'].sum() / _tc_tot:.0%} of FY{_ov_cfy}.",
                             "Open Customer Detail", ("Customer Detail", "cd_focus", _c1)))
    for _i, (_col, (_tag, _count, _title, _body, _btn, _args)) in enumerate(
            zip(st.columns(2) if _t_cards else [], _t_cards)):
        with _col, st.container(border=True, height="stretch", key=f"ov_top_{_i}"):
            st.markdown(f'<span class="fc-pill fc-pill--neutral">{html.escape(_tag)}</span>'
                        f'<div class="attn-count">{html.escape(_count)}</div>'
                        f'<div class="attn-title">{html.escape(_title)}</div>'
                        f'<div class="attn-body">{html.escape(_body)}</div>', unsafe_allow_html=True)
            st.button(_btn, key=f"_ov_top_{_i}", on_click=open_item, args=_args, icon=":material/arrow_forward:")

    # --- What needs doing: each finding with a way to the page that acts on it.
    # Counted exactly as Inventory Plan counts them, on the same product selection.
    _attn = []
    if inv_plan is not None:
        _ap = inv_plan[inv_plan[col_product].isin(selected_products)
                       & (inv_plan["forecast_monthly_units"] > 0.5)]
        _under = int((_ap["safety_stock_delta_vs_legacy"].astype(float) > 0).sum())
        _legacy_m = build_meta.get("inv_params", {}).get("legacy", 1.5)
        _attn.append(("bad", "Stock-out risk", f"{_under:,}", "items under-stocked",
                      f"Below the forecast-driven safety stock under today's flat {_legacy_m:g}-month rule, "
                      f"out of {len(_ap):,} items with forecast demand.",
                      "Open Inventory Plan", "Inventory Plan", None))
    if obs_flags is not None and len(obs_flags):
        _of = obs_flags[obs_flags[col_product].isin(set(selected_products))
                        | ~obs_flags[col_product].isin(set(all_products))]
        _oval = (short_baht(_of["value_at_risk_thb"].sum())
                 if "value_at_risk_thb" in _of.columns else f"{len(_of):,}")
        _attn.append(("warn", "Obsolescence", _oval, "at obsolescence risk",
                      f"{len(_of):,} items that are discontinued, dormant or declining, valued at list price. "
                      f"Stop reordering them.",
                      "Open obsolescence list", "Inventory Plan", None))
    if last_year_units:
        # The raw growth figure is dominated by the model's known low lean, so lead with
        # the bias-corrected reading. The honest backtest measures that lean at both
        # horizons the forecast is used at, which brackets the answer; without it, fall
        # back to the single held-out aggregate bias.
        _hz_tbl = (build_meta.get("honest") or {}).get("table")
        _biases = []
        try:
            if _hz_tbl is not None and "Bias" in _hz_tbl.index:
                _biases = [float(_hz_tbl.loc["Bias", c]) for c in HORIZON_LABELS if c in _hz_tbl.columns]
        except Exception:
            _biases = []
        if not _biases:
            try:
                if eval_report is not None and model_name in eval_report["summary"].index:
                    _biases = [float(eval_report["summary"].loc[model_name, "Bias"])]
            except Exception:
                _biases = []
        _biases = [b for b in _biases if np.isfinite(b) and -0.9 < b < 0]
        _ov_count = f"{yoy_growth:+.1%}"
        _ov_title = f"FY{int(forecast_year)} units vs FY{int(latest_year)}"
        _ov_body = ("The model leans low on this data, so read the raw drop against its measured bias "
                    "before treating it as a market signal.")
        if _biases and not build_meta.get("bias_applied"):
            _corr = sorted(total_forecast_units / (1 + b) / last_year_units - 1 for b in _biases)
            _units_corr = sorted(total_forecast_units / (1 + b) for b in _biases)
            _pct = lambda x: "0%" if round(x, 2) == 0 else f"{x:+.0%}"  # never "-0%"
            _ov_count = (_pct(_corr[0]) if len(_corr) == 1
                         else f"{_pct(_corr[0])} to {_pct(_corr[-1])}")
            _ov_title = f"FY{int(forecast_year)} units vs FY{int(latest_year)}, bias-corrected"
            # Biases are negative, so max() is the smallest lean and min() the largest.
            _ran = (f"{abs(max(_biases)):.0%} to {abs(min(_biases)):.0%}" if len(_biases) > 1
                    else f"{abs(_biases[0]):.0%}")
            _near = (f"{_units_corr[0]:,.0f} to {_units_corr[-1]:,.0f}" if len(_units_corr) > 1
                     else f"{_units_corr[0]:,.0f}")
            _ov_body = (f"Raw forecast {total_forecast_units:,.0f} units ({yoy_growth:+.1%}). The model ran "
                        f"{_ran} low on the held-out year, putting FY{int(forecast_year)} nearer "
                        f"{_near} units.")
        _attn.append(("neutral", "Outlook", _ov_count.replace("-", "\u2212"), _ov_title, _ov_body,
                      "Open Model Accuracy", "Model Accuracy", "Model accuracy"))
    if _attn:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Needs attention", help=(
            "The findings to act on, each linked to the page that explains it. Counts follow the "
            "Scope control, like the rest of this page. The Outlook range corrects the raw forecast by "
            "the bias the model showed on the held-out year, at both horizons the forecast is used at "
            "(1 month and 12 months ahead)."))
        for _i, (_col, (_tone, _tag, _count, _title, _body, _btn, _pg, _tab)) in enumerate(
                zip(st.columns(len(_attn)), _attn)):
            with _col, st.container(border=True, height="stretch", key=f"attn_card_{_i}"):
                st.markdown(f'<span class="fc-pill fc-pill--{_tone}">{html.escape(_tag)}</span>'
                            f'<div class="attn-count">{html.escape(_count)}</div>'
                            f'<div class="attn-title">{html.escape(_title)}</div>'
                            f'<div class="attn-body">{html.escape(_body)}</div>', unsafe_allow_html=True)
                st.button(_btn, key=f"_attn_{_i}", on_click=go_to, args=(_pg, _tab),
                          icon=":material/arrow_forward:")

    # --- The decision the project is judged by, in the units the business uses ---
    # Model Accuracy owns this section (method, frontier chart, small print); here it
    # is three numbers and a way in. Both read frontier_points(), so they can't drift.
    _honest = build_meta.get("honest") or {}
    if res_df is not None and inv_plan is not None and _honest.get("res") is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        # Title and button on one row, the button flush right (a column slot left it short of the edge).
        _head = st.container(horizontal=True, vertical_alignment="center", key="plain_ov_biz_head")
        _head.markdown("#### Same money, fewer stockouts?", width="stretch", help=(
            f"Replays a quarterly-review ordering policy against FY{int(validation_year)}'s real demand using "
            f"only what was knowable at the start of that year, and compares the two policies at the SAME stock "
            f"value. Model Accuracy -> Business impact has the method and the full curve."))
        _head.button("Open Model Accuracy", key="_ov_biz", on_click=go_to,
                   args=("Model Accuracy", "Business impact"),
                   icon=":material/arrow_forward:")
        try:
            _ip = build_meta.get("inv_params", {})
            _lead = float(_ip.get("lead_time", 2.0))
            _review = float(_ip.get("review", 3.0))
            _legacy = float(_ip.get("legacy", 1.5))
            _svc = float(_ip.get("service", 0.95))
            _fprice = (history.drop_duplicates(col_product).set_index(col_product)["ListPrice"].to_dict()
                       if "ListPrice" in history.columns else {})
            _ovfr = run_policy_frontier(
                _honest["res"], col_product, model_name, _honest["sigma"], _honest["trailing"], _fprice,
                _lead, _review, tuple(sorted(set(FRONTIER_SERVICE_LEVELS) | {round(_svc, 2)})),
                tuple(sorted(set(FRONTIER_FLAT_MONTHS) | {_legacy})))
            fill_rate_kpis(frontier_points(_ovfr, round(_svc, 2), _legacy), _svc, _legacy, int(validation_year))
        except Exception as e:
            empty_state(f"The same-budget replay isn't available for this build: {e}")

# ============================= PAGE: CATEGORY PLAN ============================
# The grain planning actually happens at. Three of the four sections are pure
# aggregation of the build's own outputs, so they cost nothing; only the
# aggregation-level cross-check retrains anything, and that sits behind a
# button like the other expensive analyses on Model Accuracy.
elif page == "Category Plan":
    _levels = [c for c in ("Category1", "Category2", "Category") if c in history.columns]
    if not _levels:
        empty_state("**No product categories loaded,** so there's no category grain to plan at. "
                    "Add an item master with a category column on Build Forecast and rebuild.",
                    button="Go to Build Forecast")
        st.stop()

    # Both switches render beside the page title, above this point.
    level_col = st.session_state.get("cat_level", _levels[0])
    cat_metric = st.session_state.get("cat_metric", "Units")

    as_rev = cat_metric == "Revenue"
    _measure = "Revenue" if as_rev else "Units"
    _unit_lbl = "Revenue (฿)" if as_rev else "Units"
    _price = price_per_item(history, col_product)
    level_map = level_map_from(history, col_product, level_col)
    _lvl_name = _pretty_label(level_col)

    hist_l = history.copy()
    hist_l["_lvl"] = hist_l[col_product].map(level_map).fillna(UNKNOWN_LEVEL)
    fc_l = forecast.copy()
    fc_l["_lvl"] = fc_l[col_product].map(level_map).fillna(UNKNOWN_LEVEL)
    # Forecast revenue mirrors Overview: units x each item's average realised
    # price, so the two pages can't disagree on the same total.
    fc_l["_val"] = (fc_l["Final_Forecast"] * fc_l[col_product].map(_price).fillna(_price.mean())
                    if as_rev else fc_l["Final_Forecast"])

    _last_fy = int(hist_l["FiscalYear"].max())
    _plan_col = f"FY{int(forecast_year)} plan"
    _has_bt = res_df is not None and f"{model_name}_Pred" in res_df.columns

    # --- KPI row -----------------------------------------------------------
    _plan_total = fc_l["_val"].sum()
    _last_total = hist_l[hist_l["FiscalYear"] == _last_fy][_measure].sum()
    _cat_wape = None
    if _has_bt:
        _bu = rollup_backtest_to_level(res_df, col_product, level_col, level_map, f"{model_name}_Pred")
        _cat_wape = _wape_fn(_bu["Actual"], _bu["Pred"])
    _item_wape = None
    if metrics is not None and model_name in metrics.columns:
        _w = metrics[metrics["Metric"].str.contains("WAPE")]
        if len(_w):
            _item_wape = float(_w[model_name].values[0])

    k1, k2, k3 = st.columns(3)
    with k1:
        _fmt = short_baht(_plan_total) if as_rev else f"{_plan_total:,.0f}"
        _yoy = (_plan_total / _last_total - 1) if _last_total else None
        kpi_card(f"FY{int(forecast_year)} plan ({cat_metric.lower()})", _fmt, _yoy,
                 delta_text=f"{abs(_yoy):.1%} vs FY{_last_fy}" if _yoy is not None else None,
                 exact=f"฿{_plan_total:,.0f}" if as_rev else None,
                 note=f"FY{_last_fy} actual: " + (short_baht(_last_total) if as_rev else f"{_last_total:,.0f} units")
                      + ". Built bottom-up: the sum of the item forecasts.")
    with k2:
        # Deliberately no delta arrow here. The item-grain WAPE is not a worse
        # model to beat -- it is the SAME forecast scored finer -- so a green
        # "+38%" against it would read as an accuracy claim it doesn't support.
        # The note states the relationship instead.
        kpi_card(f"WAPE at {_lvl_name} × month",
                 f"{_cat_wape:.1%}" if _cat_wape is not None else "N/A",
                 note=(f"Item × month: {_item_wape:.1%}. Same forecast at a coarser grain, not a better model."
                       if _item_wape is not None else None))
    with k3:
        kpi_card(f"{_lvl_name} groups", f"{fc_l['_lvl'].nunique():,}", note="From the item master.")

    # --- history + plan chart ----------------------------------------------
    # The board's chart: the five biggest categories plus one "Other" stack, so
    # the legend stays short at any grain; the plan bars lighter with a dashed
    # outline; each annual bar's total on top.
    with st.container(border=True, key="cp_chart"):
        cc1, cc2 = st.columns([2.9, 1.6])
        with cc1:
            st.markdown(f"##### {_lvl_name} demand", help=f"History and the FY{int(forecast_year)} plan, by {_lvl_name}.")
        with cc2, st.container(horizontal=True, horizontal_alignment="right"):
            cat_gran = st.segmented_control("Granularity", ["Annual", "Quarterly", "Monthly"], default="Annual",
                                            required=True, key="cat_gran", label_visibility="collapsed")
        try:
            _hp = hist_l[["_lvl", "FiscalYear", "FiscalMonth", _measure]].rename(columns={_measure: "Value"})
            _fp = fc_l[["_lvl", "FiscalYear", "FiscalMonth", "_val"]].rename(columns={"_val": "Value"})
            both = pd.concat([_hp, _fp], ignore_index=True)
            both["Plan"] = both["FiscalYear"].astype(int) >= int(forecast_year)
            _top = both.groupby("_lvl")["Value"].sum().nlargest(5).index.tolist()
            _n_other = both["_lvl"].nunique() - len(_top)
            _other = f"Other ({_n_other} categor{'y' if _n_other == 1 else 'ies'})"
            both["_grp"] = both["_lvl"].where(both["_lvl"].isin(_top), _other)
            if cat_gran == "Annual":
                both["Period"] = "FY" + both["FiscalYear"].astype(int).astype(str) + np.where(both["Plan"], " plan", "")
                both["t"] = both["FiscalYear"].astype(int)
            else:
                both = bucket_periods(both, "FiscalYear", "FiscalMonth", cat_gran)
            ch_df = both.groupby(["t", "Period", "Plan", "_grp"], as_index=False)["Value"].sum().sort_values("t")
            _order = ch_df.drop_duplicates("Period").sort_values("t")["Period"].tolist()
            _cat_axis = (alt.Axis(labelAngle=0) if cat_gran == "Annual" else period_axis(_order, cat_gran))
            _grp_dom = _top + ([_other] if _n_other else [])
            ch_df["_grp_rank"] = ch_df["_grp"].map({g: i for i, g in enumerate(_grp_dom)})
            _totals = ch_df.groupby(["t", "Period"], as_index=False)["Value"].sum()
            _ymax = float(_totals["Value"].max()) if len(_totals) else 0.0
            _yscale = (alt.Scale(domain=[0, _ymax * 1.15], nice=False)
                       if cat_gran == "Annual" and _ymax > 0 else alt.Undefined)
            _is_plan = alt.datum.Plan
            bars = alt.Chart(ch_df).mark_bar().encode(
                x=alt.X("Period:N", sort=_order, title=None, axis=_cat_axis),
                y=alt.Y("Value:Q", title=_unit_lbl, stack="zero", scale=_yscale,
                        axis=revenue_axis() if as_rev else alt.Undefined),
                color=alt.Color("_grp:N", title=None, sort=_grp_dom,
                                scale=category_color_scale(level_col, _top, _other if _n_other else None),
                                legend=alt.Legend(orient="right", labelLimit=240)),
                order=alt.Order("_grp_rank:Q"),
                opacity=alt.condition(_is_plan, alt.value(0.55), alt.value(1.0)),
                stroke=alt.condition(_is_plan, alt.value(TEXT_MAIN), alt.value(None)),
                strokeDash=alt.condition(_is_plan, alt.value([4, 3]), alt.value([])),
                tooltip=[alt.Tooltip("_grp:N", title=_lvl_name), alt.Tooltip("Period:N", title="Period"),
                         alt.Tooltip("Value:Q", format=",.0f")],
            )
            _chart = bars
            if cat_gran == "Annual":
                _fmt_total = (lambda v: f"฿{v:,.0f}") if as_rev else (lambda v: f"{v:,.0f}")
                _totals["Label"] = _totals["Value"].map(_fmt_total)
                _chart = bars + alt.Chart(_totals).mark_text(
                    align="center", baseline="bottom", dy=-4, fontWeight=600, color=TEXT_MAIN,
                ).encode(x=alt.X("Period:N", sort=_order), y=alt.Y("Value:Q", scale=_yscale), text="Label:N")
            st.altair_chart(_chart.properties(height=300), width="stretch")
            st.caption(f"Solid bars are actuals through FY{_last_fy}. The lighter, dashed "
                       f"{'bar is' if cat_gran == 'Annual' else 'bars are'} the FY{int(forecast_year)} plan.")
        except Exception as e:
            st.error(f"Couldn't build the category chart: {e}")

    # --- the plan table ----------------------------------------------------
    brk = contrib = None
    if _has_bt:
        brk, contrib = level_accuracy_breakdown(res_df, history, col_product, level_col, f"{model_name}_Pred")
    _last_col = f"FY{_last_fy} actual"
    _by_item = fc_l.groupby(["_lvl", col_product])["_val"].sum()
    _g = _by_item.groupby(level=0)
    plan = pd.DataFrame({
        "Items": _g.size(),
        _last_col: hist_l[hist_l["FiscalYear"] == _last_fy].groupby("_lvl")[_measure].sum(),
        _plan_col: _g.sum(),
    }).fillna(0)
    # A display string: a category with no plan has no share, and a bare NaN
    # renders as "None" in the grid.
    _share = (_g.max() / _g.sum().where(_g.sum() > 0)).reindex(plan.index)
    plan["Top item share of plan"] = ["--" if pd.isna(v) else f"{v:.0%}" for v in _share]
    plan = plan.sort_values(_plan_col, ascending=False).rename_axis(level_col).reset_index()

    # The model's aggregate under-forecast makes nearly every category fall, so
    # only a fall bigger than that bias gets the warning pill. Bias read live off
    # the backtest, so the rule stays true if the model or the data changes.
    _bias_val = None
    try:
        if eval_report is not None and model_name in eval_report["summary"].index:
            _bias_val = float(eval_report["summary"].loc[model_name, "Bias"])
    except Exception:
        _bias_val = None
    _warn_pts = round(abs(_bias_val) * 100) if _bias_val is not None and _bias_val < 0 else 0
    _chg_lbl, _chg_warn = [], {}
    for n, p in zip(plan[_plan_col], plan[_last_col]):
        if p <= 0:
            _chg_lbl.append([])
            continue
        _pct = round((n / p - 1) * 100)
        _t = f"{'▼' if _pct < 0 else '▲'} {abs(_pct)}%"
        _chg_warn[_t] = _pct < -_warn_pts
        _chg_lbl.append([_t])
    plan["vs last FY"] = _chg_lbl
    _acc_col = f"Accuracy vs {_cat_wape:.0%}" if _cat_wape is not None else None
    if _acc_col is not None and brk is not None:
        _cw = brk[brk["Actual"] > 0].set_index(level_col)["WAPE"]
        plan[_acc_col] = [[] if pd.isna(_cw.get(c)) else
                          ["Meets target" if _cw.get(c) <= _cat_wape else "Above target"]
                          for c in plan[level_col]]
    plan = plan[[level_col, "Items", _last_col, _plan_col, "vs last FY", "Top item share of plan"]
                + ([_acc_col] if _acc_col in plan.columns else [])]

    with st.container(border=True, key="cp_plan"):
        _pl_hc, _pl_bc = st.columns([3, 1], vertical_alignment="center")
        # The CSV keeps plain text where the screen shows pills.
        _plan_csv = plan.assign(**{c: plan[c].str[0] for c in ("vs last FY", _acc_col) if c in plan.columns})
        _pl_bc.download_button("Download CSV", prettify(_plan_csv).to_csv(index=False).encode("utf-8"),
                               file_name=f"category_plan_{level_col}_{cat_metric.lower()}.csv",
                               mime="text/csv", icon=":material/download:", width="stretch",
                               key="cp_plan_csv")
        _pl_hc.markdown(f"##### FY{int(forecast_year)} plan by {_lvl_name}", help=(
            "The FY forecast rolled up to product category -- the number an S&OP review works with. "
            "It is the same forecast as every other page, summed; nothing is re-modelled here. "
            "'Top item share of plan' is the largest single item's share of the category's plan."))
        _cap = f"Sorted by planned {cat_metric.lower()}."
        if _acc_col in plan.columns:
            _cap += f" Accuracy is compared with the {_cat_wape:.0%} catalog-wide category WAPE."
        if as_rev:
            _cap += " Revenue = forecast units × each item's average realised price."
        st.caption(_cap)
        _money = "฿{:,.0f}" if as_rev else "{:,.0f}"
        _opts = sorted(_chg_warn)
        _cfg = {_lvl_name: st.column_config.TextColumn(width=260),
                "Top item share of plan": st.column_config.TextColumn(alignment="right"),
                "vs last FY": st.column_config.MultiselectColumn(
                    "vs last FY", options=_opts, color=[PILL["warn"] if _chg_warn[o] else PILL["neutral"] for o in _opts])}
        if _acc_col in plan.columns:
            _cfg[_acc_col] = st.column_config.MultiselectColumn(
                _acc_col, options=["Meets target", "Above target"], color=[PILL["good"], PILL["neutral"]])
        st.dataframe(
            prettify(plan).style.format({"Items": "{:,.0f}", _last_col: _money, _plan_col: _money}),
            hide_index=True, height=fit_height(len(plan)), column_config=_cfg,
        )
        with st.expander("How to read 'vs last FY'"):
            _read = [f"It compares the FY{int(forecast_year)} plan with FY{_last_fy} actuals."]
            if _warn_pts:
                _read.append(
                    f"The model under-forecasts total demand by about {_warn_pts}% on the held-out year, so a "
                    f"category has to fall by more than that before the drop is a real signal and not the bias "
                    f"showing through. Those falls are marked orange. Bias correction on Build Forecast removes "
                    f"most of it, at a cost in item-level accuracy -- Model Accuracy shows the exact trade.")
            else:
                _read.append("The model does not under-forecast total demand on the held-out year, so every "
                             "fall is marked orange.")
            st.markdown(" ".join(_read))

    # --- per-category accuracy --------------------------------------------
    # Not on the board, kept on purpose: it is the per-category accuracy record
    # behind the plan table's pills, with the bias and concentration they omit.
    _trust_card = st.container(border=True, key="cp_trust")
    _trust_card.markdown(f"##### Which categories can you trust?", help=(
        f"The same held-out FY{int(validation_year)} backtest, scored per category at monthly grain. "
        f"'Top item share' is the largest single item's share of that category's volume -- a category "
        f"dominated by one project SKU is not a demand stream, and its error says more about the "
        f"category's shape than about the model."))
    if brk is None:
        _trust_card.info("No backtest available for the current settings.")
    else:
        _brk_disp = brk.copy()
        _brk_disp["Top item share"] = ["--" if pd.isna(v) else f"{v:.0%}"
                                       for v in _brk_disp["Top item share"]]
        _good = _cat_wape if _cat_wape is not None else 0.35

        def _style_wape(v):
            if pd.isna(v):
                return ""
            return f"color: {GREEN if v <= _good else (RED if v > 2 * _good else TEXT_MUTED)}; font-weight: 600;"

        _trust_card.dataframe(
            prettify(_brk_disp).style
            .format({"Actual": "{:,.0f}", "Forecast": "{:,.0f}", "WAPE": "{:.1%}",
                     "Bias": "{:+.1%}", "Items sold": "{:,.0f}"})
            .map(_style_wape, subset=["WAPE"]),
            hide_index=True, height=fit_height(len(_brk_disp)),
            column_config={_pretty_label(level_col): st.column_config.TextColumn(width="medium")},
        )
        _trust = brk[(brk["WAPE"] <= _good) & (brk["Actual"] > 0)]
        _conc = brk[(brk["Top item share"] > 0.5) & (brk["Actual"] > 0)]
        _bits = [f"Green = at or better than the {_good:.0%} catalog-wide category WAPE."]
        if len(_trust):
            # Name every one when the list is short; the count above must match the names.
            _names = _trust[level_col].astype(str).tolist()
            _named = ", ".join(_names) if len(_names) <= 6 else ", ".join(_names[:5]) + f" and {len(_names) - 5} more"
            _bits.append(f"{len(_trust)} categor{'y' if len(_trust) == 1 else 'ies'} "
                         f"({_named}) meet it and carry "
                         f"{_trust['Actual'].sum() / max(brk['Actual'].sum(), 1):.0%} of the volume.")
        if len(_conc):
            _bits.append(f"{len(_conc)} are over 50% concentrated in a single item, where no model can "
                         f"forecast reorder timing -- plan those by customer contact, not by statistics.")
        _trust_card.caption("  ".join(_bits))

        # --- error attribution ---------------------------------------------
        # The board's one-line finding with a status pill; the full top-15
        # list stays one click away.
        with st.container(border=True, key="cp_errors"):
            st.markdown("##### Top error items", help=(
                "The biggest absolute contributors to error in the held-out year, with each one's share of "
                "the catalog's total error. Turns a bad category number into a short list of line items a "
                "planner can go and ask about."))
            _c_disp = contrib.sort_values("AbsError", ascending=False).head(15).copy()
            if len(_c_disp):
                _t = _c_disp.iloc[0]
                # Lead with the Item Code planners know; the internal ID only when there is none.
                _t_code = code_lookup.get(_t[col_product])
                _t_name = str(_t_code) if pd.notna(_t_code) and str(_t_code).strip() else str(_t[col_product])
                st.markdown(
                    f"Item <span style=\"font-family: 'IBM Plex Mono', monospace\">"
                    f"{html.escape(_t_name)}</span> "
                    f"({html.escape(str(_t[level_col]))}) alone is {_t['Share of total error']:.1%} of the "
                    f"catalog's forecast error: {_t['Actual']:,.0f} units sold against a "
                    f"{_t['Forecast']:,.0f}-unit forecast.",
                    unsafe_allow_html=True)
            if code_lookup:
                _c_disp.insert(2, "Item Code", _c_disp[col_product].map(code_lookup))
            with st.expander(f"Top {len(_c_disp)} items by error"):
                st.dataframe(
                    prettify(_c_disp).style.format(prettify_fmt({"Actual": "{:,.0f}", "Forecast": "{:,.0f}",
                                                                 "AbsError": "{:,.0f}",
                                                                 "Share of total error": "{:.1%}"})),
                    hide_index=True, height=fit_height(len(_c_disp)), column_config=item_pins(col_product),
                )

    # --- aggregation-level cross-check (the only part that retrains) -------
    # One card holds the title, the run button and -- once run -- the chart, like
    # Walk-forward on Model Accuracy. It is created here and filled further down
    # (elements are appended in call order); the two titled result cards stay
    # separate below it, so no card ends up nested inside another.
    _xc_card = st.container(border=True, key="cp_xcheck")
    # Title and caption on the left, the run button top right, as on the board.
    with _xc_card:
        _xc_hc, _xc_bc = st.columns([3, 1], vertical_alignment="center")
    _xc_hc.markdown(f"##### Direct {_pretty_label(level_col)} model", help=(
        "Two ways to get a category number: sum the item forecasts (bottom-up, what this app does), or "
        "aggregate the history first and forecast the category series directly. This scores both -- plus "
        "Croston, ETS and SARIMA fitted to each category's own series -- at two different origin years, "
        "because a single origin can make any one method look best by luck."))
    _xc_hc.caption("A direct category model is a challenger only. The plan of record stays bottom-up.")
    # Keyed on build_version too: without it a cross-check from a previous
    # build stayed on screen after a rebuild, describing a different model.
    _xk = f"_lvl_xcheck_{level_col}_v{st.session_state.get('build_version', 0)}"
    _tys = sorted(int(y) for y in train_years)
    _prior_ok = len(_tys) >= 3
    # Disappears once this level's result exists, like the Model Accuracy
    # experiment buttons. The result is stored per level, so switching to a
    # level that hasn't been run yet brings the button back for that level.
    _xc_slot = _xc_bc.empty()
    if _xk not in st.session_state and _xc_slot.button(
            "Run cross-check (about 20 s)", key=f"act_{_xk}", width="stretch"):
        with _xc_card, st.spinner(f"Forecasting at {_pretty_label(level_col)} level and re-scoring..."):
            try:
                st.session_state[_xk] = run_level_crosscheck(
                    history, res_df, forecast, col_product, level_col, tuple(_tys), int(validation_year),
                    int(fiscal_start_month), model_name,
                    tuple(build_meta.get("extra_features", [])),
                    tuple(build_meta.get("base_features") or []) or None,
                    tuple(build_meta.get("category_flag") or []) or None,
                    int(forecast_year),
                )
                remember_result(_xk)
                _xc_slot.empty()
            except Exception as e:
                st.error(f"Cross-check failed: {e}")
    xc = st.session_state.get(_xk)
    if xc is not None:
        comp = xc["comparison"]
        _recent = f"FY{int(validation_year)}"
        _chart_src = comp[comp["WAPE"].notna()].copy()
        # SARIMA can fail to estimate on 24 months and return a wildly
        # divergent forecast. That's a real finding, so the number stays in
        # the table -- but one 5,000% bar would flatten every other one.
        _dropped = _chart_src[_chart_src["WAPE"] > 1.5]
        _chart_src = _chart_src[_chart_src["WAPE"] <= 1.5].copy()
        # Every method here is fitted at the same level, and the x-axis title
        # already says so -- repeating "at Category1" in each y label just
        # pushed the distinguishing part of the name past the label width.
        _chart_src["Label"] = (_chart_src["Method"]
                               .str.replace(f" trained at {level_col}", "", regex=False)
                               .str.replace(f" at {level_col}", "", regex=False))
        # One chart per origin, NOT one row-faceted chart. Vega-Lite forces
        # autosize:"pad" on any facet spec (vega/vega-lite#5219), so a faceted
        # chart is laid out at its natural content width and overflowed a
        # narrow column -- measured 611px inside a 420px container -- whatever
        # `width` was set to. Two plain charts each honour width="stretch".
        # Pinning the y domain and the x domain is what keeps them comparable,
        # which the facet used to do for free: without it each chart would
        # scale and order itself independently and the two panels could not be
        # read against each other.
        _primary = f"Validated on {_recent}"
        _rank = _chart_src[_chart_src["Origin"] == _primary].set_index("Label")["WAPE"].to_dict()
        # Best-first at the most recent origin; methods absent there (dropped
        # as off-scale) fall to the bottom rather than disappearing.
        _yorder = sorted(_chart_src["Label"].unique(),
                         key=lambda lbl: (_rank.get(lbl, float("inf")), lbl))
        _xmax = float(_chart_src["WAPE"].max()) * 1.08
        _origins = sorted(_chart_src["Origin"].unique())
        with _xc_card:
            for _i, _org in enumerate(_origins):
                _last = _i == len(_origins) - 1
                _c = alt.Chart(_chart_src[_chart_src["Origin"] == _org]).mark_bar().encode(
                    # Full domain on every panel, so a method missing at one
                    # origin still occupies its row and the rows line up.
                    y=alt.Y("Label:N", sort=_yorder, title=None,
                            scale=alt.Scale(domain=_yorder),
                            axis=alt.Axis(labelLimit=240)),
                    x=alt.X("WAPE:Q", scale=alt.Scale(domain=[0, _xmax], nice=False),
                            title=(f"WAPE at {_pretty_label(level_col)} × month" if _last else None),
                            axis=alt.Axis(format=".0%")),
                    color=alt.Color("Approach:N",
                                    scale=alt.Scale(domain=["Bottom-up", "Direct"],
                                                    range=[ORANGE, TEXT_MUTED]),
                                    legend=alt.Legend(title=None, orient="top") if _i == 0 else None),
                    tooltip=["Method", "Approach", "Origin", alt.Tooltip("WAPE:Q", format=".1%"),
                             alt.Tooltip("Bias:Q", format="+.1%")],
                ).properties(height=alt.Step(24), title=_org)
                st.altair_chart(_c, width="stretch")
            if len(_dropped):
                st.caption(f"Off-scale and omitted from the chart: " + ", ".join(
                    f"{r.Method} ({r.Origin}) at {r.WAPE:.0%}" for r in _dropped.itertuples()) +
                    " -- these are fits that failed to estimate, not close contenders.")

        _piv = comp.pivot_table(index=["Method", "Approach"], columns="Origin", values="WAPE").reset_index()
        _bias = comp[comp["Origin"] == f"Validated on {_recent}"].set_index("Method")["Bias"]
        _piv["Bias"] = _piv["Method"].map(_bias)
        _ocols = [c for c in _piv.columns if c.startswith("Validated on")]
        _piv = _piv.sort_values(_ocols[-1] if _ocols else "Method")
        with st.container(border=True, key="cp_xc_years"):
            st.markdown("##### Accuracy at each validation year")
            st.dataframe(
                _piv.style.format({**{c: "{:.1%}" for c in _ocols}, "Bias": "{:+.1%}"}),
                hide_index=True,
            )
            st.caption(xc["verdict"])

        fwd = xc.get("forward")
        if fwd is not None and len(fwd):
            with st.container(border=True, key="cp_xc_range"):
                st.markdown(f"##### FY{int(forecast_year)} sanity range", help=(
                    "What the direct category model would forecast, next to the plan. This is a sanity range, "
                    "not a second plan: the table above is its accuracy record, and it is only shown because "
                    "a large disagreement is worth knowing about."))
                _fd = fwd.copy()
                _fd["Gap %"] = ["--" if b <= 0 else f"{(d / b - 1):+.0%}"
                                for d, b in zip(_fd["Direct"], _fd["Bottom-up"])]
                # No red/green here, unlike 'vs last FY' above: this is the spread
                # between two methods, not a result. A green "+161%" would imply
                # the higher number is the better one, which is exactly the
                # inference the accuracy table above rules out.
                st.dataframe(
                    _fd.style.format({"Bottom-up": "{:,.0f}", "Direct": "{:,.0f}"}),
                    hide_index=True,
                )
                _bt, _dt = _fd["Bottom-up"].sum(), _fd["Direct"].sum()
                st.caption(
                    f"Catalog total: **{_bt:,.0f}** units bottom-up vs **{_dt:,.0f}** direct "
                    f"({(_dt / _bt - 1):+.0%}). Bottom-up stays the plan of record -- only it also gives the "
                    f"per-item detail the inventory plan needs.")

# ============================= PAGE: MODEL ACCURACY ============================
elif page == "Model Accuracy":
    # Laid out to the Forecast Console board. Three sections, picked by the Section
    # switch beside the title (see the header above); only the open one is drawn.
    # "Business impact" -- the fill-rate replay the deck leads with -- comes first.
    acc_tab = st.session_state["acc_tab"]
    if acc_tab == "Business impact":
        # --- Same money, fewer stockouts? The stocking policy replayed honestly ---
        # The one section on this page in the units the business uses. WAPE measures
        # the forecast; this measures the DECISION it drives, by replaying the
        # stocking policy month by month against the held-out year's real demand.
        # It used to lead with an in-sample 95.9% vs 77.9%, which was optimistic
        # twice over: the policy held ~1.9x the stock, and each item's sigma was
        # measured on the very year being replayed. Both policies are now swept across
        # budgets and compared at the SAME stock value, using only what a planner
        # could have known at the start of the year (build_meta['honest']).
        st.markdown("#### Same money, fewer stockouts?", help=(
            "Replays a quarterly-review ordering policy against the held-out year's real demand, month by month, "
            "and measures how much of it would have shipped from stock. Everything is what a planner could have "
            "known at the start of that year: the 12-month-ahead forecast, each item's error measured on the "
            "year before, and sales up to that date. Each policy is tested across a range of stock budgets, so "
            "the two are compared at the same stock value rather than at whatever budget each happens to pick."))
        _honest = build_meta.get("honest") or {}
        if res_df is None or inv_plan is None or _honest.get("res") is None:
            empty_state("Needs a completed backtest, inventory plan and 12-month-ahead backtest."
                        + (f" ({_honest['error']})" if _honest.get("error") else ""))
        else:
            try:
                _ip = build_meta.get("inv_params", {})
                _lead = float(_ip.get("lead_time", 2.0))
                _review = float(_ip.get("review", 3.0))
                _legacy_m = float(_ip.get("legacy", 1.5))
                _svc = float(_ip.get("service", 0.95))
                _vy = int(validation_year)
                _price = (history.drop_duplicates(col_product).set_index(col_product)["ListPrice"].to_dict()
                          if "ListPrice" in history.columns else {})
                _fr = run_policy_frontier(
                    _honest["res"], col_product, model_name, _honest["sigma"], _honest["trailing"], _price,
                    _lead, _review, tuple(sorted(set(FRONTIER_SERVICE_LEVELS) | {round(_svc, 2)})),
                    tuple(sorted(set(FRONTIER_FLAT_MONTHS) | {_legacy_m})))
                _out = render_frontier(_fr, round(_svc, 2), _legacy_m, f"Fill rate vs stock held, FY{_vy} replay", _vy)
                _mine, _cur, _flat_same, _money = _out["mine"], _out["cur"], _out["flat_same"], _out["money"]
                _vc = _out["value_col"]
                _mult = float(_mine[_vc]) / float(_cur[_vc]) if float(_cur[_vc]) else np.nan
                # Every plan point against the flat rule at that same budget -- the
                # "ahead at every budget" claim is counted, never assumed.
                _pairs = [(float(r["Fill rate"]), fill_rate_at_budget(_fr, float(r[_vc]), value_col=_vc))
                          for _, r in _fr[_fr["Family"] == FAMILY_MODEL].iterrows()]
                _pairs = [(a, b) for a, b in _pairs if pd.notna(b)]
                _n_above = sum(a >= b for a, b in _pairs)
                # Beside the chart it explains, as on the board.
                with _out["text"]:
                    if pd.notna(_flat_same):
                        _g = (_mine["Fill rate"] - _flat_same) * 100
                        st.markdown(
                            f":small[**Same stock budget, {abs(_g):.1f} points {'more' if _g >= 0 else 'less'} demand "
                            f"shipped:** {_mine['Fill rate']:.1%} vs {_flat_same:.1%} at {_money(_mine[_vc])}. "
                            f"The gain is where the buffer sits, not more of it."
                            + (f" The plan is ahead at all {len(_pairs)} budgets tested." if _pairs and _n_above == len(_pairs)
                               else f" The plan is ahead at {_n_above} of {len(_pairs)} budgets tested." if _pairs else "") + "]")
                    _d_cur = (_mine["Fill rate"] - _cur["Fill rate"]) * 100
                    st.markdown(
                        f":small[Against today's {_legacy_m:g}-month rule it ships {abs(_d_cur):.1f} points "
                        f"{'more' if _d_cur >= 0 else 'less'}, on "
                        + (f"{_mult:.1f}x the stock" if pd.notna(_mult) else "more stock")
                        + f" ({_money(_mine[_vc])} vs {_money(_cur[_vc])}). That part is bought with working capital.]")
                    st.markdown(":small[Service level is the dial: Inventory Plan, Stocking rules.]")
                    st.caption(f"Lines: every budget tested. Replayed with only what was known at the start of "
                               f"FY{_vy}, scored on items with demand that year.")
                    # The method and the in-sample figure stay on the page, folded away.
                    with st.expander("How this was measured"):
                        st.caption(
                            f"A replay of FY{_vy} using only what was known at its start -- the 12-month-ahead "
                            f"forecast, each item's error (σ) measured on {_honest.get('sigma_basis')}, and sales up "
                            f"to the end of FY{_vy - 1}. Quarterly review, {_lead:g}-month lead time, unmet demand "
                            f"lost rather than backordered, scored on the {int(_mine['Items scored']):,} items that "
                            f"had demand in FY{_vy} (items that sold nothing cannot stock out, and counting them "
                            f"pushes every policy towards 100%). Stock starts at the order-up-to level with an empty "
                            f"pipeline. No stock-on-hand data yet, so this replays the policy, not MELFT's shelves.")
                        # The in-sample version, kept as a stated footnote rather than
                        # hidden: it is the number an earlier draft of the deck quoted.
                        try:
                            _ins = run_service_simulation(res_df, history, col_product, model_name, tuple(sorted(train_years)),
                                                          _vy, _lead, _review, _svc, _legacy_m)
                            _im = _ins[_ins["Policy"].str.startswith("Forecast-driven")].iloc[0]
                            _if = _ins[_ins["Policy"].str.startswith("Flat")].iloc[0]
                            st.caption(
                                f"Scored the easy way -- the 1-month-ahead backtest, with each item's error measured "
                                f"on FY{_vy} itself and no budget matching -- the same comparison reads "
                                f"{_im['Fill rate']:.1%} vs {_if['Fill rate']:.1%}. We don't quote that: its safety "
                                f"stock had already seen the year it was tested on"
                                + (f", and it held {_im['Avg inventory value'] / _if['Avg inventory value']:.1f}x the stock."
                                   if pd.notna(_im["Avg inventory value"]) and pd.notna(_if["Avg inventory value"])
                                   and _if["Avg inventory value"] > 0 else "."))
                        except Exception:
                            pass
            except Exception as e:
                empty_state(f"Service-level simulation unavailable: {e}")

    elif acc_tab == "Model accuracy":
        if metrics is None:
            empty_state("Not enough history to run a backtest with the current settings.")
        else:
            _vy = int(validation_year)
            _wrow = metrics[metrics["Metric"] == "WAPE (volume-weighted %)"].iloc[0]
            _grain = (eval_report["by_grain"][model_name]
                      if eval_report is not None and model_name in eval_report["by_grain"].columns else None)
            # The planning grain beside the hardest one: the same forecast, scored per
            # category per month and per item per month. Category 1 when there is one.
            _cat_rows = [] if _grain is None else [g for g in _grain.index
                                                   if g.endswith(" x month") and g != "SKU-month"]
            _cat_row = next((g for g in _cat_rows if g.startswith("Category1")), _cat_rows[0] if _cat_rows else None)
            _zero_share = float((res_df["Actual"] == 0).mean()) if res_df is not None and len(res_df) else None
            _k1, _k2 = st.columns(2)
            if _cat_row is not None:
                with _k1:
                    kpi_card(f"{pretty_grain(_cat_row)} WAPE · planning grain", f"{_grain[_cat_row]:.1%}",
                             note=f"how far the monthly forecast for each {pretty_grain(_cat_row.split(' x ')[0])} "
                                  f"is off. This is the level S&OP and purchasing plan at.")
            with (_k2 if _cat_row is not None else _k1):
                kpi_card("Item × month WAPE · hardest grain", f"{float(_wrow[model_name]):.1%}",
                         note=("the strictest test" + (f", with {_zero_share:.0%} of item-months at zero"
                                                       if _zero_share is not None else "")
                               + ". Used to rank models, not to judge the plan."))

            # --- Model comparison: the chart and the exact numbers side by side ------
            _models = ["Baseline"] + [c for c in metrics.columns if c not in ("Metric", "Baseline")]
            _mrows = metrics.set_index("Metric")
            _cd = pd.DataFrame({"Model": _models, "Value": [float(_wrow[c]) for c in _models]})
            _cd["Highlight"] = _cd["Model"].apply(
                lambda m: "Selected for forecast" if m == model_name else ("Baseline" if m == "Baseline" else "Other model"))
            _cd = _cd.sort_values("Value")
            _cmp_col, _num_col = st.columns(2)
            with _cmp_col, st.container(border=True, height="stretch"):
                st.markdown("##### WAPE by model", help="Item × month WAPE on the held-out year. Lower is better.")
                try:
                    _bars = alt.Chart(_cd).mark_bar().encode(
                        x=alt.X("Value:Q", title="WAPE", axis=alt.Axis(format=".0%")),
                        y=alt.Y("Model:N", sort=_cd["Model"].tolist(), title=None, axis=alt.Axis(labelLimit=220)),
                        color=alt.Color("Highlight:N", scale=alt.Scale(
                            domain=["Selected for forecast", "Other model", "Baseline"],
                            range=[ORANGE, CHART_ACTUAL, TEXT_MUTED]), legend=None),
                        tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("Value:Q", format=".1%", title="WAPE")],
                    )
                    _lbl = alt.Chart(_cd).mark_text(align="left", dx=6, fontWeight="bold", color=TEXT_MAIN).encode(
                        x="Value:Q", y=alt.Y("Model:N", sort=_cd["Model"].tolist()), text=alt.Text("Value:Q", format=".1%"))
                    st.altair_chart((_bars + _lbl).properties(height=alt.Step(34)), width="stretch")
                except Exception as e:
                    st.error(f"Couldn't build the comparison chart: {e}")
                # Which model and why, from the numbers: lowest WAPE, a near-tie, or a manual pick.
                _ml = _cd[_cd["Model"] != "Baseline"].set_index("Model")["Value"]
                _best, _mw = _ml.idxmin(), float(_ml.get(model_name, np.nan))
                _ties = [m for m in _ml.index if m != model_name and abs(float(_ml[m]) - _mw) < 0.0005]
                st.caption(
                    f"{model_name} drives the FY{int(forecast_year)} forecast: "
                    + ((f"lowest WAPE, tied with {', '.join(_ties)} at {_mw:.1%}." if _ties else "lowest WAPE.")
                       if _best == model_name or _ties else
                       f"picked manually; the lowest WAPE is {_best} at {float(_ml[_best]):.1%}.")
                    + " Baseline can never be the final model.")

            with _num_col, st.container(border=True, height="stretch"):
                st.markdown(f"##### FY{_vy} scores", help=(
                    f"The FY{_vy} held-out year. MAPE = average % error per row. MAE = average units off "
                    "per item-month. RMSE = like MAE, but a few big misses weigh more. Lower is better "
                    "for all four. WAPE leads because ~78% of item-months sell nothing: MAPE can't divide "
                    "by those zeros, and RMSE and MAE are in units, so big-volume items dominate them."))
                _tbl = pd.DataFrame({
                    "Model": _models,
                    "Status": [["In use"] if m == model_name else [] for m in _models],
                    "WAPE": [f"{float(_mrows.loc['WAPE (volume-weighted %)', m]):.1%}" for m in _models],
                    "MAPE": [f"{float(_mrows.loc['MAPE (row-average %)', m]):.1%}" for m in _models],
                    "MAE": [f"{float(_mrows.loc['MAE (avg units off by)', m]):.2f}" for m in _models],
                    # A build saved before RMSE existed has no RMSE row: leave the column out.
                    **({"RMSE": [f"{float(_mrows.loc['RMSE (big misses weigh more)', m]):.2f}" for m in _models]}
                       if "RMSE (big misses weigh more)" in _mrows.index else {}),
                }).set_index("Model").loc[_cd["Model"]].reset_index()
                st.dataframe(_tbl, hide_index=True, column_config={
                    "Status": st.column_config.MultiselectColumn("", options=["In use"], color=[PILL["good"]], width="small"),
                    **{c: st.column_config.TextColumn(alignment="right") for c in ("WAPE", "MAPE", "MAE", "RMSE")}})

            # --- How each model missed, month by month, in a year you pick ---------
            # Only the held-out year has every model's monthly predictions from the
            # build. An earlier year is backtested on demand (every model, trained on
            # the years before it) and kept like walk-forward's result.
            if res_df is not None:
                _all_yrs = sorted(set(int(y) for y in history["FiscalYear"].unique()))
                _miss_years = [y for y in _all_yrs if y < _vy and sum(t < y for t in _all_yrs) >= 2] + [_vy]
                if st.session_state.get("pv_year") not in _miss_years:
                    st.session_state.pop("pv_year", None)
                with st.container(border=True):
                    pv1, pv2 = st.columns([2, 1.4])
                    with pv2, st.container(horizontal=True, horizontal_alignment="right"):
                        if len(_miss_years) > 1:
                            pv_year = st.segmented_control(
                                "Year", _miss_years, default=_vy, required=True, key="pv_year",
                                format_func=lambda y: f"FY{y}", label_visibility="collapsed")
                        else:
                            pv_year = _vy
                        pv_metric = st.segmented_control("Show", ["Units", "Revenue"], default="Units", required=True, key="pv_metric", label_visibility="collapsed")
                    with pv1:
                        st.markdown(f"##### Monthly miss by model", help=(
                            "Trained on the earlier years only, then scored on a year it never saw. "
                            "Each line is one model's monthly gap between forecast and actual: above zero = "
                            "over-forecast (excess stock), below = under-forecast (stockout risk). The cards give "
                            "each model's average monthly miss against the catalog total, as a % of actual. "
                            f"FY{_vy} is the build's held-out year; an earlier year is backtested on request."))
                    _yr_key = f"_miss_fy{pv_year}_v{st.session_state.get('build_version', 0)}"
                    _yr_res = res_df if pv_year == _vy else st.session_state.get(_yr_key)
                    if _yr_res is None:
                        # Note and button in one slot, cleared together once the result is in.
                        _yr_slot = st.empty()
                        with _yr_slot.container(key="plain_miss_year"):
                            st.caption(f"FY{pv_year} isn't part of the build. Backtesting it retrains every model on "
                                       f"FY{_all_yrs[0]} to FY{pv_year - 1} and scores FY{pv_year}.")
                            _yr_go = st.button(f"Run FY{pv_year} backtest (about 2 min)", key="act_miss_year")
                        if _yr_go:
                            with st.spinner(f"Backtesting every model on FY{pv_year}..."):
                                try:
                                    _yr_res = run_year_backtest(
                                        history, col_product, tuple(build_meta.get("category_flag") or []) or None,
                                        int(fiscal_start_month),
                                        # Baseline and Croston are added by run_backtest itself.
                                        tuple(c for c in metrics.columns if c not in ("Metric", "Baseline", "Croston (SBA)")),
                                        tuple(build_meta.get("extra_features", [])),
                                        tuple(t for t in _all_yrs if t < pv_year), int(pv_year),
                                        tuple(build_meta.get("base_features") or []) or None)
                                    if _yr_res is None:
                                        raise ValueError("not enough history before that year")
                                    st.session_state[_yr_key] = _yr_res
                                    remember_result(_yr_key)
                                    _yr_slot.empty()
                                except Exception as e:
                                    _yr_res = None
                                    st.error(f"FY{pv_year} backtest failed: {e}")
                    if _yr_res is not None:
                        try:
                            pv_rev = pv_metric == "Revenue"
                            pv_price = price_per_item(history, col_product) if pv_rev else None
                            # Croston's flat rate says nothing about month-to-month shape.
                            pred_names = [c[:-5] for c in _yr_res.columns
                                          if c.endswith("_Pred") and c[:-5] != "Croston (SBA)"]

                            r = _yr_res.copy()
                            if pv_rev:
                                pmul = r[col_product].map(pv_price).fillna(pv_price.mean())
                                for nm in ["Actual"] + [f"{n}_Pred" for n in pred_names]:
                                    r[nm] = r[nm] * pmul
                            monthly = r.groupby("FiscalMonth").agg(
                                **{"Actual": ("Actual", "sum"), **{n: (f"{n}_Pred", "sum") for n in pred_names}}
                            ).reindex(range(1, 13)).fillna(0).reset_index()
                            monthly["Month"] = monthly["FiscalMonth"].apply(lambda m: fmonth_labels[int(m) - 1])

                            err = monthly[["Month"]].copy()
                            for n in pred_names:
                                err[n] = monthly[n] - monthly["Actual"]
                            elong = err.melt(id_vars=["Month"], value_vars=pred_names, var_name="Model", value_name="Gap")
                            elong = elong.merge(monthly[["Month", "Actual"]], on="Month")
                            elong["Gap %"] = elong["Gap"] / elong["Actual"].replace(0, np.nan)
                            # Average monthly miss against the catalog total, per model.
                            _a = monthly["Actual"].to_numpy(float)
                            with np.errstate(divide="ignore", invalid="ignore"):
                                _offs = {n: float(np.nanmean(np.where(
                                    _a > 0, np.abs(_a - monthly[n].to_numpy(float)) / _a, np.nan))) for n in pred_names}
                            zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#B0B4BB", strokeWidth=1.5).encode(y="y:Q")
                            band = alt.Chart(monthly).transform_calculate(lo="-0.1 * datum.Actual", hi="0.1 * datum.Actual").mark_area(
                                color="#EAF1F6", opacity=0.6).encode(
                                x=alt.X("Month:N", sort=fmonth_labels, title=None), y="lo:Q", y2="hi:Q")
                            # Same colour key as the bar charts above: orange = the model that
                            # drives the forecast, navy shades = the other models, grey =
                            # Baseline (dashed here, since lines need more than colour to tell
                            # apart). The domain is pinned -- an unpinned range is assigned in
                            # alphabetical order, which is how the selected model ended up pink.
                            _others = [n for n in pred_names if n not in (model_name, "Baseline")]
                            _navy_shades = [CHART_ACTUAL, "#5B6F99", "#9AAAC8", "#34466E", "#C3CCDD"]
                            _dom = ([model_name] if model_name in pred_names else []) + _others + (["Baseline"] if "Baseline" in pred_names else [])
                            _rng = ([ORANGE] if model_name in pred_names else []) + [_navy_shades[i % len(_navy_shades)] for i in range(len(_others))] + ([TEXT_MUTED] if "Baseline" in pred_names else [])
                            # One card per model, best first: its line colour and its miss.
                            # The cards are the chart's legend.
                            _tiles = []
                            for n in sorted(_offs, key=lambda k: (np.isnan(_offs[k]), _offs[k])):
                                _c = _rng[_dom.index(n)]
                                _line = (f"background:repeating-linear-gradient(90deg,{_c} 0 5px,transparent 5px 9px)"
                                         if n == "Baseline" else f"background:{_c}")
                                _tiles.append(
                                    f'<div class="miss-tile{" miss-tile--on" if n == model_name else ""}">'
                                    f'<div class="miss-name"><span class="miss-sw" style="{_line}"></span>'
                                    f'{html.escape(n)}</div>'
                                    f'<div class="miss-val">{"--" if np.isnan(_offs[n]) else f"{_offs[n]:.0%}"}</div></div>')
                            st.markdown(f'<div class="miss-tiles">{"".join(_tiles)}</div>', unsafe_allow_html=True)
                            _enc = dict(
                                x=alt.X("Month:N", sort=fmonth_labels, title=None),
                                y=alt.Y("Gap:Q", title=("Forecast − Actual (฿)" if pv_rev else "Forecast − Actual (units)"),
                                        axis=revenue_axis() if pv_rev else alt.Undefined),
                                color=alt.Color("Model:N", scale=alt.Scale(domain=_dom, range=_rng), legend=None),
                                strokeDash=alt.condition(alt.datum.Model == "Baseline",
                                                         alt.value([5, 4]), alt.value([1, 0])),
                                tooltip=[alt.Tooltip("Month:N"), alt.Tooltip("Model:N"),
                                         alt.Tooltip("Gap:Q", format="+,.0f", title="Gap"),
                                         alt.Tooltip("Gap %:Q", format="+.0%", title="Gap %")],
                            )
                            # Two layers so the selected model is drawn last, on top, and thicker.
                            _rest = alt.Chart(elong[elong["Model"] != model_name]).mark_line(
                                point={"size": 45}, strokeWidth=2).encode(**_enc)
                            _mine = alt.Chart(elong[elong["Model"] == model_name]).mark_line(
                                point={"size": 70}, strokeWidth=3.5).encode(**_enc)
                            st.altair_chart((band + zero + _rest + _mine).properties(height=360), width="stretch")
                            st.caption(f"Cards: average monthly miss against the FY{pv_year} catalog total, best first. "
                                       "Shaded band = within ±10% of actual.")
                        except Exception as e:
                            st.error(f"Couldn't build the model-error chart: {e}")

            # --- The same model scored 1 month ahead and 12 months ahead ----------
            _honest = build_meta.get("honest") or {}
            if _honest.get("table") is not None:
                _hz_card = st.container(border=True)
                _hz_card.markdown("##### 1 vs 12 months ahead", help=(
                    "Every other number on this page predicts each held-out month from the REAL months just before "
                    "it. The forecast you actually receive is made once, at the start of the year, and each later "
                    "month is built on its own earlier predictions. This scores the same model both ways, on the "
                    "same items and months."))
                _one, _twelve = HORIZON_LABELS
                _ht = _honest["table"]

                def _hfmt(metric, v):
                    if pd.isna(v):
                        return "--"
                    return f"{v:+.1%}" if metric == "Bias" else (f"{v:.2f}" if metric == "MASE" else f"{v:.1%}")

                _hz_card.dataframe(pd.DataFrame({
                    "Measure": [m.replace("WAPE, ", "WAPE: ").replace(" x ", " × ")
                                .replace("Category1", "Category 1").replace("Category2", "Category 2") for m in _ht.index],
                    _one: [_hfmt(m, v) for m, v in _ht[_one].items()],
                    _twelve: [_hfmt(m, v) for m, v in _ht[_twelve].items()],
                }), hide_index=True, column_config={c: st.column_config.TextColumn(alignment="right") for c in HORIZON_LABELS})
                _w1, _w12 = _ht.loc["WAPE, item x month", _one], _ht.loc["WAPE, item x month", _twelve]
                _b1, _b12 = _ht.loc["Bias", _one], _ht.loc["Bias", _twelve]
                _ly = int(history["FiscalYear"].max())
                _ly_units = float(history.loc[history["FiscalYear"] == _ly, "Units"].sum())
                _fc_gap = (float(forecast["Final_Forecast"].sum()) / _ly_units - 1) if _ly_units else np.nan
                _small = abs(_w12 - _w1) < 0.02
                _bias_small = abs(_b12 - _b1) < 0.03
                _cap = [
                    (f"**Item × month WAPE barely moves** ({_w1:.1%} → {_w12:.1%}), so the headline holds at the "
                     f"horizon the plan is actually made at." if _small else
                     f"**Item × month WAPE moves from {_w1:.1%} to {_w12:.1%}** at the horizon the plan is actually made at."),
                    (f"**Bias holds too** ({_b1:+.1%} → {_b12:+.1%})" if _bias_small else
                     f"**Bias moves: {_b1:+.1%} → {_b12:+.1%}.** Each month's forecast becomes the next month's "
                     f"input, so the lean builds through the year")
                    # Tie it to the forward forecast only when both point the same way.
                    + (f" -- consistent with the FY{int(forecast_year)} forecast sitting {abs(_fc_gap):.0%} "
                       f"{'below' if _fc_gap < 0 else 'above'} FY{_ly} actuals."
                       if pd.notna(_fc_gap) and pd.notna(_b12) and (_b12 < 0) == (_fc_gap < 0) else "."),
                ]
                _cat = next((m for m in _ht.index if m.startswith("WAPE, Category")
                             and pd.notna(_ht.loc[m, _one]) and pd.notna(_ht.loc[m, _twelve])), None)
                if _cat:
                    _cap.append(f"Category-level WAPE: {_ht.loc[_cat, _one]:.1%} → {_ht.loc[_cat, _twelve]:.1%}.")
                _cap.append("Plain WAPE also rewards under-forecasting on data that is mostly zeros -- read it next to "
                            "the bias, never alone.")
                _hz_card.caption("  ".join(_cap))

            # --- Accuracy in context: the same forecast summed to coarser grains ----
            if _grain is not None:
                with st.container(border=True):
                    st.markdown("##### WAPE by grain", help=(
                        "The headline WAPE scores every item in every month -- the hardest possible test. "
                        "These rows sum the same forecast to the grain planning actually happens at before "
                        "scoring it. The catalog total is also where classical methods (ETS / SARIMA) can run, "
                        "so that row compares against the best of them."))
                    # "Total (volume)" collapses WAPE to |aggregate bias| -- an identity, not
                    # a measurement -- so the total is shown against the classical methods,
                    # which is the one comparison that row can honestly make.
                    _gr = pd.DataFrame({"Grain": [pretty_grain(g) for g in _grain.index if g != "Total (volume)"],
                                        "WAPE": [f"{_grain[g]:.1%}" for g in _grain.index if g != "Total (volume)"]})
                    _cv = None
                    try:
                        _mlm = (res_df.groupby("FiscalMonth")[f"{model_name}_Pred"].sum()
                                .reindex(range(1, 13)).fillna(0).to_numpy()) if res_df is not None else None
                        _bt_u, _ = run_classical_crosscheck(history, forecast, int(fiscal_start_month), "Units",
                                                            tuple(_mlm) if _mlm is not None else None)
                        _cv = classical_verdict(_bt_u)
                    except Exception:
                        pass
                    if _cv:
                        _gr.loc[len(_gr)] = [f"Catalog total, {_cv[0]} vs ours", f"{_cv[1]:.1%} vs {_cv[2]:.1%}"]
                    st.dataframe(_gr, hide_index=True, column_config={"WAPE": st.column_config.TextColumn(alignment="right")})
                    st.caption((f"On the catalog total, {_cv[3]}. " if _cv else "")
                               + "The model's real edge is the per-item detail classical methods cannot produce at all.")
                    with st.expander("Every measure, every model"):
                        fmt = {"WAPE": "{:.1%}", "MAE": "{:.2f}", "Bias": "{:+.1%}", "MASE": "{:.2f}", "MAPE": "{:.1%}", "n_rows": "{:,.0f}"}
                        for _ttl, _key in (("All items", "summary"), ("Active items only", "active_only")):
                            st.markdown(f"**{_ttl}**")
                            st.dataframe(prettify(eval_report[_key]).style.format(prettify_fmt(fmt)),
                                         height=fit_height(len(eval_report[_key])))
                        st.markdown("**WAPE by grain**")
                        st.dataframe(prettify(eval_report["by_grain"]).rename(index=pretty_grain).style.format("{:.1%}"))
                        try:
                            _b = float(eval_report["summary"].loc[model_name, "Bias"])
                            _b_txt = f"the model's {abs(_b):.0%} {'under' if _b < 0 else 'over'}-forecast"
                        except Exception:
                            _b_txt = "the model's aggregate bias"
                        st.caption(f"**Total (volume)** collapses WAPE to one number, the aggregate bias, so that "
                                   f"row measures {_b_txt}, not precision.")

            # --- Walk-forward: the button beside the title, as on the board ---------
            with st.container(border=True):
                _wt, _wb = st.columns([2, 1], vertical_alignment="center")
                with _wt:
                    st.markdown("##### Walk-forward validation", help=(
                        "Retrains the model at successive points in history and scores the next untouched year each time "
                        "(train->FY22 predict FY23; train->FY23 predict FY24). Two hold-out years instead of one, and it "
                        "shows the model improving as data accumulates."))
                    st.caption(f"Re-scores the model on several past years, not just FY{_vy}.")
                _all_years = sorted(set(int(y) for y in history["FiscalYear"].unique()))
                _wf_key = f"_wf_result_v{st.session_state.get('build_version', 0)}"
                # A slot cleared once a result exists, so the button disappears in the
                # same run the result appears (it returns after a rebuild or a failure).
                _wf_slot = _wb.empty()
                if _wf_key not in st.session_state and _wf_slot.button("Run walk-forward (about 1 min)", width="stretch",
                                                                                    key="act_walk_forward"):
                    with st.spinner("Running rolling-origin backtest..."):
                        try:
                            wf = run_walk_forward(
                                history, col_product, tuple(build_meta.get("category_flag") or []) or None,
                                int(fiscal_start_month), model_name,
                                tuple(build_meta.get("extra_features", [])), tuple(_all_years),
                                tuple(build_meta.get("base_features") or []) or None,
                            )
                            st.session_state[_wf_key] = wf
                            remember_result(_wf_key)
                            _wf_slot.empty()
                        except Exception as e:
                            st.error(f"Walk-forward failed: {e}")
                wf = st.session_state.get(_wf_key)
                if wf is not None and len(wf):
                    line = alt.Chart(wf[wf["Model"].isin([model_name, "Baseline"])]).mark_line(point=True).encode(
                        x=alt.X("ValidationYear:O", title="Validation fiscal year", axis=alt.Axis(labelAngle=0)),
                        y=alt.Y("WAPE:Q", title="WAPE", axis=alt.Axis(format=".0%")),
                        # Pin the domain: without it Altair orders the scale
                        # alphabetically, which handed the orange accent to "Baseline".
                        # Orange means "the model we chose" everywhere else on this page.
                        color=alt.Color("Model:N",
                                        scale=alt.Scale(domain=[model_name, "Baseline"], range=[ORANGE, TEXT_MUTED]),
                                        legend=alt.Legend(title=None, orient="top")),
                        strokeDash=alt.condition(alt.datum.Model == "Baseline",
                                                 alt.value([5, 4]), alt.value([1, 0])),
                        tooltip=[alt.Tooltip("ValidationYear:O", title="Validation year"), "Model",
                                 alt.Tooltip("WAPE:Q", format=".1%")],
                    ).properties(height=300)
                    st.altair_chart(line, width="stretch")
                    st.dataframe(
                        prettify(wf.pivot_table(index="ValidationYear", columns="Model", values="WAPE")).style.format("{:.1%}"),
                        height="content",
                    )

    elif acc_tab == "Diagnostics":
        # Every section is one card, as on the Diagnostics board.
        if metrics is not None:
            # --- Residual bias correction: before / after on the held-out year -----
            _bc = build_meta.get("bias_correction")
            if _bc and res_df is not None and f"{model_name}_Pred" in res_df.columns:
                with st.container(border=True):
                    _applied = bool(build_meta.get("bias_applied"))
                    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center",
                                      key="plain_bc_head"):
                        st.markdown("##### Residual bias correction", help=(
                            "A model trained to minimise mean error on mostly-zero data leans low on the occasional "
                            "spike. We measure that lean on the held-out year and rescale the forecast by a factor per "
                            "sparsity tier. Applied to the live forecast only when the 'Correct residual forecast bias' "
                            "box is ticked on Build Forecast."))
                        st.markdown(f'<span class="fc-pill fc-pill--{"good" if _applied else "neutral"}">'
                                    f'{"Applied" if _applied else "Not applied"}</span>',
                                    unsafe_allow_html=True, width="content")
                    try:
                        _pcol = f"{model_name}_Pred"
                        _r = res_df[[col_product, "Actual", _pcol]].copy()
                        _fac = _bias_factor_series(_r, _bc, history, col_product)
                        _corr = _r[_pcol].to_numpy(dtype=float) * _fac
                        _raww, _corw = _wape_fn(_r["Actual"], _r[_pcol]), _wape_fn(_r["Actual"], _corr)
                        _rawb, _corb = _bias_fn(_r["Actual"], _r[_pcol]), _bias_fn(_r["Actual"], _corr)
                        bc1, bc2 = st.columns(2)
                        with bc1:
                            st.markdown("**Aggregate bias & WAPE**")
                            st.dataframe(pd.DataFrame({
                                "Model": ["Raw model", "Bias-corrected"],
                                "Aggregate bias": [f"{_rawb:+.1%}", f"{_corb:+.1%}"],
                                "WAPE": [f"{_raww:.1%}", f"{_corw:.1%}"],
                            }), hide_index=True)
                        with bc2:
                            _tiers = _bc.get("by_tier") or {}
                            if _tiers:
                                st.markdown("**Sparsity tier multiplier**")
                                st.dataframe(pd.DataFrame(
                                    {"Sparsity tier": list(_tiers), "Multiplier": [f"×{v:.2f}" for v in _tiers.values()]}),
                                    hide_index=True)
                        _state = "Applied to the FY forecast" if _applied else "Currently measured but not applied"
                        _tradeoff = ("cuts aggregate bias but costs item-level WAPE" if _corw > _raww + 0.002
                                     else "improves aggregate bias at little WAPE cost")
                        _dir = "under-forecasts" if _rawb < 0 else "over-forecasts"
                        st.caption(f"The raw model {_dir} total demand by {abs(_rawb):.1%}. Correcting it {_tradeoff} "
                                   f"— it also rescales items that were already fine. {_state}. "
                                   f"Tier factors are capped to ×[0.7, 1.4] (a one-year estimate is noisy).")
                    except Exception as e:
                        st.info(f"Bias-correction panel unavailable: {e}")

            # --- Accuracy by item sparsity (moved here from Model accuracy, per the board)
            if eval_report is not None and eval_report.get("by_sparsity") is not None:
                with st.container(border=True):
                    st.markdown("##### Accuracy by item sparsity", help=(
                        "Sparse items (sell in few months) are ~75% of the catalog and much harder to forecast -- "
                        "which is what pulls the blended number down."))
                    st.caption("WAPE by model, split by how often each item sells. Sparse items are most of the "
                               "catalog and hardest to forecast.")
                    by_sp = eval_report["by_sparsity"]
                    model_wape_cols = [c for c in by_sp.columns if c not in ("Sparsity Tier", "N items")]
                    _sp_disp = prettify(by_sp)
                    st.dataframe(
                        _sp_disp.style.format({**{c: "{:.1%}" for c in model_wape_cols}, "Items": "{:,}"})
                        .set_properties(subset=[c for c in _sp_disp.columns if c == model_name],
                                        **{"font-weight": "600"}),
                        hide_index=True,
                    )

            # --- Prediction-interval calibration ---------------------------------
            _cov = build_meta.get("coverage_report")
            if _cov:
                with st.container(border=True):
                    st.markdown("##### Interval calibration", help=(
                        "The p10-p90 band is only useful if it's honest. On the held-out year we check how often the "
                        "real value actually landed inside the band (should be ~80%) and report the pinball loss -- the "
                        "proper score for a quantile forecast, lower is better."))
                    _tgt, _obs = _cov["target"], _cov["coverage"]
                    _uhit, _utgt = _cov.get("upper_hit"), _cov.get("upper_target", 0.9)
                    # The p90 hit rate leads: it is the side that sizes safety stock. The
                    # two-sided number is shown without a verdict arrow (see the caveat).
                    ci1, ci2 = st.columns(2, gap="medium")
                    with ci2:
                        kpi_card("p10-p90 coverage", f"{_obs:.1%}",
                                 note=f"Target {_tgt:.0%} over {_cov['n']:,} item-months.")
                    with ci1:
                        if _uhit is not None:
                            _hu = (build_meta.get("honest") or {}).get("upper_hit")
                            kpi_card(f"Demand at or below p{int(_utgt*100)}", f"{_uhit:.1%}",
                                     (_uhit - _utgt), delta_positive_is_good=True,
                                     delta_text=f"{abs(_uhit - _utgt) * 100:.1f} pts "
                                                f"{'above' if _uhit >= _utgt else 'below'} target",
                                     note=f"Target {_utgt:.0%}. This is the side that sizes safety stock."
                                          + (f" Forecast 12 months ahead, as it is actually used: {_hu:.1%}."
                                             if _hu is not None else ""))
                    st.markdown("**Pinball loss by quantile**", help=(
                        "The proper score for a quantile forecast: it penalises a p90 more for "
                        "falling short than for overshooting, and the reverse for a p10. Lower is better; "
                        "compare a quantile against itself across builds, not across quantiles."))
                    _pb = _cov.get("pinball", {})
                    st.dataframe(pd.DataFrame({
                        "Quantile": [f"p{int(q*100)}" for q in _pb],
                        "Pinball loss": [f"{v:.3f}" for v in _pb.values()],
                    }), hide_index=True)
                    _verdict = ("conservative -- the band is wider than it needs to be, which is the safe direction "
                                "for stocking" if _obs > _tgt + 0.03
                                else ("narrow -- real demand escaped the band more often than 1 month in 5"
                                      if _obs < _tgt - 0.03 else "well calibrated"))
                    _zero = _cov.get("lower_at_zero")
                    _caveat = ""
                    if _zero is not None and _zero > 0.5:
                        _caveat = (f" Read the two-sided number with care: demand is so intermittent that the p10 model "
                                   f"correctly predicts zero for {_zero:.0%} of item-months, so the lower edge of the band "
                                   f"sits on the floor and swallows every zero actual for free. The p"
                                   f"{int(_utgt*100)} hit-rate card on the left is the honest, decision-relevant one.")
                    st.caption(f"Two-sided coverage is {_verdict}.{_caveat}")

            with st.container(border=True):
                _at, _ab = st.columns([2, 1], vertical_alignment="center")
                _at.markdown("##### Feature ablation", help=(
                    "Adds one feature group at a time and re-scores. Shows which features actually move accuracy "
                    "and which just add complexity -- the answer to 'why so many features?'"))
                _at.caption(f"Adds one feature group at a time and re-scores FY{int(validation_year)}.")
                # Keyed on build_version, like PRODUCT_KEY: a stale ablation from the
                # previous build must not survive a rebuild -- its rows now vary with
                # the feature selection, so a leftover table would be actively wrong.
                _abl_key = f"_abl_result_v{st.session_state.get('build_version', 0)}"
                # The button lives in a slot that is cleared once a result exists, so it
                # disappears in the same run the table appears. It returns after a
                # rebuild (new build_version, no result yet) or if the run failed.
                _abl_slot = _ab.empty()
                if _abl_key not in st.session_state and _abl_slot.button("Run feature ablation (about 1 min)",
                                                                         width="stretch", key="act_ablation"):
                    with st.spinner("Ablating feature groups..."):
                        try:
                            st.session_state[_abl_key] = run_feature_ablation(
                                history, col_product, tuple(sorted(train_years)), int(validation_year), int(fiscal_start_month), model_name)
                            remember_result(_abl_key)
                            _abl_slot.empty()
                        except Exception as e:
                            st.error(f"Ablation failed: {e}")
                abl = st.session_state.get(_abl_key)
                if abl is not None and len(abl):
                    abl_disp = abl.copy()
                    _cm = "WAPE (Category-month)"
                    if _cm in abl_disp.columns:
                        # The base row has no category feature yet, so there is no
                        # category grain to score and the value is NaN. Streamlit's grid
                        # prints a bare NaN as "None", so build the display string here
                        # rather than leaving it to a number formatter.
                        abl_disp[_cm] = ["--" if pd.isna(v) else f"{v:.1%}" for v in abl_disp[_cm]]
                    _abl_fmt = {"WAPE (SKU-month)": "{:.1%}", "MASE": "{:.2f}",
                                "Bias": "{:+.1%}", "n_features": "{:,.0f}"}
                    st.dataframe(
                        prettify(abl_disp).style.format(prettify_fmt(_abl_fmt)),
                        hide_index=True,
                    )
                    base_w = abl["WAPE (SKU-month)"].iloc[0]
                    full_w = abl["WAPE (SKU-month)"].iloc[-1]
                    # Name the group from the table: this used to be fixed text naming
                    # distinct customers whatever the numbers said.
                    _steps = abl["WAPE (SKU-month)"].diff().iloc[1:]
                    _driver = ""
                    if len(_steps) and _steps.min() < 0:
                        _bi = _steps.idxmin()
                        _bname = str(abl.loc[_bi, "Step"]).lstrip("+ ").lower()
                        _bpts = abs(_steps.min()) * 100
                        _driver = (f", most of it from one ({_bname}, {_bpts:.1f} points)"
                                   if base_w > full_w and _steps.min() <= (full_w - base_w) / 2
                                   else f"; the biggest single step is {_bname} ({_bpts:.1f} points)")
                    st.caption(
                        f"Base features {base_w:.1%} WAPE, full set {full_w:.1%}: every engineered feature together "
                        f"moves the headline ~{abs(base_w - full_w) * 100:.0f} points{_driver}. The data is the "
                        f"ceiling, not the feature count. Rows are the feature groups this build included."
                    )

        _imp_help = "Which features the selected model relies on most, measured on held-out data."
        if importance_df is not None:
            try:
                importance_category_cols = [c for c in ["Category", "Category1", "Category2"] if c in history.columns]
                display_importance = importance_df
                # The grouping checkbox sits on the title row of the card whose chart it controls.
                with st.container(border=True):
                    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center",
                                      key="plain_imp_head"):
                        st.markdown("##### What drives the forecast", help=_imp_help)
                        if importance_category_cols and st.checkbox("Group category into one bar", value=True,
                                                                    width="content"):
                            display_importance = group_importance_by_category(importance_df, importance_category_cols)

                    # Built-in importances (XGBoost gain shares, LightGBM split counts) are shown
                    # as shares of their total; permutation scores are not shares.
                    _shares = str(importance_method).startswith("Built-in")
                    if _shares and display_importance["Importance"].sum() > 0:
                        display_importance = display_importance.assign(
                            Importance=display_importance["Importance"] / display_importance["Importance"].sum())
                    _ifmt = ".1%" if _shares else ".3f"
                    top_n = display_importance.head(12).copy()
                    top_n["Feature"] = top_n["Feature"].map(_pretty_label)
                    top_n = top_n.sort_values("Importance")
                    _y = alt.Y("Feature:N", sort=top_n["Feature"].tolist(), title=None, axis=alt.Axis(labelLimit=260))
                    _bars = alt.Chart(top_n).mark_bar(color=ORANGE).encode(
                        x=alt.X("Importance:Q", title=None,
                                axis=alt.Axis(format=".0%") if _shares else alt.Undefined),
                        y=_y,
                        tooltip=["Feature:N", alt.Tooltip("Importance:Q", format=_ifmt)],
                    )
                    _vals = alt.Chart(top_n).mark_text(align="left", dx=6, fontWeight="bold", color=TEXT_MAIN).encode(
                        x="Importance:Q", y=_y, text=alt.Text("Importance:Q", format=_ifmt))
                    st.altair_chart((_bars + _vals).properties(height=alt.Step(30)), width="stretch")
                    if _shares and len(display_importance) > 3:
                        _ranked = display_importance.sort_values("Importance", ascending=False)
                        _names = [_pretty_label(f) for f in _ranked["Feature"].head(3)]
                        _v = _ranked["Importance"].to_numpy()
                        st.caption(f"{_names[0]} leads at {_v[0]:.1%}, and with {_names[1]} and {_names[2]} the top "
                                   f"three carry {_v[:3].sum():.0%} of the model's weight. No other feature "
                                   f"contributes more than {_v[3]:.1%}.")
            except Exception as e:
                st.error(f"Couldn't build the feature importance chart: {e}")
        else:
            empty_state("Feature importance isn't available for this run.")

        # --- Classical time-series cross-check (aggregate level) ---
        # Moved here from Overview in the Forecast Console redesign: it answers
        # "why not SARIMA?", which is a diagnostic question, not a planning one.
        with st.container(border=True):
            st.markdown("##### ETS / SARIMA cross-check", help=(
                "SARIMA and ETS can't handle a per-item series that's 78% zeros, but they work on the total. "
                "So we fit them to the monthly total and compare their prediction of the held-out year to ours. "
                "This is the 'why not SARIMA?' answer: it's valid at the aggregate, where it lands in the same "
                "range as our model -- but only our model also gives the per-item detail SARIMA can't."))
            try:
                ml_monthly_val = (res_df.groupby("FiscalMonth")[f"{model_name}_Pred"].sum()
                                  .reindex(range(1, 13)).fillna(0).to_numpy()) if res_df is not None else None
                bt_u, fw_u = run_classical_crosscheck(history, forecast, int(fiscal_start_month), "Units",
                                                      tuple(ml_monthly_val) if ml_monthly_val is not None else None)
                c1, c2 = st.columns(2)
                with c1:
                    # The actual total is the same on every row, so it goes in the
                    # heading rather than in a column that the half-width card clipped.
                    _act_total = float(bt_u["TotalActual"].iloc[0]) if len(bt_u) else float("nan")
                    st.markdown(f"**FY{int(validation_year)} back-test** — held out, actual {_act_total:,.0f} units")
                    st.dataframe(bt_u.drop(columns=["TotalActual"]).rename(columns={"TotalForecast": "Forecast total"})
                                 .style.format({"WAPE": "{:.1%}", "MAE": "{:,.0f}", "Forecast total": "{:,.0f}"}),
                                 hide_index=True)
                with c2:
                    st.markdown(f"**FY{int(forecast_year)} forecast** — total units")
                    st.dataframe(fw_u.rename(columns={"ForecastTotal": "Forecast total units"})
                                 .style.format({"Forecast total units": "{:,.0f}"}),
                                 hide_index=True)
                _cv = classical_verdict(bt_u)
                if _cv:
                    st.caption(f"On the one series classical methods can model -- the catalog total -- {_cv[3]}. "
                               f"Our model's real edge is the per-item detail they can't produce at all.")
                else:
                    st.caption("Classical methods can only model the catalog total. Our model's real edge is the "
                               "per-item detail they can't produce at all.")
            except Exception as e:
                st.caption(f":material/info: Classical cross-check unavailable: {e}")

elif page == "Product Detail":
    # A fragment: the filters, search and focus pickers rerun only this page, not the whole app.
    @st.fragment
    def _product_detail_page():
        with st.container(key="plain_product_detail"):
            _product_detail_body()

    def _product_detail_body():
        # Per-product forecast revenue, using each product's average historical
        # price (Revenue / Units from training data) applied to its FY forecast
        # units. This is revenue, not true profit -- we don't have a cost/margin
        # column, only sale price -- but it's the closest proxy available and
        # the practical equivalent for prioritizing which items matter most.
        hist_totals = hist_f.groupby(col_product).agg(Hist_Units=("Units", "sum"), Hist_Revenue=("Revenue", "sum"))
        avg_price = (hist_totals["Hist_Revenue"] / hist_totals["Hist_Units"].replace(0, pd.NA)).fillna(0)
        fc_units_by_product = fc_f.groupby(col_product)["Final_Forecast"].sum()
        fc_revenue_by_product = (fc_units_by_product * avg_price.reindex(fc_units_by_product.index).fillna(0)).fillna(0)

        last_year_units_by_product = hist_f[hist_f["FiscalYear"] == latest_year].groupby(col_product)["Units"].sum()
        variance_pct_by_product = ((fc_units_by_product - last_year_units_by_product) / last_year_units_by_product.replace(0, pd.NA)).fillna(0)

        available_category_cols = [c for c in ["Category", "Category1", "Category2"] if c in hist_f.columns]

        # Filter narrows WHICH products are eligible below (unlike sorting, which
        # just reorders the same full list) -- much faster than sorting +
        # scrolling to find one category's block in a catalog this size.
        # Every control sits in one card: the category filters on the first row,
        # the sort, the search and the focus picker on the second.
        filtered_products = selected_products
        _ctl_card = st.container(border=True)
        if available_category_cols:
            filter_cols = _ctl_card.columns(len(available_category_cols))
            prior_choice_for_key = ""
            for i, cat_col in enumerate(available_category_cols):
                cat_lookup_for_filter = hist_f.drop_duplicates(col_product).set_index(col_product)[cat_col]
                # cat_values (and this widget's key) are built from the ALREADY
                # narrowed pool -- so Category2's options only show values that
                # actually co-occur with whatever Category1 is currently chosen,
                # and changing Category1 gives Category2 a brand-new widget key
                # (defaulting back to "All") instead of erroring on a selection
                # that's no longer valid for the new Category1.
                cat_values = sorted({str(v) for v in cat_lookup_for_filter.reindex(filtered_products).dropna().unique()})
                widget_key = f"filter_{cat_col}_{prior_choice_for_key}"
                with filter_cols[i]:
                    chosen_cat = st.selectbox(f"Filter by {_pretty_label(cat_col)}", ["All"] + cat_values, key=widget_key)
                if chosen_cat != "All":
                    filtered_products = [p for p in filtered_products if str(cat_lookup_for_filter.get(p, "")) == chosen_cat]
                prior_choice_for_key += f"_{chosen_cat}"
            if not filtered_products:
                _ctl_card.warning("No products match this filter combination.")
                filtered_products = selected_products

        sort_col, search_col, focus_col = _ctl_card.columns([1.1, 1.4, 1])
        with sort_col:
            sort_options = ["Product ID", "Forecast revenue (high to low)", "Forecast units (high to low)",
                            "Biggest drop vs last year (units)", "Biggest rise vs last year (units)"]
            # Default to the biggest sellers: sorted by ID the page opened on an item
            # that barely sells, with an empty chart.
            sort_by = st.selectbox("Sort products by", sort_options, index=2)

        if sort_by == "Forecast revenue (high to low)":
            ordered_products = fc_revenue_by_product.sort_values(ascending=False).index.tolist()
        elif sort_by == "Forecast units (high to low)":
            ordered_products = fc_units_by_product.sort_values(ascending=False).index.tolist()
        elif sort_by in ("Biggest drop vs last year (units)", "Biggest rise vs last year (units)"):
            # Ranked on units, not %: by % the top was items going from 2 units to 12.
            _chg_units = fc_units_by_product.sub(last_year_units_by_product, fill_value=0)
            ordered_products = _chg_units.sort_values(
                ascending=sort_by.startswith("Biggest drop"), kind="stable").index.tolist()
        else:
            ordered_products = sorted(filtered_products)
        ordered_products = [p for p in ordered_products if p in filtered_products]

        with search_col:
            if len(filtered_products) > 30:
                focus_search = st.text_input("Search within selected products", placeholder="Type to filter")
            else:
                focus_search = ""
        focus_options = [p for p in ordered_products if not focus_search or focus_search.lower() in p.lower() or focus_search.lower() in str(code_lookup.get(p, "")).lower()]
        if not focus_options:
            focus_options = ordered_products

        def _plabel(pid):
            """'15959 - FFX' -- mirrors the Customer Detail selector, which shows
            'id - company name'. The internal ID is the grouping key and the only
            unique one, so it stays first; the item code is what people recognise."""
            code = str(code_lookup.get(pid, "")).strip()
            return f"{pid} - {code}" if code and code.lower() != "nan" else str(pid)

        with focus_col:
            focus_product = st.selectbox("Focus product", with_clicked(focus_options, "pd_focus"),
                                         format_func=_plabel, key="pd_focus")

        # The top 10 under the page's filters, on the chart's measure; a row click
        # opens that item. In the controls card so the page changes right below it.
        _pd_measure = "Revenue" if st.session_state.get("pd_metric") == "Revenue" else "Units"
        _pd_top, _pd_tot = top_products(hist_f[hist_f[col_product].isin(filtered_products)],
                                        fc_f[fc_f[col_product].isin(filtered_products)],
                                        col_product, _pd_measure, latest_year)
        with _ctl_card, st.expander(f"Top {len(_pd_top)} products: click one to open it",
                                    icon=":material/leaderboard:", expanded=True):
            _sfx = " (฿)" if _pd_measure == "Revenue" else ""
            _fc_col = f"FY{int(forecast_year)} forecast {_pd_measure.lower()}{_sfx}"
            _act_col = f"FY{int(latest_year)} actual {_pd_measure.lower()}{_sfx}"
            _tp = pd.DataFrame({"Rank": range(1, len(_pd_top) + 1),
                                "Item Code": [str(code_lookup.get(p, "") or "") for p in _pd_top.index]})
            if "Category1" in hist_f.columns:
                _tp["Category1"] = _pd_top.index.map(level_map_from(hist_f, col_product, "Category1"))
            _tp[_fc_col], _tp[_act_col] = _pd_top["Forecast"].values, _pd_top["Actual"].values
            _chg = (_tp[_fc_col] / _tp[_act_col].replace(0, np.nan) - 1) * 100
            _disp = prettify(_tp)
            _disp["Change"], _chg_cfg = change_pills("Change", _chg)
            ranking_table(_disp.style.format({_fc_col: "{:,.0f}", _act_col: "{:,.0f}"}), list(_pd_top.index),
                          "Product Detail", "pd_focus", "pd_top", height=fit_height(len(_tp)),
                          column_config={"Change": _chg_cfg, "Rank": st.column_config.Column(pinned=True),
                                         **item_pins(col_product)})
            if _pd_tot > 0:
                st.caption(f"Ranked by FY{int(forecast_year)} forecast {_pd_measure.lower()} (the chart's Units / "
                           f"Revenue switch), within the Scope and the category filters above. These "
                           f"{len(_pd_top)} are {_pd_top['Forecast'].sum() / _pd_tot:.0%} of it. Change = forecast "
                           f"vs FY{int(latest_year)} actual (blank where the item sold nothing that year).")
            _out = prettify(_tp.assign(**{c: _tp[c].round().astype("Int64") for c in (_fc_col, _act_col)},
                                       Change=_chg.map(lambda v: "" if pd.isna(v) else f"{v / 100:+.0%}")))
            st.download_button("Download top products (CSV)", _out.to_csv(index=False), "top_products.csv",
                               icon=":material/download:")

        prod_row = hist_f[hist_f[col_product] == focus_product]
        # The board's identity line -- the item, its production status as a pill
        # and one muted line of facts -- over four KPI cards. It replaced the row
        # of profile tiles, which said the same things in more space.
        _code = code_lookup.get(focus_product)
        _name = (f"{focus_product} - {_code}" if _code is not None and pd.notna(_code)
                 else str(focus_product))
        _status = (str(prod_row["ProductionStatus"].iloc[0])
                   if "ProductionStatus" in prod_row.columns and len(prod_row)
                   and pd.notna(prod_row["ProductionStatus"].iloc[0]) else "")
        # Continue / Discontinue are the item master's own words; anything else
        # ("To Be Advised") is a state, not a problem, so it stays neutral.
        _tone = {"Continue": "good", "Discontinue": "bad"}.get(_status, "neutral")
        _facts = [str(prod_row[c].iloc[0]) for c in available_category_cols
                  if len(prod_row) and pd.notna(prod_row[c].iloc[0])]
        _lp = (float(prod_row["ListPrice"].iloc[0])
               if "ListPrice" in prod_row.columns and len(prod_row) else 0.0)
        if _lp > 0:
            _facts.append(f"list price ฿{_lp:,.0f}")
        if "sparsity_coverage" in prod_row.columns and len(prod_row):
            _facts.append(f"sold in {prod_row['sparsity_coverage'].iloc[0]:.0%} of training months")
        _pill = f'<span class="fc-pill fc-pill--{_tone}">{html.escape(_status)}</span>' if _status else ""
        st.markdown(f'<div class="ident-row"><span class="ident-name">{html.escape(_name)}</span>'
                    f'{_pill}<span class="ident-meta">{html.escape(" · ".join(_facts))}</span></div>',
                    unsafe_allow_html=True)

        _fc_units = float(fc_units_by_product.get(focus_product, 0.0))
        _last_units = float(last_year_units_by_product.get(focus_product, 0.0))
        _var = variance_pct_by_product.get(focus_product)
        _fc_rows = fc_f[fc_f[col_product] == focus_product]
        _peak = (_fc_rows.loc[_fc_rows["Final_Forecast"].idxmax()]
                 if len(_fc_rows) and _fc_rows["Final_Forecast"].notna().any() else None)
        _hist_rows = hist_f[hist_f[col_product] == focus_product]
        _sold_months = int((_hist_rows["Units"] > 0).sum())
        # Revenue at this item's own average realised price, the same basis
        # Overview and Category Plan use -- not the list price the board shows,
        # which would give the app two different revenue definitions.
        _rev = float(fc_revenue_by_product.get(focus_product, 0.0))
        _avg_price = float(avg_price.get(focus_product, 0.0))
        _k1, _k2, _k3, _k4 = st.columns(4)
        with _k1:
            kpi_card(f"FY{int(forecast_year)} forecast (units)", f"{_fc_units:,.0f}",
                     _var if _last_units else None,
                     delta_text=f"{abs(_var):.1%} vs FY{int(latest_year)}" if _last_units else None,
                     note=f"FY{int(latest_year)} actual: {_last_units:,.0f} units")
        with _k2:
            kpi_card("Forecast per month", f"{_fc_units / 12:,.0f}",
                     note=("Average across the 12 months.  Peak: "
                           f"{fmonth_labels[int(_peak['FiscalMonth']) - 1]}, "
                           f"{_peak['Final_Forecast']:,.0f} units" if _peak is not None
                           else "Average across the 12 months"))
        with _k3:
            kpi_card("Forecast revenue", short_baht(_rev), exact=f"฿{_rev:,.0f}",
                     note=f"Forecast units x ฿{_avg_price:,.0f}, this item's average realised price")
        with _k4:
            kpi_card("Months with sales", f"{_sold_months:,} of {len(_hist_rows):,}",
                     note=("Sold in every month of the history" if _sold_months == len(_hist_rows)
                           else "Months with any units sold, since the history starts"))

        # Order rhythm: when this item usually sells and how much. Read from its own
        # history, not predicted -- a customer x item reorder model lost to the main model.
        _hr = hist_f[hist_f[col_product] == focus_product].sort_values(["FiscalYear", "FiscalMonth"])
        _rh = order_rhythm(_hr["Units"].to_numpy()) if len(_hr) else None
        if _rh is not None:
            _p0 = int(_hr["FiscalYear"].iloc[0]) * 12 + int(_hr["FiscalMonth"].iloc[0]) - 1

            def _mlabel(idx):
                # Calendar year, as the charts label months: "Feb FY2024" read as Feb 2024.
                _y, _m = divmod(_p0 + idx, 12)
                return f"{fmonth_labels[_m]} {_y + (_m + int(fiscal_start_month) > 12)}"

            _gap = _rh["gap"]
            _tag = _rh["last_vs_typical"]
            if _gap is None:
                _n = _rh["sale_months"]
                _next = ("Next sale", "Not enough sales", f"Only {_n} month{'s' if _n > 1 else ''} with sales")
            elif _rh["next_idx"] >= _rh["months"]:
                _next = ("Next sale expected", f"Around {_mlabel(_rh['next_idx'])}", "If the usual gap holds")
            else:
                _next = ("Next sale expected", "Overdue", f"Was due around {_mlabel(_rh['next_idx'])}")
            _tiles = [
                ("Last sale", _mlabel(_rh["last_idx"]),
                 f"{_rh['last_size']:,.0f} units" + (f", {_tag} for this item" if _tag else "")),
                ("Usual gap", "-" if _gap is None else "Every month" if _gap <= 1 else f"Every {_gap:g} months",
                 f"Sold in {_rh['sale_months']} of {_rh['months']} months"),
                _next,
                ("Typical month's sale", f"{_rh['typical_size']:,.0f} units",
                 f"Usually {_rh['size_lo']:,.0f} to {_rh['size_hi']:,.0f}"),
            ]
            with st.container(border=True, key="pd_rhythm"):
                st.markdown("##### Order rhythm", help=(
                    "When this item usually sells and how much, from its own sales history. A 'sale' is a "
                    "month with any units sold. Usual gap = median months between those months (shown from 3 selling "
                    "months up); next sale = last sale + usual gap. Big / small compares the last sale with this item's middle half "
                    "of selling months (above the 75th or below the 25th percentile). Descriptive, not a "
                    "forecast: for timing, an order backlog from MELFT would beat any pattern in the past."))
                st.caption("From past sales only. The forecast below decides stock; this shows the buying pattern.")
                for _col, (_lbl, _val, _note) in zip(st.columns(len(_tiles)), _tiles):
                    with _col:
                        profile_card(_lbl, _val, _note, inset=True)

        with st.container(border=True, key="pd_chart"):
            tg1, tg2 = st.columns([2.6, 2.15])
            with tg1:
                st.markdown("##### History and Forecast")
            with tg2, st.container(horizontal=True, horizontal_alignment="right"):
                pd_metric = st.segmented_control("Show", ["Units", "Revenue"], default="Units", required=True, key="pd_metric", label_visibility="collapsed")
                pd_gran = st.segmented_control("Granularity", ["Monthly", "Quarterly"], default="Monthly", required=True, key="pd_gran", label_visibility="collapsed")
            pd_rev = pd_metric == "Revenue"
            pd_price = float(price_per_item(history, col_product).get(focus_product, np.nan))
            if pd_rev and (np.isnan(pd_price) or pd_price <= 0):
                pd_price = float(price_per_item(history, col_product).mean())

            _unit = "Revenue (฿)" if pd_rev else "Units"
            h = hist_f[hist_f[col_product] == focus_product].copy()
            f = fc_f[fc_f[col_product] == focus_product].copy()
            try:
                h["Value"] = h["Revenue"] if pd_rev else h["Units"]
                h = h[["FiscalYear", "FiscalMonth", "Value"]]

                f["FiscalYear"] = int(forecast_year)
                bdf = None
                if {"q10", "q90"}.issubset(f.columns):
                    bmul = pd_price if pd_rev else 1.0
                    bdf = f[["FiscalYear", "FiscalMonth"]].assign(lo=f["q10"] * bmul, hi=f["q90"] * bmul)
                f["Value"] = f["Final_Forecast"] * pd_price if pd_rev else f["Final_Forecast"]
                f = f[["FiscalYear", "FiscalMonth", "Value"]]

                chart, legend = trend_chart(
                    h, f, bdf, pd_gran,
                    # Whole-unit ticks: a sparse item otherwise got an axis of 0.5, 1.5, ...
                    revenue_axis() if pd_rev else alt.Axis(tickMinStep=1, format=",d"), _unit, 380)
                st.markdown(legend, unsafe_allow_html=True)
                st.altair_chart(chart, width="stretch")
            except Exception as e:
                st.error(f"Couldn't build the chart for this product: {e}")

        _md_card = st.container(border=True, key="pd_monthly")
        _md_card.markdown("##### Monthly detail")
        _ly = int(latest_year)
        _md_card.caption(f"**Forecast** is the number the stocking plan is built on, set beside the same "
                         f"month of FY{_ly}. **Low / High** is the p10-p90 range shaded on the chart.")
        # The board's columns (forecast beside last year's actual) plus the p10-p90
        # range the chart shades. Baseline and Expected (p50) stay on Forecast Table:
        # here they read as rival forecasts.
        _raw_rows = fc_f[fc_f[col_product] == focus_product].sort_values("FiscalMonth")
        _mul = pd_price if pd_rev else 1.0
        _ly_rows = hist_f[(hist_f[col_product] == focus_product) & (hist_f["FiscalYear"] == latest_year)]
        _ly_act = (_ly_rows.groupby("FiscalMonth")["Revenue" if pd_rev else "Units"].sum()
                   .reindex(_raw_rows["FiscalMonth"], fill_value=0).to_numpy(dtype=float).round(0))
        _fc = (_raw_rows["Final_Forecast"].to_numpy(dtype=float) * _mul).round(0)

        # Pills as on Category Plan: orange only where the fall is bigger than the
        # model's own measured under-forecast, which every item carries.
        try:
            _pb = float(eval_report["summary"].loc[model_name, "Bias"]) if eval_report is not None else 0.0
        except Exception:
            _pb = 0.0
        _chg_cells, _chg_cfg = change_pills(
            f"vs FY{_ly}", [(f - a) / a * 100 if a > 0 and not np.isnan(f) else None for f, a in zip(_fc, _ly_act)],
            warn_below=round(abs(_pb) * 100) if _pb < 0 else 0)
        show = pd.DataFrame({"Month": [fmonth_labels[m - 1] for m in _raw_rows["FiscalMonth"]],
                             "Forecast": _fc, f"FY{_ly} actual": _ly_act,
                             f"vs FY{_ly}": _chg_cells})
        _numcols = ["Forecast", f"FY{_ly} actual"]
        for q, lbl in (("q10", "Low (p10)"), ("q90", "High (p90)")):
            if q in _raw_rows.columns:
                show[lbl] = (_raw_rows[q].to_numpy(dtype=float) * _mul).round(0)
                _numcols.append(lbl)
        _md_card.dataframe(show.style.format({c: "{:,.0f}" for c in _numcols}, na_rep=""),
                           hide_index=True, height="content", column_config={f"vs FY{_ly}": _chg_cfg})
    _product_detail_page()

# ============================= PAGE: CUSTOMER DETAIL ==========================
elif page == "Customer Detail":
    cch = build_meta.get("cust_cat_history")
    ccustcol = build_meta.get("col_customer")
    if cch is None or ccustcol is None or not len(cch):
        empty_state("**Customer Detail needs a Customer ID column plus product categories** (item master). "
                    "Set both on Build Forecast and rebuild.", button="Go to Build Forecast")
        st.stop()
    # A fragment, like Product Detail: its filters and pickers rerun only this page.
    @st.fragment
    def _customer_detail_page():
        with st.container(key="plain_customer_detail"):
            _customer_detail_body()

    def _customer_detail_body():

        cat_levels = [c for c in ("Category1", "Category2", "Category") if c in cch.columns]
        main_cat = cat_levels[0]
        name_map = build_meta.get("cust_name_map", {}) or {}
        type_map = build_meta.get("cust_type_map", {}) or {}
        ind_map = build_meta.get("cust_industry_map", {}) or {}
        data_end_t = int(cch["FiscalYear"].max() * 12 + cch[cch["FiscalYear"] == cch["FiscalYear"].max()]["FiscalMonth"].max())

        # The Units/Revenue radio is rendered inside the chart card below, to match
        # Overview and Product Detail. But the customer ranking, the filter caption and
        # the profile line all sit ABOVE that card and need the chosen measure first,
        # so read it from session state here: Streamlit writes a widget's new value
        # into session state before rerunning the script, so this is always the current
        # selection, and it falls back to Units on the very first run.
        val = "Revenue" if st.session_state.get("cd_metric") == "Revenue" else "Units"
        cd_rev = val == "Revenue"

        ranked = cch.groupby(ccustcol)[val].sum().sort_values(ascending=False)
        all_custs = ranked.index.tolist()

        def _clabel(cid):
            nm = str(name_map.get(cid, "")).strip()
            return f"{cid} - {nm}" if nm and nm.lower() != "nan" else str(cid)

        _JUNK = {"", "nan", "none", "- none -", "other"}

        def _clean(v):
            return v if str(v).strip().lower() not in _JUNK else None

        # --- Filters: customer type -> FA sub-industry (cascading) + free-text search.
        # The industry list is rebuilt from only the customers that match the chosen
        # type, so you can't land on a type/industry pair that has no customers. The
        # industry widget's key embeds the chosen type, so changing type resets the
        # industry back to "All" instead of erroring on a now-invalid value.
        types_present = sorted({str(type_map[c]) for c in all_custs if _clean(type_map.get(c))})
        # Same shape as Product Detail: one bordered card, filters on the first
        # row, sort / search / focus picker on the second.
        _ctl_card = st.container(border=True)
        fcol1, fcol2 = _ctl_card.columns(2)
        with fcol1:
            sel_type = st.selectbox("Customer type", ["All"] + types_present) if types_present else "All"
        pool_for_ind = [c for c in all_custs if sel_type == "All" or str(type_map.get(c)) == sel_type]
        inds_present = sorted({str(ind_map[c]) for c in pool_for_ind if _clean(ind_map.get(c))})
        with fcol2:
            sel_ind = (st.selectbox("FA Sub Industry", ["All"] + inds_present, key=f"cd_ind_{sel_type}")
                       if inds_present else "All")
        # Sort, search, then the customer picker. The option labels carry no
        # "units"/"revenue" word -- Streamlit resets a selectbox whose options
        # change, so flipping the measure would silently throw the chosen sort away.
        # The totals themselves follow the measure.
        sort_col, search_col, focus_col = _ctl_card.columns([1.2, 1.4, 1])
        _fy_last = int(cch["FiscalYear"].max())
        _srt_life, _srt_latest = "Lifetime total (high to low)", f"FY{_fy_last} total (high to low)"
        _srt_growth = f"Growth FY{_fy_last - 1} to FY{_fy_last} (high to low)"
        _srt_dormant, _srt_name = "Months since last order (dormant first)", "Customer name (A-Z)"
        with sort_col:
            cust_sort = st.selectbox(
                "Sort customers by", [_srt_life, _srt_latest, _srt_growth, _srt_dormant, _srt_name],
                help="Totals follow the Units / Revenue switch on the chart below.")
        with search_col:
            cust_search = st.text_input("Search customer (ID or name)", placeholder="Type to filter")

        # Every ordering is derived from the same customer x year totals on whichever
        # measure the page is showing. A customer the sort key can't score (nothing in
        # the latest year, no earlier year to grow from) keeps its lifetime rank behind
        # the ones it can, rather than dropping out of the list: the sort decides the
        # order, only the filters decide who appears.
        def _ordered(keys):
            return (pd.Series(keys, dtype="float64").reindex(all_custs)
                    .sort_values(ascending=False, na_position="last", kind="stable").index.tolist())

        if cust_sort == _srt_name:
            all_custs = sorted(all_custs, key=lambda c: _clabel(c).lower())
        elif cust_sort == _srt_dormant:
            _t = (cch["FiscalYear"] * 12 + cch["FiscalMonth"]).astype(int)
            all_custs = _ordered(data_end_t - _t.groupby(cch[ccustcol]).max())
        elif cust_sort != _srt_life:
            _cy = cch.groupby([ccustcol, "FiscalYear"])[val].sum()
            _last = _cy.xs(_fy_last, level="FiscalYear")
            if cust_sort == _srt_latest:
                all_custs = _ordered(_last)
            elif (_fy_last - 1) in set(_cy.index.get_level_values("FiscalYear")):
                _prev = _cy.xs(_fy_last - 1, level="FiscalYear").reindex(all_custs)
                all_custs = _ordered((_last.reindex(all_custs).fillna(0.0) - _prev) / _prev.replace(0, np.nan))
            else:
                st.caption(f"No FY{_fy_last - 1} history to compare against -- keeping the lifetime order.")

        def _passes(c):
            if sel_type != "All" and str(type_map.get(c)) != sel_type:
                return False
            if sel_ind != "All" and str(ind_map.get(c)) != sel_ind:
                return False
            if cust_search:
                s = cust_search.lower()
                if s not in str(c).lower() and s not in str(name_map.get(c, "")).lower():
                    return False
            return True

        eligible = [c for c in all_custs if _passes(c)]
        _no_match = not eligible
        if _no_match:
            eligible = all_custs
        with focus_col:
            focus_cust = st.selectbox("Customer", with_clicked(eligible, "cd_focus"), format_func=_clabel,
                                      key="cd_focus")

        # The top 10 inside the type / industry filters (not the search box), on the
        # page's measure: ranked by the latest FY, lifetime beside it. A row click
        # opens that customer. In the controls card so the page changes right below it.
        _seg = [c for c in ranked.index if (sel_type == "All" or str(type_map.get(c)) == sel_type)
                and (sel_ind == "All" or str(ind_map.get(c)) == sel_ind)]
        _cd_top, _cd_tot = top_customers(cch, ccustcol, val, _fy_last, custs=_seg)
        with _ctl_card, st.expander(f"Top {len(_cd_top)} customers: click one to open it",
                                    icon=":material/leaderboard:", expanded=True):
            _sfx = " (฿)" if cd_rev else ""
            _fy_col, _life_col = f"FY{_fy_last} {val.lower()}{_sfx}", f"Lifetime {val.lower()}{_sfx}"
            _top = pd.DataFrame({"Rank": range(1, len(_cd_top) + 1),
                                 "Customer": [_clabel(c) for c in _cd_top.index],
                                 "Customer type": [str(type_map.get(c)) if _clean(type_map.get(c)) else ""
                                                   for c in _cd_top.index],
                                 _fy_col: _cd_top["FY"].values, _life_col: _cd_top["Lifetime"].values})
            ranking_table(_top.style.format({_fy_col: "{:,.0f}", _life_col: "{:,.0f}"}), list(_cd_top.index),
                          "Customer Detail", "cd_focus", "cd_top", height=fit_height(len(_top)))
            _seg_name = " · ".join(v for v in (sel_type, sel_ind) if v != "All")
            if _cd_tot > 0:
                st.caption(f"Ranked by FY{_fy_last} {val.lower()} (the chart's Units / Revenue switch). These "
                           f"{len(_top)} are {_cd_top['FY'].sum() / _cd_tot:.0%} of it across {len(_seg):,} customers"
                           + (f" ({_seg_name})." if _seg_name else "."))
            _top_out = _top.assign(**{c: _top[c].round().astype("Int64") for c in (_fy_col, _life_col)})
            st.download_button("Download top customers (CSV)", _top_out.to_csv(index=False), "top_customers.csv",
                               icon=":material/download:")
        if _no_match:
            st.warning("No customers match these filters -- showing all.")
        elif len(eligible) < len(all_custs):
            st.caption(f"{len(eligible):,} of {len(all_custs):,} customers match the filters, "
                       f"sorted by {cust_sort[0].lower() + cust_sort[1:]} on {val.lower()}.")
        cc = cch[cch[ccustcol] == focus_cust].copy()
        active = cc[cc[val] > 0]

        # The board's identity line -- name, an account-status pill and one muted
        # line of facts (customer type, FA sub industry) -- over three KPI cards.
        # Status reuses the same "months since last order vs typical gap" rule as
        # the per-category Reorder status table below, applied across every
        # category the account has bought (not derived, not a forecast).
        _acct_months = sorted(set((active["FiscalYear"] * 12 + active["FiscalMonth"]).astype(int)))
        _acct_status = "-"
        if len(_acct_months) >= 2:
            _typ = float(np.mean(np.diff(_acct_months)))
            _since = data_end_t - _acct_months[-1]
            _acct_status = "Overdue" if _since >= 1.3 * _typ else ("Due soon" if _since >= 0.8 * _typ else "On track")
        _acct_tone = {"Overdue": "bad", "Due soon": "warn", "On track": "good"}.get(_acct_status, "neutral")
        _meta_facts = []
        if _clean(type_map.get(focus_cust)):
            _meta_facts.append(str(type_map[focus_cust]))
        if _clean(ind_map.get(focus_cust)):
            _meta_facts.append(str(ind_map[focus_cust]))
        _pill = (f'<span class="fc-pill fc-pill--{_acct_tone}">{html.escape(_acct_status)}</span>'
                 if _acct_status != "-" else "")
        st.markdown(f'<div class="ident-row"><span class="ident-name">{html.escape(_clabel(focus_cust))}</span>'
                    f'{_pill}<span class="ident-meta">{html.escape(" · ".join(_meta_facts))}</span></div>',
                    unsafe_allow_html=True)

        yrs = sorted(int(y) for y in active["FiscalYear"].unique())
        _k1, _k2, _k3 = st.columns(3)
        with _k1:
            kpi_card("Lifetime units", f"{cc['Units'].sum():,.0f}", note="All orders in the sales history")
        with _k2:
            kpi_card(f"{_pretty_label(main_cat)} bought", f"{active[main_cat].nunique():,}" if yrs else "0",
                      note="Distinct product families ordered")
        with _k3:
            kpi_card("Active period", f"FY{yrs[0]}-{yrs[-1]}" if yrs else "-",
                      note=(f"FY{yrs[0]} to FY{yrs[-1]}: first and last fiscal year with an order" if yrs
                            else "No orders in the sales history"))

        with st.container(border=True, key="cd_chart"):
            # The title level with the Units/Revenue switch; the year switch on the row below.
            ch_l, ch_r = st.columns([1, 1.2], vertical_alignment="center")
            with ch_l:
                st.markdown(f"##### Purchases by {_pretty_label(main_cat)}", help=(
                    "Descriptive view of one account: what they've bought by product category, and a simple "
                    "next-year projection (their most recent full year, category by category) with a "
                    "reorder-due flag."))
            by_cy = cc.groupby([main_cat, "FiscalYear"], as_index=False)[val].sum()
            _cd_years = ["All years"] + [f"FY{y}" for y in sorted(by_cy["FiscalYear"].astype(int).unique())]
            # Years differ per customer: drop a stale pick before the widget draws.
            if st.session_state.get("cust_year") not in _cd_years:
                st.session_state.pop("cust_year", None)
            with ch_r, st.container(horizontal_alignment="right", key="plain_cd_controls"):
                st.segmented_control("Show", ["Units", "Revenue"], default="Units", required=True, key="cd_metric",
                         label_visibility="collapsed")
            # The year the share list reads; the chart keeps every year and
            # highlights this one.
            with st.container(horizontal_alignment="right", key="plain_cd_year"):
                cd_year = st.segmented_control("Year", _cd_years, default=_cd_years[-1], required=True,
                                               key="cust_year", label_visibility="collapsed")
            _cd_scale = category_color_scale(main_cat)
            # Label the axis "FY2021", not a bare "2021" -- the fiscal year runs
            # Apr-Mar, and every other chart in the app is labelled that way.
            by_cy["FY"] = "FY" + by_cy["FiscalYear"].astype(int).astype(str)
            _fy_order = [f"FY{y}" for y in sorted(by_cy["FiscalYear"].astype(int).unique())]
            # One bold total per bar, like the board -- the stack has no other
            # running total on screen.
            _fy_totals = by_cy.groupby("FY", as_index=False)[val].sum()
            _fmt_total = (lambda v: f"฿{v:,.0f}") if cd_rev else (lambda v: f"{v:,.0f}")
            _fy_totals["Label"] = _fy_totals[val].map(_fmt_total)
            _ymax = float(_fy_totals[val].max()) if len(_fy_totals) else 0.0
            # 15% headroom above the tallest bar so its total label isn't clipped
            # against the chart's top edge.
            _yscale = alt.Scale(domain=[0, _ymax * 1.15], nice=False) if _ymax > 0 else alt.Undefined
            bars = alt.Chart(by_cy).mark_bar().encode(
                x=alt.X("FY:N", title=None, sort=_fy_order, axis=alt.Axis(labelAngle=0)),
                y=alt.Y(f"{val}:Q", title=("Revenue (฿)" if cd_rev else "Units"), stack="zero",
                        scale=_yscale, axis=revenue_axis() if cd_rev else alt.Undefined),
                # Same pinned category20c scale as Category Plan, so a category keeps
                # its colour from page to page.
                # The pinned domain holds every category; the legend lists only this
                # customer's.
                # The share list beside the chart is the legend.
                color=alt.Color(f"{main_cat}:N", title=None, legend=None, scale=_cd_scale),
                opacity=(alt.value(1) if cd_year == "All years"
                         else alt.condition(alt.datum.FY == cd_year, alt.value(1), alt.value(0.3))),
                tooltip=[alt.Tooltip(f"{main_cat}:N", title="Category"), alt.Tooltip("FY:N", title="Fiscal year"),
                         alt.Tooltip(f"{val}:Q", format=",.0f")],
            )
            totals = alt.Chart(_fy_totals).mark_text(
                align="center", baseline="bottom", dy=-4, fontWeight=600, color=TEXT_MAIN,
            ).encode(x=alt.X("FY:N", sort=_fy_order), y=alt.Y(f"{val}:Q", scale=_yscale), text="Label:N")
            # How much of each category the account bought over all years, largest
            # first, each with its chart colour: the answer without hovering the stack.
            _in_year = by_cy if cd_year == "All years" else by_cy[by_cy["FY"] == cd_year]
            _share = _in_year.groupby(main_cat)[val].sum().sort_values(ascending=False)
            _share = _share[_share > 0]
            _dom = list(_cd_scale.domain)
            _tot = float(_share.sum()) or 1.0
            _rows = "".join(
                f'<div class="cd-share-row"><span class="cd-sw" style="background:'
                f'{CATEGORY20C[_dom.index(str(c)) % 20] if str(c) in _dom else "#B8B5AC"}"></span>'
                f'<span class="cd-share-name">{html.escape(str(c))}</span>'
                f'<span class="cd-share-val">{short_baht(v) if cd_rev else f"{v:,.0f}"}'
                f'<em>{v / _tot:.0%}</em></span></div>' for c, v in _share.items())
            _ch_c, _sh_c = st.columns([2.6, 1], gap="large")
            with _ch_c:
                # Grows to the share list's height, so it sits on the card's bottom edge.
                st.altair_chart((bars + totals).properties(height=340), width="stretch", height="stretch")
            with _sh_c:
                st.markdown(f'<div class="cd-share"><div class="cd-share-h">{cd_year}, largest first</div>'
                            f'{_rows or "No purchases this year."}</div>', unsafe_allow_html=True)

        latest_fy = int(cc["FiscalYear"].max())
        _fy_next = int(forecast_year)
        # The card and its top-right download slot are created here (same shape as
        # Inventory Plan's worklist cards) and filled below, once the projection
        # is computed.
        _lp_card = st.container(border=True, key="cd_projection")
        with _lp_card:
            _lp_hc, _lp_bc = st.columns([2, 1], vertical_alignment="top")
        _lp_hc.markdown(f"##### FY{_fy_next} projection by {_pretty_label(main_cat)}", help=(
            f"Starts from this customer's own most recent full year in each category, then re-weights it by how "
            f"the FY{_fy_next} forecast moves that category RELATIVE to the whole catalog. Their total stays "
            f"roughly flat -- we don't forecast per customer -- but the mix shifts toward what the model expects "
            f"to hold up. Reorder status compares months since their last order in the category to their own "
            f"typical gap."))

        # --- Category trend from the real model ---------------------------------
        # The projection used to be flat (next year = last year), which made the
        # "actual" and "projected" columns identical in every row. Instead, scale the
        # customer's own last-year volume by the category's modelled growth:
        #   growth = model FY{forecast_year} total  /  actual FY{latest_fy} total
        # computed on the same measure the page is showing (units or revenue).
        # Clipped to [0.25, 4] so a near-dead category the model expects to rebound
        # can't produce an absurd multiplier, and falling back to 1.0 (the old flat
        # behaviour) wherever there's no usable denominator.
        cat_growth = {}
        if main_cat in history.columns:
            try:
                _measure = "Revenue" if cd_rev else "Units"
                _hist_cat = history[history["FiscalYear"] == latest_fy].groupby(main_cat)[_measure].sum()
                _pcat = history.drop_duplicates(col_product).set_index(col_product)[main_cat]
                _fcat_key = forecast[col_product].map(_pcat)
                if cd_rev:
                    _price = price_per_item(history, col_product)
                    _fvals = forecast["Final_Forecast"] * forecast[col_product].map(_price).fillna(_price.mean())
                else:
                    _fvals = forecast["Final_Forecast"]
                _fc_cat = _fvals.groupby(_fcat_key).sum()
                _ratio = (_fc_cat / _hist_cat.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
                # Normalise against the catalog-wide ratio. The model under-forecasts
                # TOTAL demand by ~15-20% (see Model Accuracy -> Residual bias), so the
                # raw ratios are almost all negative and would read as "every category
                # is shrinking" when most of that is model bias, not market signal.
                # Dividing the common factor out leaves the part that is real: how this
                # category moves RELATIVE to the rest of the catalog.
                _common = _fc_cat.index.intersection(_hist_cat.index)
                _den = float(_hist_cat.loc[_common].sum())
                _overall = (float(_fc_cat.loc[_common].sum()) / _den) if _den else 0.0
                if _overall > 0:
                    _ratio = _ratio / _overall
                cat_growth = _ratio.clip(0.5, 2.0).to_dict()
            except Exception:
                cat_growth = {}

        _unit_sfx = " (฿)" if cd_rev else ""
        _last_col = f"FY{latest_fy} {val.lower()}{_unit_sfx}"
        _trend_col = "Trend vs catalog"
        _proj_col = f"Projected FY{_fy_next}{_unit_sfx}"
        rows = []
        for cat, g in active.groupby(main_cat):
            months = sorted(set((g["FiscalYear"] * 12 + g["FiscalMonth"]).astype(int)))
            status = "-"
            if len(months) >= 2:
                typ = float(np.mean(np.diff(months)))
                since = data_end_t - months[-1]
                status = "Overdue" if since >= 1.3 * typ else ("Due soon" if since >= 0.8 * typ else "On track")
            last_fy_val = cc[(cc[main_cat] == cat) & (cc["FiscalYear"] == latest_fy)][val].sum()
            _g = float(cat_growth.get(cat, 1.0))
            rows.append({"Category": cat, _last_col: last_fy_val,
                         _trend_col: (_g - 1) * 100, _proj_col: last_fy_val * _g,
                         "Reorder status": status})
        proj = pd.DataFrame(rows).sort_values(_proj_col, ascending=False)
        if not cat_growth:
            proj = proj.drop(columns=[_trend_col])
        # Both numeric columns are named explicitly. The previous format dict keyed
        # off `str(latest_fy) in column`, which matched "FY2024 units" but silently
        # skipped "Projected FY2025" -- so that column rendered without separators.
        with _lp_card:
            _statuses = {"Overdue": PILL["bad"], "Due soon": PILL["warn"], "On track": PILL["good"], "-": PILL["neutral"]}
            # Trend as pills, like Category Plan's "vs last FY"; the catalog-wide ratio
            # is already divided out, so any fall relative to the catalog is marked.
            _disp = proj.assign(**{"Reorder status": proj["Reorder status"].map(lambda v: [v])})
            _trend_cfg = {}
            if _trend_col in _disp.columns:
                _disp[_trend_col], _trend_cfg[_trend_col] = change_pills(_trend_col, proj[_trend_col])
            st.dataframe(
                _disp.style.format({_last_col: "{:,.0f}", _proj_col: "{:,.0f}"}),
                hide_index=True, height=fit_height(len(proj)),
                column_config={**_trend_cfg, "Category": st.column_config.TextColumn(width="medium"),
                               "Reorder status": st.column_config.MultiselectColumn(
                                   "Reorder status", options=list(_statuses), color=list(_statuses.values()))},
            )
            if cat_growth:
                st.caption(
                    f"**{_trend_col}** is each category's FY{_fy_next} forecast against its FY{latest_fy} actual with "
                    f"the catalog-wide ratio divided out, so +8% means *holds up 8% better than average*, not growth "
                    f"(that also cancels the model's under-forecast bias). Clipped 0.5x-2x."
                )
            # Round for the export as well as the display. The Styler's "{:,.0f}" only
            # formats what's on screen -- `proj.to_csv()` shipped the raw float, so the
            # downloaded file read "10290.565274137583" units.
            _proj_out = proj.copy()
            if _trend_col in _proj_out.columns:
                _proj_out[_trend_col] = _proj_out[_trend_col].map(lambda v: f"{v / 100:+.0%}")
            for _c in (_last_col, _proj_col):
                if _c in _proj_out.columns:
                    _proj_out[_c] = _proj_out[_c].round().astype("Int64")
        _lp_bc.download_button("Download customer projection (CSV)", _proj_out.to_csv(index=False),
                               f"customer_{focus_cust}_projection.csv", icon=":material/download:", width="stretch")
    _customer_detail_page()

# ============================= PAGE: FORECAST TABLE ============================
elif page == "Forecast Table":
    # Laid out to the Forecast Console board: a KPI trio, the forecast table with
    # a Pivot / Range switch, then the workbook card.
    _fy = int(forecast_year)
    # Order by the item's own FY total, biggest first. Sorting by product ID put
    # the sparse low-numbered items at the top, so the first screenful of a
    # 5,000-row table was nothing but zeros. The Products filter's search and
    # multiselect are how you find a specific item; this view should lead with
    # the items that actually carry volume.
    _fc_totals = fc_f.groupby(col_product)["Final_Forecast"].sum()
    show = fc_f.copy()
    show["_total"] = show[col_product].map(_fc_totals)
    show = show.sort_values(["_total", col_product, "FiscalMonth"], ascending=[False, True, True])
    show["Month"] = show["FiscalMonth"].apply(lambda m: fmonth_labels[m-1])
    _id_cols = [col_product]
    for _c in ("Item Code", "Category1"):
        if _c in hist_f.columns:
            show.insert(len(_id_cols), _c, show[col_product].map(
                hist_f.drop_duplicates(col_product).set_index(col_product)[_c]))
            _id_cols.append(_c)
    # Blank, not NaN: a NaN code or category would drop the item from the pivot and read "None".
    show[_id_cols[1:]] = show[_id_cols[1:]].fillna("")
    _n_items = show[col_product].nunique()
    _fy_start = pd.Timestamp(year=_fy, month=int(st.session_state["build_config"]["fiscal_start_month"]), day=1)
    _fy_span = f"{_fy_start:%B %Y} to {_fy_start + pd.DateOffset(months=11):%B %Y}"

    # Revenue at each item's average realised price, as on Overview and Product Detail.
    _price = price_per_item(history, col_product)
    _fc_rev = _fc_totals * _price.reindex(_fc_totals.index)

    _k1, _k2, _kr, _k3 = st.columns(4)
    with _k1:
        kpi_card("Products in the forecast", f"{_n_items:,}",
                 note=("Discontinued items are left out. They stay on the obsolescence list."
                       if build_meta.get("n_discontinued_excluded") else None))
    with _k2:
        kpi_card(f"FY{_fy} forecast (units)", f"{_fc_totals.sum():,.0f}",
                 note=f"Sum of the item forecasts, {_fy_span}")
    with _kr:
        kpi_card(f"FY{_fy} forecast (revenue)", short_baht(_fc_rev.sum()), exact=f"฿{_fc_rev.sum():,.0f}",
                 note="Forecast units × each item's average realised price")
    with _k3:
        if len(_fc_totals) and _fc_totals.sum() > 0:
            _top = _fc_totals.idxmax()
            _top_code = show.loc[show[col_product] == _top, "Item Code"].iloc[0] if "Item Code" in show.columns else None
            kpi_card("Largest item", html.escape(str(_top_code or _top)),
                     note=f"{_top} · {_fc_totals[_top]:,.0f} units, {_fc_totals[_top] / _fc_totals.sum():.1%} of the forecast")
        else:
            kpi_card("Largest item", "–", note="No item has forecast demand")

    _qcols = [c for c in ("q10", "q50", "q90") if c in show.columns]
    _range_label = "Range (p10 / p90)" if _qcols else "By month"
    with st.container(border=True, key="ft_table"):
        _ft_hc, _ft_bc = st.columns([3, 1], vertical_alignment="center")
        _ft_hc.markdown(f"##### FY{_fy} forecast by product")
        st.caption(f"{_n_items:,} products × 12 months, largest FY{_fy} total first.")
        with st.container(horizontal=True, vertical_alignment="center"):
            _layout = st.segmented_control("Layout", ["Pivot", _range_label], default="Pivot",
                                           required=True, key="layout_table", label_visibility="collapsed")
            if _layout == "Pivot":
                st.caption("Pivot shows the point forecast only. Switch to Range for the p10 and p90 per month."
                           if _qcols else "Pivot shows the point forecast. Switch to By month for the baseline beside it.")
            elif _qcols:
                st.caption("Actual demand should land at or below High (p90) about 9 months in 10.")
        _pin = item_pins(col_product)  # "Item code" on screen only; the CSV keeps the full name
        if _layout == "Pivot":
            pivoted = show.pivot_table(index=_id_cols, columns="Month", values="Final_Forecast", sort=False)
            pivoted = pivoted[[m for m in fmonth_labels if m in pivoted.columns]]  # fiscal month order, not alphabetical
            # Annual total up front, so the row ordering is self-evident and you can
            # read an item's yearly number without scrolling across 12 months.
            pivoted.insert(0, f"FY{_fy} total (units)", pivoted.sum(axis=1))
            pivoted.insert(1, f"FY{_fy} revenue (฿)",
                           pivoted.index.get_level_values(col_product).map(_fc_rev).to_numpy())
            out, _csv_name = pivoted.reset_index().round(0), "forecast_pivoted.csv"
        else:
            # Drop the raw "Model" column when it's identical to the shipped
            # "Forecast" everywhere (the usual case) -- it only differs on months the
            # model couldn't score and the baseline took over.
            _show_model = ("ML_Forecast" in show.columns and not np.allclose(
                show["ML_Forecast"].fillna(-1).to_numpy(dtype=float),
                show["Final_Forecast"].fillna(-1).to_numpy(dtype=float)))
            display_cols = _id_cols + ["Month", "Baseline_Forecast"] + (["ML_Forecast"] if _show_model else []) + \
                ["Final_Forecast"] + _qcols
            out, _csv_name = show[display_cols].round(0), "forecast_selected_products.csv"
        out = out.assign(**{c: out[c].astype("Int64") for c in out.select_dtypes("number").columns})
        _po = prettify(out)
        # Thousands separators, as everywhere else (a 5,000-row Styler would be slow).
        st.dataframe(_po, hide_index=True, column_config={**localized_numbers(_po), **_pin})
        st.caption(f"Units per month, {_fy_span}."
                   + (" Revenue = forecast units × each item's average realised price." if _layout == "Pivot" else ""))
        _ft_bc.download_button("Download forecast (CSV)", _po.to_csv(index=False).encode("utf-8"), _csv_name,
                               mime="text/csv", icon=":material/download:", width="stretch", key="csv_table")

    # --- Full multi-sheet workbook ----------------------------------------
    # This used to be written on EVERY render of this page. openpyxl takes
    # ~57s to serialise 250k history rows + the backtest + the forecast, and
    # st.download_button needs the bytes eagerly -- so opening this page (or
    # touching any widget on it) blocked for a minute, during which Streamlit
    # leaves the previous page's elements on screen greyed out. That was the
    # whole of the "the old page takes ages to clear" problem.
    # Now it is built on request and kept in session state, keyed on
    # build_version so a rebuild invalidates it.
    with st.container(border=True, key="ft_workbook"):
        _wb_hc, _wb_bc = st.columns([3, 1], vertical_alignment="center")
        _wb_hc.markdown("##### Download full workbook")
        _wb_hc.caption("One .xlsx with the aggregated history, backtest, accuracy report, forecast, inventory plan "
                       "and obsolescence list. It takes about a minute to write, so it's built on request.")
        _wb_key = (f"_workbook_v{st.session_state.get('build_version', 0)}_"
                   + "_".join(f"{_v:g}" for _v in build_meta["inv_params"].values()))
        # Same disappearing-button slot as the Model Accuracy experiments and the
        # Category Plan cross-check: shown only while no workbook exists for this
        # build, and cleared in the same run the build finishes -- the old
        # disabled= flag was read before the click, so the button stayed live for
        # one more run and a second click spent another minute rebuilding. A
        # failed build leaves it for a retry; a rebuild (new build_version) brings
        # it back.
        _wb_slot = _wb_bc.empty()
        if _wb_key not in st.session_state and _wb_slot.button("Build workbook (about 1 min)", width="stretch",
                                                                                 key="act_workbook"):
            with st.spinner("Writing sheets..."):
                try:
                    st.session_state[_wb_key] = build_full_workbook(
                        history, res_df, metrics, eval_report, forecast, inv_plan, obs_flags)
                    remember_result(_wb_key)
                    _wb_slot.empty()
                except Exception as e:
                    st.error(f"Workbook build failed: {e}")
        if _wb_key in st.session_state:
            _wb_bc.download_button("Download forecast_output.xlsx", st.session_state[_wb_key],
                                   "forecast_output.xlsx", icon=":material/download:", width="stretch")

# ============================= PAGE: INVENTORY PLAN ===========================
elif page == "Inventory Plan":
    # Laid out to the Forecast Console board: four counters, then the three
    # worklists a planner works through -- Reorder now, Over-stocked,
    # Obsolescence risk. One "Recommended stocking parameters" table used to
    # hold every row at once, which is a report, not a worklist; the CSV still
    # carries all of them.
    ip = build_meta.get("inv_params", {})
    st.markdown("#### Stocking plan", help=(
        "Safety stock is sized from each item's own forecast error, the supplier lead time, and the "
        "quarterly review cycle -- not a flat formula."))
    with st.container(border=True):
        st.markdown("##### Stocking rules", help=(
            "Change any rule and the plan is re-made at once from the same forecast -- no rebuild. "
            "Overview, the fill-rate replay on Model Accuracy and the workbook use the same rules."))
        st.caption("They turn the forecast into safety stock and reorder points. The demand forecast itself "
                   "does not change.")
        # One small bordered card per rule, two by two, as on the board: the input,
        # what it means, and its range.
        _rule_cards = [
            ("lead_time", "How long an order takes to arrive.", "Range 0 to 24, step 0.5",
             lambda: st.number_input("Supplier lead time (months)", min_value=0.0, max_value=24.0,
                                     value=float(ip["lead_time"]), step=0.5, key="inv_lt",
                                     on_change=_set_inv_rule, args=("lead_time", "inv_lt"))),
            ("review", "How often stock is checked and reordered.", "Range 1 to 12, step 1",
             lambda: st.number_input("Stock review cycle (months)", min_value=1.0, max_value=12.0,
                                     value=float(ip["review"]), step=1.0, key="inv_rp",
                                     on_change=_set_inv_rule, args=("review", "inv_rp"))),
            # Whole percents on screen; the rule itself stays a fraction.
            ("service", "Higher means more safety stock.", "Range 80% to 99%",
             lambda: st.slider("Target service level (fill rate)", min_value=80, max_value=99,
                               value=int(round(ip["service"] * 100)), step=1, format="%d%%", key="inv_sl_pct",
                               on_change=_set_inv_rule, args=("service", "inv_sl_pct", 100))),
            ("legacy", "Today's rule, safety stock = N months of average demand. Kept only for "
             "comparison: the plan shows how many items it leaves under-stocked.", "Range 0 to 12, step 0.5",
             lambda: st.number_input("Current flat rule (months)", min_value=0.0,
                                     max_value=12.0, value=float(ip["legacy"]), step=0.5, key="inv_legacy",
                                     on_change=_set_inv_rule, args=("legacy", "inv_legacy"))),
        ]
        for _row in (_rule_cards[:2], _rule_cards[2:]):
            for _col, (_name, _what, _range, _draw) in zip(st.columns(2), _row):
                with _col, st.container(key=f"rulecard_{_name}", height="stretch"):
                    _draw()
                    st.caption(_what)
                    st.caption(_range)
    if inv_plan is None:
        st.info("No inventory plan available for this run.")
    else:
        plan_f = inv_plan[inv_plan[col_product].isin(selected_products)].copy()
        active_plan = plan_f[plan_f["forecast_monthly_units"] > 0.5]

        legacy_n = ip.get("legacy", 1.5)
        _delta = active_plan["safety_stock_delta_vs_legacy"].astype(float)
        lower = int((_delta < 0).sum())
        higher = int((_delta > 0).sum())

        # The obsolescence list is filtered up here because its count is the
        # third counter at the top of the page, above its own section.
        # Respect the product selection, but NEVER drop a flagged item that the
        # selector cannot offer in the first place. `all_products` comes from
        # `forecast`, which excludes discontinued items -- so a plain
        # isin(selected_products) silently deleted all 447 of them, even though
        # the sidebar promises they still appear here. Keep anything selected,
        # plus anything outside the forecastable universe. (selected_products is
        # never empty here: the page stops earlier when nothing is selected.)
        obs_f = None
        if obs_flags is not None and len(obs_flags):
            obs_f = obs_flags[obs_flags[col_product].isin(set(selected_products))
                              | ~obs_flags[col_product].isin(set(all_products))]
        _obs_value = (float(obs_f["value_at_risk_thb"].sum())
                      if obs_f is not None and "value_at_risk_thb" in obs_f.columns else 0.0)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi_card("Reorder now", f"{higher:,}",
                     note=f"under-stocked: the {legacy_n:g}-month rule holds less than the plan")
        with k2:
            kpi_card("Over-stocked", f"{lower:,}",
                     note=f"the {legacy_n:g}-month rule holds more than the plan. Lower these to free cash.")
        with k3:
            if obs_f is None or not len(obs_f):
                kpi_card("Obsolescence risk", "0", note="nothing flagged with the current rules")
            else:
                kpi_card("Obsolescence risk", f"{len(obs_f):,}",
                         note=(f"{short_baht(_obs_value)} at list price. " if _obs_value else "")
                              + "Discontinued, dormant or declining: stop reordering.")
        with k4:
            if "safety_stock_value_thb" in plan_f.columns:
                # Scoped to the SAME items as the two counters beside it. Priced
                # across the whole catalog this read ~฿1.05bn, 44% of which sat
                # on items whose own recommended action is "forecast zero" --
                # so the KPI row was counting 953 items and pricing 5,259.
                # The full-catalog figure is stated in the caption below, not
                # dropped.
                _inv_active = float(active_plan["safety_stock_value_thb"].sum())
                kpi_card("Safety-stock investment", short_baht(_inv_active), exact=f"฿{_inv_active:,.0f}",
                         note=f"at list price, on the {len(active_plan):,} items")
            else:
                kpi_card("Safety-stock investment", "add list price")

        disp = plan_f.copy()
        if "Item Code" in hist_f.columns:
            disp.insert(1, "Item Code", disp[col_product].map(hist_f.drop_duplicates(col_product).set_index(col_product)["Item Code"]))
        _has_p90 = "safety_stock_p90" in disp.columns
        # The board's columns: forecast, current rule, recommended, p90, change,
        # its value, reorder point. The action column is gone -- each worklist IS
        # the action (its title says it).
        show_cols = [col_product] + (["Item Code"] if "Item Code" in disp.columns else []) + [
            "forecast_monthly_units", "legacy_safety_stock", "safety_stock",
        ] + (["safety_stock_p90"] if _has_p90 else []) + ["safety_stock_delta_vs_legacy"]
        if "list_price" in disp.columns:
            # What the raise or the cut is worth. Both factors are already whole
            # units / whole baht (inventory.py rounds before deriving anything),
            # so this is the plan's own arithmetic. Display only -- the exported
            # CSV keeps its own columns.
            disp["delta_value_thb"] = (disp["safety_stock_delta_vs_legacy"].astype("Int64")
                                       * disp["list_price"].astype("Int64"))
            show_cols.append("delta_value_thb")
        show_cols.append("reorder_point")

        # Shorter on-screen headers, as on the board. Display only: the CSV keeps
        # the full names, which verify_exports.py reads.
        _num = st.column_config.NumberColumn
        _rec = _pretty_label("safety_stock")
        _p90 = _pretty_label("safety_stock_p90")
        _id = _pretty_label(col_product)
        _stock_cfg = {
            **item_pins(col_product),
            _pretty_label("forecast_monthly_units"): _num("Forecast / mo", format="localized"),
            _pretty_label("legacy_safety_stock"): _num("Current", format="localized"),
            _rec: _num("Recommended", format="localized", min_value=0, step=1,
                       help="Edit any value before you download: the CSV carries your edits."),
            _pretty_label("delta_value_thb"): _num(format="localized"),
            _pretty_label("reorder_point"): _num("Reorder pt", format="localized"),
        }

        def _p90_label(p90, ratio):
            """The p90 figure, with how far it sits from the formula when the two
            disagree by more than 2x (the same test as the agreement rate below)."""
            if pd.notna(ratio) and ratio > 2:
                return f"{p90:,.0f} ({ratio:.1f}× higher)"
            if pd.notna(ratio) and 0 < ratio < 0.5:
                return f"{p90:,.0f} ({1 / ratio:.1f}× lower)"
            return f"{p90:,.0f}"

        def _worklist(rows, title, caption, change_word):
            """One stocking worklist: same columns and config, different filter.
            Returns the card (for a footer), its top-right slot (for the CSV
            button, filled once every list's edits are known) and the planner's
            edits as {item id: safety stock}."""
            card = st.container(border=True)
            with card:
                _hc, _bc = st.columns([2, 1], vertical_alignment="top")
                _hc.markdown(f"##### {title} · {len(rows):,} items")
                _hc.caption(caption)
                if not len(rows):
                    st.caption("Nothing on this list for the current product selection.")
                    return card, _bc, {}
                rows = rows.sort_values("forecast_monthly_units", ascending=False)
                view = prettify(rows[show_cols])
                # "Raise by 12" / "Lower by 12": each list holds one sign, so the
                # change shows as a size and the word says which way.
                _chg = _pretty_label("safety_stock_delta_vs_legacy")
                view[_chg] = view[_chg].abs()
                # Whole units on screen; the CSV keeps the plan's exact figure.
                _fcm = _pretty_label("forecast_monthly_units")
                view[_fcm] = view[_fcm].round().astype("Int64")
                _cfg = {**_stock_cfg, _chg: _num("Change", format=f"{change_word} by %d")}
                if _has_p90:
                    # A pill only where the methods disagree by over 2x; the
                    # others are the same pill in a transparent colour, so they
                    # read as plain numbers.
                    view[_p90] = [[_p90_label(p, r)] for p, r in
                                  zip(rows["safety_stock_p90"], rows["p90_vs_formula_ratio"])]
                    _opts = sorted({v[0] for v in view[_p90]})
                    _cfg[_p90] = st.column_config.MultiselectColumn(
                        "p90 method", options=_opts,
                        color=[PILL["warn"] if "×" in o else "#00000000" for o in _opts])
                # Keyed on the rows and their recommended values, so a rebuild, a
                # rule change or a new product selection starts from a clean table.
                _k = hash((title, st.session_state.get("build_version", 0),
                           tuple(view[_id]), tuple(view[_rec])))
                edited = st.data_editor(view, hide_index=True, height=fit_height(len(view)),
                                        column_config=_cfg, key=f"_inv_edit_{_k}",
                                        disabled=[c for c in view.columns if c != _rec])
            _changed = edited[_rec] != view[_rec]
            return card, _bc, dict(zip(edited.loc[_changed, _id], edited.loc[_changed, _rec]))

        _active = disp["forecast_monthly_units"] > 0.5
        _d = disp["safety_stock_delta_vs_legacy"].astype(float)

        _re_card, _re_slot, _re_edits = _worklist(
            disp[_active & (_d > 0)], "Reorder now",
            f"Raise safety stock to the recommended value, largest forecast demand first. Edit any recommended "
            f"value before you download. The download holds all {len(plan_f):,} items in the plan, including "
            f"the ones already at the right level.", "Raise")
        with _re_card:
            if _has_p90:
                # Measured on items with real forecast demand, not the whole catalog.
                # Across all 5,000+ rows the sparse tail (where one method or the
                # other rounds to ~0) dominates and drags the agreement rate down to
                # a number that says nothing about the items anyone actually orders.
                _v = active_plan[active_plan["safety_stock"] > 0]
                _agree = float((_v["p90_vs_formula_ratio"].between(0.5, 2.0)).mean()) if len(_v) else 0.0
                with st.expander("How are the two safety-stock numbers derived?"):
                    st.caption(
                        f"**Safety stock** is the formula (z x error x sqrt(lead + review)); the **p90 method** "
                        f"re-derives it from the model's own 90th percentile, assuming nothing about the error's "
                        f"shape. They agree within 2x on {_agree:.0%} of the {len(_v):,} items with forecast demand; "
                        f"where they diverge (orange), the higher number is the safer bet."
                    )

        _, _, _over_edits = _worklist(
            disp[_active & (_d < 0)], "Over-stocked",
            f"Today's flat {legacy_n:g}-month rule holds more than the plan needs. Lower the safety stock and "
            f"stop topping these up, to free the cash.", "Lower")

        # The planner's edits go into the CSV as their own column beside the
        # plan's number, so the model's recommendation is never overwritten.
        _plan_csv = prettify(plan_f)
        _edits = {**_re_edits, **_over_edits}
        if _edits:
            _plan_csv.insert(_plan_csv.columns.get_loc(_rec) + 1, "Safety stock (your edit)",
                             [_edits.get(p, s) for p, s in zip(plan_f[col_product], plan_f["safety_stock"])])
        _re_slot.download_button("Download inventory plan (CSV)", _plan_csv.to_csv(index=False),
                                 "inventory_plan.csv", icon=":material/download:", width="stretch")

        st.markdown("<br>", unsafe_allow_html=True)
        _obs_head = st.container(key="plain_obs_head")
        _obs_head.markdown("#### Obsolescence risk")
        if obs_f is None or not len(obs_f):
            st.success("No items flagged for obsolescence risk with the current rules.")
        else:
            _rc = obs_f["reasons"].astype(str) if "reasons" in obs_f.columns else pd.Series(dtype=str)
            _rn = [int(_rc.str.contains(k).sum()) for k in ("discontinued", "dormant", "declining")]
            _cover = float(ip.get("lead_time", 2.0)) + float(ip.get("review", 3.0))
            # The counts and the value used to be two KPI cards; on the board they
            # are this one line, under the counter at the top of the page.
            _cap_obs = []
            if len(_rc):
                _cap_obs.append(f"{_rn[0]:,} discontinued · {_rn[1]:,} dormant · {_rn[2]:,} declining"
                                + (" (some have two reasons)" if sum(_rn) > len(obs_f) else ""))
            if _obs_value:
                _cap_obs.append(f"**{short_baht(_obs_value)}** at risk, {_cover:g} months of cover per item")
            _cap_obs.append("Discontinued, dormant or sharply declining items -- stop reordering, plan a run-down. "
                            "Discontinued items always appear here, whatever the product selection.")
            _obs_cap = ("  ·  ".join(_cap_obs[:-1]) + (".  " if len(_cap_obs) > 1 else "") + _cap_obs[-1])
            obs_disp = obs_f.copy()
            _codes = build_meta.get("item_code_map") or code_lookup
            if _codes:
                obs_disp.insert(1, "Item Code", obs_disp[col_product].map(_codes).fillna(""))
            if "reasons" in obs_disp.columns:
                obs_disp["reasons"] = obs_disp["reasons"].map(pretty_reasons)
            _obs_fmt = {c: "{:,.0f}" for c in
                        ("trailing_actual_units", "forecast_next_units", "value_at_risk_thb")
                        if c in obs_disp.columns}
            with st.container(border=True):
                _hc, _bc = st.columns([2.3, 1], vertical_alignment="center")
                _hc.markdown("##### Flagged items", help="Largest value at risk first.")
                _bc.download_button("Download obsolescence list (CSV)", prettify(obs_disp).to_csv(index=False),
                                    "obsolescence_flags.csv", icon=":material/download:", width="stretch")
                st.caption(_obs_cap)
                # Shorter on-screen headers so "Valued from" fits beside the added Item
                # Code column. Display only: the CSV keeps the full names, which
                # verify_exports.py reads.
                # Each reason as its own pill (on screen only; the CSV keeps the text).
                _obs_view = prettify(obs_disp)
                _why = _pretty_label("reasons")
                _pills = {}
                if _why in _obs_view.columns:
                    _obs_view[_why] = _obs_view[_why].map(lambda t: [r for r in str(t).split(", ") if r])
                    _pills[_why] = reason_pills(_why, _obs_view[_why])
                st.dataframe(_obs_view.style.format(prettify_fmt(_obs_fmt)),
                             hide_index=True,
                             column_config={
                                 **_pills,
                                 **item_pins(col_product),
                                 "Units sold (last 12m)": st.column_config.Column("Sold, 12m"),
                                 "Forecast next 12m": st.column_config.Column("Forecast, 12m"),
                                 "Valued from": st.column_config.Column(width="medium"),
                             })
                if "value_basis" in obs_disp.columns:
                    _est = int(obs_disp["value_basis"].astype(str).str.contains("last 12m", case=False).sum())
                    st.expander("How is value at risk calculated?").caption(
                        f"**Value at risk** prices each item at **{_cover:g} months of cover** (lead time + review "
                        f"cycle), not a full year. **Valued from** names the source: real on-hand stock if loaded, "
                        f"else the forward forecast, else last year's sales ({_est:,} items, mostly discontinued). "
                        f"Real on-hand quantities are the highest-value data to ask MELFT for."
                    )
