"""The build: reads the upload, runs the whole forecast + inventory pipeline in one
cached function (build_everything), and the cached slow extras pages run on a button press
(ablation, walk-forward, cross-checks, policy frontier, workbook). Modelling itself lives in
forecast_core.py / inventory.py / classical.py.
"""
import contextvars
import io
import threading

import numpy as np
import pandas as pd
import streamlit as st

from forecast_core import (
    CUSTOMER_MIX_COLS, FEATURE_BASE, HORIZON_LABELS, QUANTILES, REORDER_CADENCE_COLS,
    UNKNOWN_LEVEL, add_sparsity_feature, aggregate_to_fiscal_monthly, apply_bias_correction,
    clean_dates, compute_bias_correction, compute_customer_mix_features,
    compute_distinct_customers_trailing, compute_feature_importance,
    compute_reorder_cadence_feature, compute_revenue, drop_duplicate_lines, drop_unpriced_items,
    evaluate_backtest, feature_ablation, filter_by_status, forecast_future, horizon_backtest,
    horizon_comparison, level_direct_forecast, level_map_from, level_method_comparison,
    merge_item_master, quantile_backtest, run_backtest, to_fiscal, train_final_model,
    walk_forward_backtest,
)
from inventory import (
    build_inventory_plan, forecast_error_sigma, plan_for_rules, policy_frontier,
    service_level_comparison,
)
from classical import aggregate_crosscheck, per_level_crosscheck
from ui.config import (
    DEFAULT_COL_DATE, DEFAULT_COL_ITEM_CODE, DEFAULT_COL_PRICE, DEFAULT_COL_PRODUCT,
    DEFAULT_COL_STATUS, DEFAULT_COL_UNITS, INV_DEFAULTS,
)
from ui.saved_build import remember_result


def set_inv_rule(name, key, scale=1):
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
build_progress = threading.local()


def _report_stage(msg):
    cb = getattr(build_progress, "cb", None)
    if cb is None or _in_cached_fn is None:
        return

    def _untracked():
        assert _in_cached_fn is not None
        _in_cached_fn.set(False)
        cb(msg)
    contextvars.copy_context().run(_untracked)


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
