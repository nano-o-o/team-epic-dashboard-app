"""Build Forecast page. render() draws it; the shell (dashboard_app.py) passes the build data it needs."""
import html
import os
import time

import streamlit as st

from forecast_core import available_model_types
from ui.components import page_title, profile_card, short_model
from ui.config import (
    ALL_FEATURE_KEYS, DEFAULT_COL_CATEGORY, DEFAULT_COL_DATE, DEFAULT_COL_ITEM_CODE,
    DEFAULT_COL_PRICE, DEFAULT_COL_PRODUCT, DEFAULT_COL_STATUS, DEFAULT_COL_UNITS,
    DEFAULT_FISCAL_START_MONTH, DEFAULT_SHEET_NAME, FEATURE_DEFAULTS, FEATURE_TOGGLE_GROUPS,
)
from ui.pipeline import build_everything, build_progress, peek_file_settings
from ui.saved_build import remember_build


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
def render(*, render_build_card, **_):
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
            build_progress.cb = _on_stage
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
                build_progress.cb = None
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
                build_progress.cb = None
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
