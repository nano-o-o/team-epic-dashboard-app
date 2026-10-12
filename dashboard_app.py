"""MELFT demand forecast dashboard -- the Streamlit app shell. Runs locally; nothing leaves the machine.

    streamlit run dashboard_app.py

Upload the sales export on the Build Forecast page and click Build Forecast.

This file only draws what every page shares: the sidebar navigation and build card, the
build data every page reads, and the page header with the Scope filter. Each page lives in
ui/pages/<page>.py; the helpers they share are in ui/ (see ui/__init__.py for the map).
Modelling is in forecast_core.py, inventory.py and classical.py.
"""
import html
import os

import altair as alt
import streamlit as st

from forecast_core import UNKNOWN_LEVEL, level_map_from
from ui.charts import fiscal_labels
from ui.components import empty_state, page_title, pretty_label
from ui.config import ACC_SECTIONS, CATEGORY20C, COMPANY_NAME, LOGO_PATH, NAV_ICONS
from ui.glossary import glossary_sections, show_glossary
from ui.pipeline import live_inventory
from ui.saved_build import restore_build
from ui.styles import inject_global_css
from ui.pages import (
    build_forecast, category_plan, customer_detail, forecast_table, inventory_plan,
    model_accuracy, overview, product_detail,
)


st.set_page_config(page_title=f"{COMPANY_NAME} Forecast Dashboard", page_icon=":material/insights:",
                   layout="wide", initial_sidebar_state="expanded")
inject_global_css()


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


# ============================= PAGES ==============================================
# Build Forecast works before any build exists; every other page reads the build.
if page == "Build Forecast":
    build_forecast.render(render_build_card=render_build_card)
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
                       + ", ".join(pretty_label(c) for c in build_meta["unavailable_features"]))
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
fmonth_labels = fiscal_labels()  # needed on every page

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
            st.multiselect(pretty_label(c), _cat_opts[c], key=f"scope_{c}", placeholder="All")
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
                                 label_visibility="collapsed", format_func=pretty_label)
            st.segmented_control("Show", ["Units", "Revenue"], default="Units", required=True, key="cat_metric",
                                 label_visibility="collapsed")
    render_build_card("Whole catalog on this page", note=_catalog_note)
else:
    page_title(page, _subtitle)
    render_build_card("Whole catalog on this page", note=_catalog_note)
st.markdown("<br>", unsafe_allow_html=True)

# Each page's render() names the build data it reads as keyword arguments;
# the **_ in its signature ignores the rest of this dict.
page_data = dict(
    history=history, forecast=forecast, res_df=res_df, metrics=metrics, model_name=model_name,
    importance_df=importance_df, importance_method=importance_method, eval_report=eval_report,
    inv_plan=inv_plan, obs_flags=obs_flags, build_meta=build_meta, col_product=col_product,
    fiscal_start_month=fiscal_start_month, forecast_year=forecast_year, validation_year=validation_year,
    train_years=train_years, all_products=all_products, code_lookup=code_lookup,
    category_color_scale=category_color_scale, fmonth_labels=fmonth_labels,
)
if uses_product_filter:
    page_data.update(
        selected_products=selected_products, hist_f=hist_f, fc_f=fc_f, latest_year=latest_year,
        total_forecast_units=total_forecast_units, last_year_units=last_year_units, yoy_growth=yoy_growth,
    )
PAGES = {
    "Overview": overview, "Category Plan": category_plan, "Model Accuracy": model_accuracy,
    "Product Detail": product_detail, "Customer Detail": customer_detail,
    "Forecast Table": forecast_table, "Inventory Plan": inventory_plan,
}
PAGES[page].render(**page_data)
