"""Small helpers every page uses: column/label formatting, status pills, KPI cards,
page titles, empty states, ranking tables and navigation buttons.
"""
import html

import numpy as np
import pandas as pd
import streamlit as st

from ui.config import FEATURE_LABELS, PRETTY_COLS
from ui.styles import FC_SCALE_CSS, FC_SCALE_PAGES


def pretty_label(c):
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
    out = df.rename(columns=pretty_label)
    if isinstance(out.index.name, str) and (out.index.name in PRETTY_COLS or "_" in out.index.name):
        out.index.name = pretty_label(out.index.name)
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
    return {pretty_label(col_product): st.column_config.Column(pinned=True),
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
    return {pretty_label(k): v for k, v in fmt.items()}


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


def page_title(title, subtitle=None):
    """The large bold heading at the top of a page. A plain div rather than a
    Markdown heading, so Streamlit adds no anchor-link icon beside it."""
    sub = f'<div class="page-subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
    css = FC_SCALE_CSS if title in FC_SCALE_PAGES else ""
    st.markdown(f'{css}<div class="page-title">{html.escape(title)}</div>{sub}', unsafe_allow_html=True)


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
