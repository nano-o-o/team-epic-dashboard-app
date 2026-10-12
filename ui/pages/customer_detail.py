"""Customer Detail page. render() draws it; the shell (dashboard_app.py) passes the build data it needs."""
import html

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from ui.charts import revenue_axis
from ui.components import (
    PILL, change_pills, empty_state, fit_height, kpi_card, pretty_label, price_per_item,
    ranking_table, short_baht, top_customers, with_clicked,
)
from ui.config import CATEGORY20C, TEXT_MAIN


def render(*, build_meta, category_color_scale, col_product, forecast, forecast_year, history, **_):
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
            kpi_card(f"{pretty_label(main_cat)} bought", f"{active[main_cat].nunique():,}" if yrs else "0",
                      note="Distinct product families ordered")
        with _k3:
            kpi_card("Active period", f"FY{yrs[0]}-{yrs[-1]}" if yrs else "-",
                      note=(f"FY{yrs[0]} to FY{yrs[-1]}: first and last fiscal year with an order" if yrs
                            else "No orders in the sales history"))

        with st.container(border=True, key="cd_chart"):
            # The title level with the Units/Revenue switch; the year switch on the row below.
            ch_l, ch_r = st.columns([1, 1.2], vertical_alignment="center")
            with ch_l:
                st.markdown(f"##### Purchases by {pretty_label(main_cat)}", help=(
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
        _lp_hc.markdown(f"##### FY{_fy_next} projection by {pretty_label(main_cat)}", help=(
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
