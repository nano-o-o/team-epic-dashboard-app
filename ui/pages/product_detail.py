"""Product Detail page. render() draws it; the shell (dashboard_app.py) passes the build data it needs."""
import html

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from forecast_core import level_map_from, order_rhythm
from ui.charts import revenue_axis, trend_chart
from ui.components import (
    change_pills, fit_height, item_pins, kpi_card, prettify, pretty_label, price_per_item,
    profile_card, ranking_table, short_baht, top_products, with_clicked,
)


def render(
        *, code_lookup, col_product, eval_report, fc_f, fiscal_start_month, fmonth_labels,
        forecast_year, hist_f, history, latest_year, model_name, selected_products, **_):
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
                    chosen_cat = st.selectbox(f"Filter by {pretty_label(cat_col)}", ["All"] + cat_values, key=widget_key)
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
