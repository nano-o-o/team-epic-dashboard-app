"""Forecast Table page. render() draws it; the shell (dashboard_app.py) passes the build data it needs."""
import html

import numpy as np
import pandas as pd
import streamlit as st

from ui.components import (
    item_pins, kpi_card, localized_numbers, prettify, price_per_item, short_baht,
)
from ui.pipeline import build_full_workbook
from ui.saved_build import remember_result


def render(
        *, build_meta, col_product, eval_report, fc_f, fmonth_labels, forecast, forecast_year, hist_f,
        history, inv_plan, metrics, obs_flags, res_df, **_):
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
