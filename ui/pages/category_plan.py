"""Category Plan page. render() draws it; the shell (dashboard_app.py) passes the build data it needs."""
import html

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from forecast_core import (
    UNKNOWN_LEVEL, wape as _wape_fn, level_accuracy_breakdown, level_map_from,
    rollup_backtest_to_level,
)
from ui.charts import bucket_periods, period_axis, revenue_axis
from ui.components import (
    PILL, empty_state, fit_height, item_pins, kpi_card, prettify, prettify_fmt, pretty_label,
    price_per_item, short_baht,
)
from ui.config import GREEN, ORANGE, RED, TEXT_MAIN, TEXT_MUTED
from ui.pipeline import run_level_crosscheck
from ui.saved_build import remember_result


# The grain planning actually happens at. Three of the four sections are pure
# aggregation of the build's own outputs, so they cost nothing; only the
# aggregation-level cross-check retrains anything, and that sits behind a
# button like the other expensive analyses on Model Accuracy.
def render(
        *, build_meta, category_color_scale, code_lookup, col_product, eval_report, fiscal_start_month,
        forecast, forecast_year, history, metrics, model_name, res_df, train_years, validation_year,
        **_):
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
    _lvl_name = pretty_label(level_col)

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
            column_config={pretty_label(level_col): st.column_config.TextColumn(width="medium")},
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
    _xc_hc.markdown(f"##### Direct {pretty_label(level_col)} model", help=(
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
        with _xc_card, st.spinner(f"Forecasting at {pretty_label(level_col)} level and re-scoring..."):
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
                            title=(f"WAPE at {pretty_label(level_col)} × month" if _last else None),
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
