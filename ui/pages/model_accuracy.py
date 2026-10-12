"""Model Accuracy page. render() draws it; the shell (dashboard_app.py) passes the build data it needs."""
import html

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from forecast_core import (
    HORIZON_LABELS, _bias_factor_series, forecast_bias as _bias_fn, wape as _wape_fn,
    group_importance_by_category,
)
from inventory import FAMILY_MODEL, fill_rate_at_budget
from ui.charts import render_frontier, revenue_axis
from ui.components import (
    PILL, classical_verdict, empty_state, fit_height, kpi_card, prettify, prettify_fmt,
    pretty_grain, pretty_label, price_per_item,
)
from ui.config import (
    CHART_ACTUAL, FRONTIER_FLAT_MONTHS, FRONTIER_SERVICE_LEVELS, ORANGE, TEXT_MAIN, TEXT_MUTED,
)
from ui.pipeline import (
    run_classical_crosscheck, run_feature_ablation, run_policy_frontier, run_service_simulation,
    run_walk_forward, run_year_backtest,
)
from ui.saved_build import remember_result


def render(
        *, build_meta, col_product, eval_report, fiscal_start_month, fmonth_labels, forecast,
        forecast_year, history, importance_df, importance_method, inv_plan, metrics, model_name,
        res_df, train_years, validation_year, **_):
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
                    top_n["Feature"] = top_n["Feature"].map(pretty_label)
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
                        _names = [pretty_label(f) for f in _ranked["Feature"].head(3)]
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
