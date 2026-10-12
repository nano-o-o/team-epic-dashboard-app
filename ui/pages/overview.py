"""Overview page. render() draws it; the shell (dashboard_app.py) passes the build data it needs."""
import html

import numpy as np
import altair as alt
import streamlit as st

from forecast_core import HORIZON_LABELS
from ui.charts import fill_rate_kpis, frontier_points, monthly_series, revenue_axis, trend_chart
from ui.components import (
    empty_state, go_to, open_item, price_per_item, short_baht, top_customers, top_products,
)
from ui.config import FRONTIER_FLAT_MONTHS, FRONTIER_SERVICE_LEVELS
from ui.pipeline import run_policy_frontier


# Laid out to the Forecast Console board: the year against its target, the trend
# that produced it, what needs acting on, and the decision the whole project is
# judged by. The Summary-by-Product table lives on Forecast Table, the build
# details in the sidebar card, and the ETS/SARIMA cross-check on Model Accuracy
# -> Diagnostics, so this page carries only what a planner opens it for.
def render(
        *, all_products, build_meta, code_lookup, col_product, eval_report, fc_f, forecast_year,
        hist_f, history, inv_plan, last_year_units, latest_year, model_name, obs_flags, res_df,
        selected_products, total_forecast_units, train_years, validation_year, yoy_growth, **_):
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
