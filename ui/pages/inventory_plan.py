"""Inventory Plan page. render() draws it; the shell (dashboard_app.py) passes the build data it needs."""
import pandas as pd
import streamlit as st

from ui.components import (
    PILL, fit_height, item_pins, kpi_card, prettify, prettify_fmt, pretty_label, pretty_reasons,
    reason_pills, short_baht,
)
from ui.pipeline import set_inv_rule


def render(
        *, all_products, build_meta, code_lookup, col_product, hist_f, inv_plan, obs_flags,
        selected_products, **_):
    # Laid out to the Forecast Console board: four counters, then the three
    # worklists a planner works through -- Reorder now, Over-stocked,
    # Obsolescence risk. One "Recommended stocking parameters" table used to
    # hold every row at once, which is a report, not a worklist; the CSV still
    # carries all of them.
    ip = build_meta.get("inv_params", {})
    st.markdown("#### Stocking plan", help=(
        "Safety stock is sized from each item's own forecast error, the supplier lead time, and the "
        "quarterly review cycle -- not a flat formula."))
    with st.container(border=True):
        st.markdown("##### Stocking rules", help=(
            "Change any rule and the plan is re-made at once from the same forecast -- no rebuild. "
            "Overview, the fill-rate replay on Model Accuracy and the workbook use the same rules."))
        st.caption("They turn the forecast into safety stock and reorder points. The demand forecast itself "
                   "does not change.")
        # One small bordered card per rule, two by two, as on the board: the input,
        # what it means, and its range.
        _rule_cards = [
            ("lead_time", "How long an order takes to arrive.", "Range 0 to 24, step 0.5",
             lambda: st.number_input("Supplier lead time (months)", min_value=0.0, max_value=24.0,
                                     value=float(ip["lead_time"]), step=0.5, key="inv_lt",
                                     on_change=set_inv_rule, args=("lead_time", "inv_lt"))),
            ("review", "How often stock is checked and reordered.", "Range 1 to 12, step 1",
             lambda: st.number_input("Stock review cycle (months)", min_value=1.0, max_value=12.0,
                                     value=float(ip["review"]), step=1.0, key="inv_rp",
                                     on_change=set_inv_rule, args=("review", "inv_rp"))),
            # Whole percents on screen; the rule itself stays a fraction.
            ("service", "Higher means more safety stock.", "Range 80% to 99%",
             lambda: st.slider("Target service level (fill rate)", min_value=80, max_value=99,
                               value=int(round(ip["service"] * 100)), step=1, format="%d%%", key="inv_sl_pct",
                               on_change=set_inv_rule, args=("service", "inv_sl_pct", 100))),
            ("legacy", "Today's rule, safety stock = N months of average demand. Kept only for "
             "comparison: the plan shows how many items it leaves under-stocked.", "Range 0 to 12, step 0.5",
             lambda: st.number_input("Current flat rule (months)", min_value=0.0,
                                     max_value=12.0, value=float(ip["legacy"]), step=0.5, key="inv_legacy",
                                     on_change=set_inv_rule, args=("legacy", "inv_legacy"))),
        ]
        for _row in (_rule_cards[:2], _rule_cards[2:]):
            for _col, (_name, _what, _range, _draw) in zip(st.columns(2), _row):
                with _col, st.container(key=f"rulecard_{_name}", height="stretch"):
                    _draw()
                    st.caption(_what)
                    st.caption(_range)
    if inv_plan is None:
        st.info("No inventory plan available for this run.")
    else:
        plan_f = inv_plan[inv_plan[col_product].isin(selected_products)].copy()
        active_plan = plan_f[plan_f["forecast_monthly_units"] > 0.5]

        legacy_n = ip.get("legacy", 1.5)
        _delta = active_plan["safety_stock_delta_vs_legacy"].astype(float)
        lower = int((_delta < 0).sum())
        higher = int((_delta > 0).sum())

        # The obsolescence list is filtered up here because its count is the
        # third counter at the top of the page, above its own section.
        # Respect the product selection, but NEVER drop a flagged item that the
        # selector cannot offer in the first place. `all_products` comes from
        # `forecast`, which excludes discontinued items -- so a plain
        # isin(selected_products) silently deleted all 447 of them, even though
        # the sidebar promises they still appear here. Keep anything selected,
        # plus anything outside the forecastable universe. (selected_products is
        # never empty here: the page stops earlier when nothing is selected.)
        obs_f = None
        if obs_flags is not None and len(obs_flags):
            obs_f = obs_flags[obs_flags[col_product].isin(set(selected_products))
                              | ~obs_flags[col_product].isin(set(all_products))]
        _obs_value = (float(obs_f["value_at_risk_thb"].sum())
                      if obs_f is not None and "value_at_risk_thb" in obs_f.columns else 0.0)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi_card("Reorder now", f"{higher:,}",
                     note=f"under-stocked: the {legacy_n:g}-month rule holds less than the plan")
        with k2:
            kpi_card("Over-stocked", f"{lower:,}",
                     note=f"the {legacy_n:g}-month rule holds more than the plan. Lower these to free cash.")
        with k3:
            if obs_f is None or not len(obs_f):
                kpi_card("Obsolescence risk", "0", note="nothing flagged with the current rules")
            else:
                kpi_card("Obsolescence risk", f"{len(obs_f):,}",
                         note=(f"{short_baht(_obs_value)} at list price. " if _obs_value else "")
                              + "Discontinued, dormant or declining: stop reordering.")
        with k4:
            if "safety_stock_value_thb" in plan_f.columns:
                # Scoped to the SAME items as the two counters beside it. Priced
                # across the whole catalog this read ~฿1.05bn, 44% of which sat
                # on items whose own recommended action is "forecast zero" --
                # so the KPI row was counting 953 items and pricing 5,259.
                # The full-catalog figure is stated in the caption below, not
                # dropped.
                _inv_active = float(active_plan["safety_stock_value_thb"].sum())
                kpi_card("Safety-stock investment", short_baht(_inv_active), exact=f"฿{_inv_active:,.0f}",
                         note=f"at list price, on the {len(active_plan):,} items")
            else:
                kpi_card("Safety-stock investment", "add list price")

        disp = plan_f.copy()
        if "Item Code" in hist_f.columns:
            disp.insert(1, "Item Code", disp[col_product].map(hist_f.drop_duplicates(col_product).set_index(col_product)["Item Code"]))
        _has_p90 = "safety_stock_p90" in disp.columns
        # The board's columns: forecast, current rule, recommended, p90, change,
        # its value, reorder point. The action column is gone -- each worklist IS
        # the action (its title says it).
        show_cols = [col_product] + (["Item Code"] if "Item Code" in disp.columns else []) + [
            "forecast_monthly_units", "legacy_safety_stock", "safety_stock",
        ] + (["safety_stock_p90"] if _has_p90 else []) + ["safety_stock_delta_vs_legacy"]
        if "list_price" in disp.columns:
            # What the raise or the cut is worth. Both factors are already whole
            # units / whole baht (inventory.py rounds before deriving anything),
            # so this is the plan's own arithmetic. Display only -- the exported
            # CSV keeps its own columns.
            disp["delta_value_thb"] = (disp["safety_stock_delta_vs_legacy"].astype("Int64")
                                       * disp["list_price"].astype("Int64"))
            show_cols.append("delta_value_thb")
        show_cols.append("reorder_point")

        # Shorter on-screen headers, as on the board. Display only: the CSV keeps
        # the full names, which verify_exports.py reads.
        _num = st.column_config.NumberColumn
        _rec = pretty_label("safety_stock")
        _p90 = pretty_label("safety_stock_p90")
        _id = pretty_label(col_product)
        _stock_cfg = {
            **item_pins(col_product),
            pretty_label("forecast_monthly_units"): _num("Forecast / mo", format="localized"),
            pretty_label("legacy_safety_stock"): _num("Current", format="localized"),
            _rec: _num("Recommended", format="localized", min_value=0, step=1,
                       help="Edit any value before you download: the CSV carries your edits."),
            pretty_label("delta_value_thb"): _num(format="localized"),
            pretty_label("reorder_point"): _num("Reorder pt", format="localized"),
        }

        def _p90_label(p90, ratio):
            """The p90 figure, with how far it sits from the formula when the two
            disagree by more than 2x (the same test as the agreement rate below)."""
            if pd.notna(ratio) and ratio > 2:
                return f"{p90:,.0f} ({ratio:.1f}× higher)"
            if pd.notna(ratio) and 0 < ratio < 0.5:
                return f"{p90:,.0f} ({1 / ratio:.1f}× lower)"
            return f"{p90:,.0f}"

        def _worklist(rows, title, caption, change_word):
            """One stocking worklist: same columns and config, different filter.
            Returns the card (for a footer), its top-right slot (for the CSV
            button, filled once every list's edits are known) and the planner's
            edits as {item id: safety stock}."""
            card = st.container(border=True)
            with card:
                _hc, _bc = st.columns([2, 1], vertical_alignment="top")
                _hc.markdown(f"##### {title} · {len(rows):,} items")
                _hc.caption(caption)
                if not len(rows):
                    st.caption("Nothing on this list for the current product selection.")
                    return card, _bc, {}
                rows = rows.sort_values("forecast_monthly_units", ascending=False)
                view = prettify(rows[show_cols])
                # "Raise by 12" / "Lower by 12": each list holds one sign, so the
                # change shows as a size and the word says which way.
                _chg = pretty_label("safety_stock_delta_vs_legacy")
                view[_chg] = view[_chg].abs()
                # Whole units on screen; the CSV keeps the plan's exact figure.
                _fcm = pretty_label("forecast_monthly_units")
                view[_fcm] = view[_fcm].round().astype("Int64")
                _cfg = {**_stock_cfg, _chg: _num("Change", format=f"{change_word} by %d")}
                if _has_p90:
                    # A pill only where the methods disagree by over 2x; the
                    # others are the same pill in a transparent colour, so they
                    # read as plain numbers.
                    view[_p90] = [[_p90_label(p, r)] for p, r in
                                  zip(rows["safety_stock_p90"], rows["p90_vs_formula_ratio"])]
                    _opts = sorted({v[0] for v in view[_p90]})
                    _cfg[_p90] = st.column_config.MultiselectColumn(
                        "p90 method", options=_opts,
                        color=[PILL["warn"] if "×" in o else "#00000000" for o in _opts])
                # Keyed on the rows and their recommended values, so a rebuild, a
                # rule change or a new product selection starts from a clean table.
                _k = hash((title, st.session_state.get("build_version", 0),
                           tuple(view[_id]), tuple(view[_rec])))
                edited = st.data_editor(view, hide_index=True, height=fit_height(len(view)),
                                        column_config=_cfg, key=f"_inv_edit_{_k}",
                                        disabled=[c for c in view.columns if c != _rec])
            _changed = edited[_rec] != view[_rec]
            return card, _bc, dict(zip(edited.loc[_changed, _id], edited.loc[_changed, _rec]))

        _active = disp["forecast_monthly_units"] > 0.5
        _d = disp["safety_stock_delta_vs_legacy"].astype(float)

        _re_card, _re_slot, _re_edits = _worklist(
            disp[_active & (_d > 0)], "Reorder now",
            f"Raise safety stock to the recommended value, largest forecast demand first. Edit any recommended "
            f"value before you download. The download holds all {len(plan_f):,} items in the plan, including "
            f"the ones already at the right level.", "Raise")
        with _re_card:
            if _has_p90:
                # Measured on items with real forecast demand, not the whole catalog.
                # Across all 5,000+ rows the sparse tail (where one method or the
                # other rounds to ~0) dominates and drags the agreement rate down to
                # a number that says nothing about the items anyone actually orders.
                _v = active_plan[active_plan["safety_stock"] > 0]
                _agree = float((_v["p90_vs_formula_ratio"].between(0.5, 2.0)).mean()) if len(_v) else 0.0
                with st.expander("How are the two safety-stock numbers derived?"):
                    st.caption(
                        f"**Safety stock** is the formula (z x error x sqrt(lead + review)); the **p90 method** "
                        f"re-derives it from the model's own 90th percentile, assuming nothing about the error's "
                        f"shape. They agree within 2x on {_agree:.0%} of the {len(_v):,} items with forecast demand; "
                        f"where they diverge (orange), the higher number is the safer bet."
                    )

        _, _, _over_edits = _worklist(
            disp[_active & (_d < 0)], "Over-stocked",
            f"Today's flat {legacy_n:g}-month rule holds more than the plan needs. Lower the safety stock and "
            f"stop topping these up, to free the cash.", "Lower")

        # The planner's edits go into the CSV as their own column beside the
        # plan's number, so the model's recommendation is never overwritten.
        _plan_csv = prettify(plan_f)
        _edits = {**_re_edits, **_over_edits}
        if _edits:
            _plan_csv.insert(_plan_csv.columns.get_loc(_rec) + 1, "Safety stock (your edit)",
                             [_edits.get(p, s) for p, s in zip(plan_f[col_product], plan_f["safety_stock"])])
        _re_slot.download_button("Download inventory plan (CSV)", _plan_csv.to_csv(index=False),
                                 "inventory_plan.csv", icon=":material/download:", width="stretch")

        st.markdown("<br>", unsafe_allow_html=True)
        _obs_head = st.container(key="plain_obs_head")
        _obs_head.markdown("#### Obsolescence risk")
        if obs_f is None or not len(obs_f):
            st.success("No items flagged for obsolescence risk with the current rules.")
        else:
            _rc = obs_f["reasons"].astype(str) if "reasons" in obs_f.columns else pd.Series(dtype=str)
            _rn = [int(_rc.str.contains(k).sum()) for k in ("discontinued", "dormant", "declining")]
            _cover = float(ip.get("lead_time", 2.0)) + float(ip.get("review", 3.0))
            # The counts and the value used to be two KPI cards; on the board they
            # are this one line, under the counter at the top of the page.
            _cap_obs = []
            if len(_rc):
                _cap_obs.append(f"{_rn[0]:,} discontinued · {_rn[1]:,} dormant · {_rn[2]:,} declining"
                                + (" (some have two reasons)" if sum(_rn) > len(obs_f) else ""))
            if _obs_value:
                _cap_obs.append(f"**{short_baht(_obs_value)}** at risk, {_cover:g} months of cover per item")
            _cap_obs.append("Discontinued, dormant or sharply declining items -- stop reordering, plan a run-down. "
                            "Discontinued items always appear here, whatever the product selection.")
            _obs_cap = ("  ·  ".join(_cap_obs[:-1]) + (".  " if len(_cap_obs) > 1 else "") + _cap_obs[-1])
            obs_disp = obs_f.copy()
            _codes = build_meta.get("item_code_map") or code_lookup
            if _codes:
                obs_disp.insert(1, "Item Code", obs_disp[col_product].map(_codes).fillna(""))
            if "reasons" in obs_disp.columns:
                obs_disp["reasons"] = obs_disp["reasons"].map(pretty_reasons)
            _obs_fmt = {c: "{:,.0f}" for c in
                        ("trailing_actual_units", "forecast_next_units", "value_at_risk_thb")
                        if c in obs_disp.columns}
            with st.container(border=True):
                _hc, _bc = st.columns([2.3, 1], vertical_alignment="center")
                _hc.markdown("##### Flagged items", help="Largest value at risk first.")
                _bc.download_button("Download obsolescence list (CSV)", prettify(obs_disp).to_csv(index=False),
                                    "obsolescence_flags.csv", icon=":material/download:", width="stretch")
                st.caption(_obs_cap)
                # Shorter on-screen headers so "Valued from" fits beside the added Item
                # Code column. Display only: the CSV keeps the full names, which
                # verify_exports.py reads.
                # Each reason as its own pill (on screen only; the CSV keeps the text).
                _obs_view = prettify(obs_disp)
                _why = pretty_label("reasons")
                _pills = {}
                if _why in _obs_view.columns:
                    _obs_view[_why] = _obs_view[_why].map(lambda t: [r for r in str(t).split(", ") if r])
                    _pills[_why] = reason_pills(_why, _obs_view[_why])
                st.dataframe(_obs_view.style.format(prettify_fmt(_obs_fmt)),
                             hide_index=True,
                             column_config={
                                 **_pills,
                                 **item_pins(col_product),
                                 "Units sold (last 12m)": st.column_config.Column("Sold, 12m"),
                                 "Forecast next 12m": st.column_config.Column("Forecast, 12m"),
                                 "Valued from": st.column_config.Column(width="medium"),
                             })
                if "value_basis" in obs_disp.columns:
                    _est = int(obs_disp["value_basis"].astype(str).str.contains("last 12m", case=False).sum())
                    st.expander("How is value at risk calculated?").caption(
                        f"**Value at risk** prices each item at **{_cover:g} months of cover** (lead time + review "
                        f"cycle), not a full year. **Valued from** names the source: real on-hand stock if loaded, "
                        f"else the forward forecast, else last year's sales ({_est:,} items, mostly discontinued). "
                        f"Real on-hand quantities are the highest-value data to ask MELFT for."
                    )
