"""Altair chart helpers: axes, period bucketing, the trend chart, legends and the
fill-rate frontier.
"""
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from forecast_core import fiscal_month_labels
from inventory import FAMILY_FLAT, FAMILY_MODEL, fill_rate_at_budget
from ui.components import kpi_card, short_baht
from ui.config import BORDER_CONTROL, CHART_ACTUAL, CHART_GRID, ORANGE, TEXT_MAIN, TEXT_MUTED


def revenue_axis():
    """Y axis for a baht chart. Vega's "~s" format uses SI prefixes, so a billion
    baht read "1G" (giga) -- meaningless to a sales audience. Label with the
    bn / M / K abbreviations the rest of the app already uses for money
    (e.g. "฿1.05bn", "฿537M"). abs() so the Model Accuracy error chart, which
    goes below zero, keeps its sign ("-200M")."""
    return alt.Axis(labelExpr=(
        "abs(datum.value) >= 1e9 ? format(datum.value / 1e9, '.2~f') + 'bn' : "
        "abs(datum.value) >= 1e6 ? format(datum.value / 1e6, '.2~f') + 'M' : "
        "abs(datum.value) >= 1e3 ? format(datum.value / 1e3, '.2~f') + 'K' : "
        "format(datum.value, '.2~f')"))


def monthly_series(long_df, group_cols, value_col, price=None, col_product=None):
    """Sum `value_col` by `group_cols`; if `price` is given, first convert
    to revenue (value x per-item price) so the same code path serves both
    the Units and Revenue views."""
    d = long_df
    if price is not None:
        d = d.copy()
        d["_v"] = d[value_col] * d[col_product].map(price).fillna(price.mean())
        value_col = "_v"
    return d.groupby(group_cols, as_index=False)[value_col].sum().rename(columns={value_col: "Value"})


def _fiscal_start():
    """Fiscal start month of the build on screen."""
    return int(st.session_state["build_config"]["fiscal_start_month"])


def fiscal_labels():
    """Month names in fiscal order for the build on screen (April first for MELFT)."""
    return fiscal_month_labels(_fiscal_start())


def bucket_periods(df, year_col, month_col, granularity):
    """Adds a Period label + a numeric `t` sort key for either a Monthly or a
    Quarterly view of a fiscal-month series. Quarterly buckets fiscal months
    1-3 -> Q1, 4-6 -> Q2, ... (Q1 is the start of the fiscal year, so for an
    April start Q1 = Apr-Jun). Callers then group by [t, Period, Series] and
    sum, so the same chart code serves both granularities."""
    d = df.copy()
    y = d[year_col].astype(int)
    m = d[month_col].astype(int)
    if granularity == "Quarterly":
        q = (m - 1) // 3 + 1
        d["Period"] = "FY" + y.astype(str) + "-Q" + q.astype(str)
        d["t"] = y * 4 + q
    else:
        # Calendar month and year ("Jan '26", as the design system labels months): "FY2025-Jan"
        # read as January 2025 when it is January 2026.
        start = _fiscal_start()
        fmonth_labels = fiscal_month_labels(start)
        cal_y = y + (m + start - 1 > 12)
        d["Period"] = m.map(lambda mm: fmonth_labels[mm - 1]) + " '" + (cal_y % 100).map("{:02d}".format)
        d["t"] = y * 12 + m
    d["FY"] = "FY" + y.astype(str)
    return d


def period_axis(period_order, granularity):
    """x axis for bucket_periods() output, labels level (never rotated, as the design
    system asks). Monthly: each fiscal quarter's first month, or every half-year past
    three years. Quarterly: every quarter, or Q1 only past three years."""
    long = len(period_order) > (36 if granularity == "Monthly" else 12)
    if granularity == "Monthly":
        fmonth_labels = fiscal_labels()
        keep = {fmonth_labels[i] for i in ((0, 6) if long else (0, 3, 6, 9))}
        values = [p for p in period_order if p.split(" ")[0] in keep]
    else:
        values = [p for p in period_order if not long or p.endswith("Q1")]
    return alt.Axis(values=values, labelAngle=0)


def frontier_points(frontier, service_level, legacy_months):
    """The three fill rates the plan is judged on: the forecast-driven plan at the
    build's service level, the flat rule read off at that SAME stock budget, and
    today's rule at its own budget. Shared by Model Accuracy (which also draws the
    frontier chart) and Overview (which only quotes the numbers), so the two pages
    can never disagree."""
    use_value = bool(frontier["Avg inventory value"].notna().all()) and float(frontier["Avg inventory value"].max()) > 0
    vcol = "Avg inventory value" if use_value else "Avg inventory (units)"
    mine = frontier[(frontier["Family"] == FAMILY_MODEL) & np.isclose(frontier["Setting"], service_level)].iloc[0]
    cur = frontier[(frontier["Family"] == FAMILY_FLAT) & np.isclose(frontier["Setting"], legacy_months)].iloc[0]
    flat_same = fill_rate_at_budget(frontier, float(mine[vcol]), value_col=vcol)

    def money(v):
        return short_baht(v) if use_value else f"{v:,.0f} units"

    return dict(use_value=use_value, value_col=vcol, mine=mine, cur=cur,
                flat_same=flat_same, money=money)


def fill_rate_kpis(pts, service_level, legacy_months, replay_year):
    """The three same-budget fill-rate cards (Overview and Model Accuracy)."""
    mine, cur, flat, money, vc = pts["mine"], pts["cur"], pts["flat_same"], pts["money"], pts["value_col"]
    _f1, _f2, _f3 = st.columns(3)
    with _f1:
        _pts = (mine["Fill rate"] - flat) * 100 if pd.notna(flat) else None
        kpi_card(f"Forecast-driven plan · {service_level:.0%} target", f"{mine['Fill rate']:.1%}",
                 delta=_pts,
                 delta_text=(f"{abs(_pts):.1f} pts {'above' if _pts >= 0 else 'below'} the flat rule "
                             f"at this budget" if _pts is not None else None),
                 note=f"of demand shipped from stock · {money(mine[vc])} held · FY{replay_year} replay")
    with _f2:
        kpi_card("Flat rule, same budget", f"{flat:.1%}" if pd.notna(flat) else "--",
                 note=("read off the flat-rule line at the plan's stock level" if pd.notna(flat)
                       else "outside the range the flat rule was tested over"))
    with _f3:
        kpi_card(f"Current rule · {legacy_months:g} months", f"{cur['Fill rate']:.1%}",
                 note=f"of demand shipped from stock · {money(cur[vc])} held")


def render_frontier(frontier, service_level, legacy_months, title, replay_year):
    """KPI row + 'fill rate vs stock held' card for one policy_frontier result,
    drawn on Model Accuracy at the build's inventory parameters. The chart takes
    the card's left side; the returned `text` column on its right is for the
    explanation. Also returns the numbers that explanation quotes."""
    _pts = frontier_points(frontier, service_level, legacy_months)
    use_value, vcol = _pts["use_value"], _pts["value_col"]
    mine, cur, flat_same, money = _pts["mine"], _pts["cur"], _pts["flat_same"], _pts["money"]
    fill_rate_kpis(_pts, service_level, legacy_months, replay_year)

    d = frontier.copy()
    d["Stock"] = d[vcol] / (1e9 if use_value else 1.0)
    x_title = "Average stock held (฿ bn)" if use_value else "Average stock held (units)"
    color = alt.Color("Family:N", scale=alt.Scale(domain=[FAMILY_MODEL, FAMILY_FLAT], range=[ORANGE, TEXT_MUTED]),
                      legend=alt.Legend(title=None, orient="top", labelLimit=360, columnPadding=24))
    # Dashed flat rule: the two lines stay apart without relying on colour alone.
    dash = alt.StrokeDash("Family:N", scale=alt.Scale(domain=[FAMILY_MODEL, FAMILY_FLAT], range=[[1, 0], [6, 4]]),
                          legend=None)
    x = alt.X("Stock:Q", title=x_title, scale=alt.Scale(zero=False))
    y = alt.Y("Fill rate:Q", title="Demand shipped from stock", axis=alt.Axis(format=".0%"), scale=alt.Scale(zero=False))
    tip = [alt.Tooltip("Policy:N"), alt.Tooltip("Fill rate:Q", format=".1%"),
           alt.Tooltip("Stock:Q", title=x_title, format=",.2f" if use_value else ",.0f"),
           alt.Tooltip("Items stocked out:Q", format=",")]
    lines = alt.Chart(d).mark_line(point=True, strokeWidth=2.5).encode(
        x=x, y=y, color=color, strokeDash=dash, order=alt.Order("Stock:Q"), tooltip=tip)
    marks = d[d["Policy"].isin([mine["Policy"], cur["Policy"]])].copy()
    marks["Label"] = np.where(marks["Policy"] == mine["Policy"],
                              f"Plan, {service_level:.0%} target: {mine['Fill rate']:.1%}",
                              f"Current rule, {legacy_months:g} months: {cur['Fill rate']:.1%}")
    points = alt.Chart(marks).mark_point(size=220, filled=True, opacity=1).encode(x=x, y=y, color=color, tooltip=tip)
    # Both lines rise to the right, so the clear space is above-left of the plan
    # point and below-right of the current-rule point (which sits near the y axis,
    # where a left-aligned label would run into the axis title).
    _txt = dict(fontWeight="bold", color=TEXT_MAIN)
    labels = (alt.Chart(marks[marks["Policy"] == mine["Policy"]])
              .mark_text(align="right", dx=-12, dy=-12, **_txt).encode(x=x, y=y, text="Label:N")
              + alt.Chart(marks[marks["Policy"] == cur["Policy"]])
              .mark_text(align="left", dx=12, dy=14, **_txt).encode(x=x, y=y, text="Label:N"))
    layers = [lines, points, labels]
    # The flat rule read off at the plan's budget: a hollow point straight below
    # the plan's, labelled below-right, where both rising lines leave space.
    if pd.notna(flat_same):
        same = pd.DataFrame({"Stock": [d.loc[d["Policy"] == mine["Policy"], "Stock"].iloc[0]],
                             "Fill rate": [flat_same],
                             "Label": [f"Flat rule, same budget: {flat_same:.1%}"]})
        layers += [alt.Chart(same).mark_point(size=200, filled=False, strokeWidth=2.5, color=TEXT_MUTED, opacity=1)
                   .encode(x=x, y=y, tooltip=[alt.Tooltip("Label:N", title="At the plan's budget")]),
                   alt.Chart(same).mark_text(align="left", dx=12, dy=16, **_txt).encode(x=x, y=y, text="Label:N")]
    card = st.container(border=True)
    with card:
        st.markdown(f"##### {title}")
        chart_col, text = st.columns([1.7, 1], gap="large")
        with chart_col:
            st.altair_chart(alt.layer(*layers).properties(height=340), width="stretch")
    return dict(mine=mine, cur=cur, flat_same=flat_same, value_col=vcol, money=money, text=text)


def trend_chart(actual, forecast, band, granularity, y_axis, y_title, height,
                target_pace=None, target_label="Target pace"):
    """History-vs-forecast chart drawn as the design system's ForecastChart (Overview
    and Product Detail): solid Actual line, dashed Forecast line with hollow points
    joined to the last actual, the p10-p90 band, a "Forecast →" divider, and the
    target pace as a thin dashed line.

    actual / forecast: [FiscalYear, FiscalMonth, Value]; band: [FiscalYear, FiscalMonth,
    lo, hi] or None. target_pace: the value each period has to reach; target_label is
    its hover text. Returns (chart, legend HTML to draw above it)."""
    both = bucket_periods(pd.concat([actual.assign(Series="Actual"), forecast.assign(Series="Forecast")],
                                    ignore_index=True), "FiscalYear", "FiscalMonth", granularity)
    combined = both.groupby(["t", "Period", "FY", "Series"], as_index=False)["Value"].sum().sort_values("t")
    # Explicit period order (history -> forecast), passed to EVERY layer's x encoding so
    # the axis order doesn't depend on which layer Altair reads first -- the band only
    # spans the forecast, so left to itself it would push the forecast to the left.
    order = combined.drop_duplicates("Period")["Period"].tolist()
    x = alt.X("Period:N", sort=order, title=None, axis=period_axis(order, granularity))
    y = alt.Y("Value:Q", title=y_title, axis=y_axis)
    _hist = combined[combined["Series"] == "Actual"]
    _fc = combined[combined["Series"] == "Forecast"]
    # The forecast line starts from the last actual, so the two read as one series.
    lines = pd.concat([combined, _hist.tail(1).assign(Series="Forecast")], ignore_index=True) \
        if len(_hist) and len(_fc) else combined
    series = dict(color=alt.Color("Series:N", scale=alt.Scale(domain=["Actual", "Forecast"], range=[CHART_ACTUAL, ORANGE]),
                                                  legend=None),
                  strokeDash=alt.StrokeDash("Series:N", scale=alt.Scale(domain=["Actual", "Forecast"], range=[[1, 0], [6, 4]]),
                                            legend=None))
    _when = "Quarter" if granularity == "Quarterly" else "Month"
    tip = [alt.Tooltip("Period:N", title=_when), alt.Tooltip("FY:N", title="Fiscal year"), "Series",
           alt.Tooltip("Value:Q", title=y_title, format=",.0f")]
    layers = [alt.Chart(lines).mark_line(strokeWidth=2).encode(x=x, y=y, **series)]
    # Actual dots only on a short series, as the board does; hollow forecast points always.
    if len(_hist) <= 30:
        layers.append(alt.Chart(_hist).mark_circle(size=24, color=CHART_ACTUAL, opacity=1).encode(x=x, y=y))
    layers.append(alt.Chart(_fc).mark_point(size=48, filled=True, fill="white", stroke=ORANGE, strokeWidth=2,
                                            opacity=1).encode(x=x, y=y))
    # Invisible hover targets on every period: a line mark's tooltip reports one point only.
    layers.append(alt.Chart(combined).mark_circle(size=160, opacity=0).encode(x=x, y=y, tooltip=tip))
    # A dashed rule where the forecast starts, labelled, so the boundary doesn't
    # depend on telling navy from orange.
    if len(_fc) and len(_hist):
        _start = pd.DataFrame({"Period": [_fc["Period"].iloc[0]], "Label": ["Forecast →"]})
        layers.append(alt.Chart(_start).mark_rule(color=BORDER_CONTROL, strokeDash=[3, 3]).encode(x=x))
        layers.append(alt.Chart(_start).mark_text(align="left", dx=6, fontSize=12, color=TEXT_MUTED)
                      .encode(x=x, y=alt.value(8), text="Label:N"))
    # The target, spread evenly over the forecast periods -- what each month (or
    # quarter) has to reach for the year to land on it. Named in the legend; the value is on hover.
    if target_pace:
        _pace = pd.DataFrame({"v": [float(target_pace)], "Label": [target_label]})
        layers.append(alt.Chart(_pace).mark_rule(color=TEXT_MAIN, strokeDash=[2, 4], strokeWidth=1).encode(
            y="v:Q", tooltip=[alt.Tooltip("Label:N", title="Target")]))
    has_band = band is not None
    if has_band:
        band = bucket_periods(band, "FiscalYear", "FiscalMonth", granularity)
        band = band.groupby(["t", "Period"], as_index=False)[["lo", "hi"]].sum().sort_values("t")
        layers.insert(0, alt.Chart(band).mark_area(color=ORANGE, opacity=0.15).encode(
            x=x, y="lo:Q", y2="hi:Q",
            tooltip=[alt.Tooltip("Period:N", title=_when), alt.Tooltip("lo:Q", title="p10", format=",.0f"),
                     alt.Tooltip("hi:Q", title="p90", format=",.0f")]))
    return (alt.layer(*layers).resolve_scale(y="shared").properties(height=height)
            .configure_axis(labelFontSize=12, labelColor=TEXT_MUTED, titleColor=TEXT_MUTED, domain=False,
                            ticks=False, gridColor=CHART_GRID)
            # A fixed y-axis width, so the plot starts at the same x on every chart and the
            # legend above (.fc-legend padding) lines up with it.
            .configure_axisY(minExtent=56, maxExtent=56)
            .configure_axisX(grid=False)), chart_legend(has_band, bool(target_pace))


def chart_legend(band, target):
    """The design system's ForecastChart legend as HTML: Altair's own legend can't mix a
    solid line, a dashed line and a shaded box in one row. Range and Target only when drawn."""
    def key(svg, label):
        return (f'<span><svg viewBox="0 0 28 10" aria-hidden="true">{svg}</svg>{label}</span>')
    keys = [key(f'<line x1="0" y1="5" x2="28" y2="5" stroke="{CHART_ACTUAL}" stroke-width="2"/>', "Actual"),
            key(f'<line x1="0" y1="5" x2="28" y2="5" stroke="{ORANGE}" stroke-width="2" stroke-dasharray="6 4"/>',
                "Forecast")]
    if band:
        keys.append(key(f'<rect x="0" y="1" width="28" height="8" fill="{ORANGE}" fill-opacity="0.15"/>',
                        "Range p10–p90"))
    if target:
        keys.append(key(f'<line x1="0" y1="5" x2="28" y2="5" stroke="{TEXT_MAIN}" stroke-width="1" '
                        f'stroke-dasharray="2 4"/>', "Target pace"))
    return f'<div class="fc-legend">{"".join(keys)}</div>'
