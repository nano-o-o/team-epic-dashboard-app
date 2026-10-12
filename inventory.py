"""
=============================================================================
 inventory.py -- turn the demand forecast into a stocking plan
=============================================================================
Bridges the gap the brief calls out as pain point #1: the current safety-
stock formula is a fixed rule that ignores lead time and demand variability.
Here safety stock is driven by (a) the forecast, (b) the model's OWN
historical error on that item, and (c) the actual lead time + the quarterly
review cycle.

Pure pandas/numpy (z-scores from the stdlib's NormalDist) -- no Streamlit -- so it can be tested standalone.

Key formulas (periodic-review, order-up-to-level system, which matches
"stock levels are reviewed quarterly"):

    protection period  P = lead_time + review_period          (months)
    expected demand over P     =  monthly_forecast * P
    demand sigma over P        =  sigma_month * sqrt(P)
    safety stock          SS   =  z(service_level) * sigma_month * sqrt(P)
    order-up-to level      S   =  monthly_forecast * P  +  SS
    reorder point        ROP   =  monthly_forecast * lead_time  +  SS
    suggested order qty        =  max(0, S - on_hand - on_order)

sigma_month is the standard deviation of the item's *monthly forecast
error* from the backtest -- i.e. how wrong the model has actually been --
falling back to a coefficient-of-variation assumption when an item has too
little history to estimate it.
=============================================================================
"""
from statistics import NormalDist

import numpy as np
import pandas as pd


def _z(service_level):
    return NormalDist().inv_cdf(float(np.clip(service_level, 0.5, 0.9999)))


def forecast_error_sigma(res_df, model_name, col_product):
    """Per-product standard deviation of the monthly forecast error, from the
    backtest results. res_df must contain 'Actual' and '<model_name>_Pred'.
    Returns a Series indexed by product."""
    pred_col = f"{model_name}_Pred"
    if res_df is None or pred_col not in res_df.columns:
        return pd.Series(dtype=float)
    err = res_df["Actual"] - res_df[pred_col]
    return err.groupby(res_df[col_product]).std().fillna(0.0)


def _resolve_sigma(products, monthly, sigma_by_product, fallback_cv):
    """Per-product sigma of monthly demand error: measured where we have it,
    else a coefficient-of-variation fallback on the forecast level. Shared by
    build_inventory_plan and policy_frontier so the two can never size a buffer
    differently."""
    sigma = np.full(len(monthly), np.nan)
    if sigma_by_product is not None and len(sigma_by_product):
        sigma = pd.Series(products).map(pd.Series(sigma_by_product)).to_numpy(dtype=float)
    fallback = fallback_cv * np.maximum(monthly, 1e-9)
    return np.where(np.isnan(sigma) | (sigma <= 0), fallback, sigma)


def _dormant_mask(products, monthly, trailing_actual_by_product):
    """True for items with no forecast demand AND no sales in the trailing
    window -- the ones that get no safety stock (see build_inventory_plan).
    Without trailing actuals nothing is dormant: a zero forecast alone is not
    evidence."""
    if not trailing_actual_by_product:
        return np.zeros(len(monthly), dtype=bool)
    t = pd.Series(products).map(pd.Series(trailing_actual_by_product)).fillna(0).to_numpy(dtype=float)
    return (monthly <= 0) & (t <= 0)


def build_inventory_plan(forecast_df, col_product, horizon_months=12,
                         lead_time_months=2.0, review_period_months=3.0, service_level=0.95,
                         sigma_by_product=None, fallback_cv=0.7,
                         on_hand=None, list_price=None,
                         legacy_safety_months=1.5,
                         monthly_p50_by_product=None, monthly_p90_by_product=None,
                         trailing_actual_by_product=None):
    """forecast_df: output of forecast_future (needs col_product + 'Final_Forecast'
    + 'FiscalMonth'). Returns one row per product with the recommended
    stocking parameters, plus the equivalent number under a naive
    'hold N months of average demand' fixed formula for side-by-side
    comparison.

    lead_time_months / review_period_months / service_level are scalars
    applied to every product. on_hand and list_price are optional
    {product -> value} mappings.

    trailing_actual_by_product: optional {product -> units sold in the last 12
    months}. The plan needs real recent sales in two places, because a zero
    forecast is this model's most common failure mode on ~78%-zero data and is
    NOT evidence that an item is dead: it stops `recommended_action` labelling
    a recently-sold item "no demand", and it zeroes the safety stock of items
    with no forecast AND no recent sales (whose sigma reflects the model's
    wobble around zero, not demand risk). Omit it and neither adjustment is
    applied -- the plan falls back to forecast-only behaviour.
    """
    plan = forecast_df.groupby(col_product)["Final_Forecast"].sum().rename("forecast_horizon_units").reset_index()
    plan["forecast_monthly_units"] = plan["forecast_horizon_units"] / max(horizon_months, 1)

    lt, rp = float(lead_time_months), float(review_period_months)
    z = _z(service_level)

    monthly = plan["forecast_monthly_units"].to_numpy(dtype=float)

    # sigma of monthly demand error: measured per-product where we have it,
    # else a coefficient-of-variation fallback on the forecast level.
    sigma = _resolve_sigma(plan[col_product], monthly, sigma_by_product, fallback_cv)

    trailing = pd.Series(trailing_actual_by_product) if trailing_actual_by_product else None

    protection = lt + rp
    plan["protection_period_months"] = protection
    plan["expected_demand_protection"] = monthly * protection
    safety = z * sigma * np.sqrt(np.maximum(protection, 0))

    # Items with no forecast demand AND no sales in the trailing window have
    # nothing to protect, so they get no buffer. Their sigma is non-zero only
    # because the model predicted small positive numbers against actual zeros
    # -- i.e. the formula was sizing a buffer against the model's own wobble
    # rather than against demand risk, and recommending stock for 334 items
    # that the very same table labelled "no demand". Zeroed rather than
    # dropped, so the item still appears in the plan with an explicit zero.
    # Needs actuals to tell dormant from active: without
    # trailing_actual_by_product nothing is zeroed, since a zero forecast
    # alone is not evidence (see the `_action` comment).
    safety = np.where(_dormant_mask(plan[col_product], monthly, trailing_actual_by_product), 0.0, safety)

    plan["safety_stock"] = safety
    plan["reorder_point"] = monthly * lt + plan["safety_stock"]
    plan["order_up_to_level"] = monthly * protection + plan["safety_stock"]

    # Legacy fixed-formula baseline: safety stock = a flat N months of
    # average demand for EVERY item, regardless of how variable it is. The
    # forecast-driven safety stock is LOWER for steady, predictable items
    # (capital freed up) and HIGHER only for the genuinely volatile ones
    # (a stockout risk the blanket rule was silently carrying).
    plan["legacy_safety_stock"] = monthly * legacy_safety_months
    # The delta is derived AFTER both quantities are rounded (see the rounding
    # block below), so the column is always exactly the difference of the two
    # columns beside it. Computed from the floats first, round(a) - round(b)
    # disagreed with round(a - b) on thousands of rows.

    # Quantile-based cross-check: instead of assuming demand is normal and
    # scaling its standard deviation, take the model's OWN 90th-percentile
    # forecast and hold enough to cover (p90 - p50) demand over the protection
    # window. If this lands near the sigma-based number, the safety stock is
    # well-founded from two independent angles; a big gap flags an item whose
    # error isn't bell-shaped (usually a lumpy, hard-to-predict SKU).
    if monthly_p50_by_product is not None and monthly_p90_by_product is not None:
        p50 = plan[col_product].map(pd.Series(monthly_p50_by_product)).to_numpy(dtype=float)
        p90 = plan[col_product].map(pd.Series(monthly_p90_by_product)).to_numpy(dtype=float)
        p50 = np.where(np.isnan(p50), monthly, p50)          # fall back to the point forecast
        p90 = np.where(np.isnan(p90), monthly, p90)
        plan["p50_monthly_units"] = p50
        plan["p90_monthly_units"] = p90
        plan["safety_stock_p90"] = np.maximum(0.0, (p90 - p50) * np.sqrt(np.maximum(protection, 0)))
        with np.errstate(divide="ignore", invalid="ignore"):
            plan["p90_vs_formula_ratio"] = np.where(
                plan["safety_stock"] > 0, plan["safety_stock_p90"] / plan["safety_stock"], np.nan)

    if on_hand is not None:
        oh = plan[col_product].map(pd.Series(on_hand)).fillna(0.0)
        plan["on_hand"] = oh
        plan["suggested_order_qty"] = np.maximum(0.0, plan["order_up_to_level"] - oh)
        plan["months_of_cover_on_hand"] = np.where(monthly > 0, oh / monthly, np.inf)

    # Stock is held and ordered in whole units -- round the quantity columns to
    # integers. This happens BEFORE the baht columns are derived, so every
    # value equals its own displayed quantity x price. Pricing the unrounded
    # float instead printed a baht figure against a zero quantity on 1,761
    # items ("safety stock 0, value 617") and overstated the catalog total by
    # ~1.7%. forecast_monthly_units stays at 1 dp (it's an average).
    whole = ["forecast_horizon_units", "expected_demand_protection", "safety_stock",
             "reorder_point", "order_up_to_level", "legacy_safety_stock",
             "on_hand", "suggested_order_qty", "safety_stock_p90"]
    for c in whole:
        if c in plan.columns:
            plan[c] = plan[c].round().astype("Int64")
    plan["safety_stock_delta_vs_legacy"] = (plan["safety_stock"] - plan["legacy_safety_stock"]).astype("Int64")

    if list_price is not None:
        lp = plan[col_product].map(pd.Series(list_price)).fillna(0.0).round().astype("Int64")
        plan["list_price"] = lp
        plan["safety_stock_value_thb"] = (plan["safety_stock"] * lp).astype("Int64")
        if "suggested_order_qty" in plan:
            plan["suggested_order_value_thb"] = (plan["suggested_order_qty"] * lp).astype("Int64")

    # A simple action recommendation for the quarterly Purchasing review.
    def _action(row):
        if "on_hand" in plan.columns:
            if row.get("months_of_cover_on_hand", 0) > 18:
                return "Reduce - overstocked"
            if row["on_hand"] < row["reorder_point"]:
                return "Reorder"
            return "OK - hold"
        if row["forecast_monthly_units"] > 0:
            return "Reorder"
        # A zero forecast is NOT evidence the item is dead -- it is this
        # model's standard failure mode on data that is ~78% zeros. Items that
        # demonstrably sold in the last 12 months get their own label: their
        # safety stock is non-zero precisely because the model has been wrong
        # about them (sigma comes from that error), so calling them "no demand"
        # contradicted their own sales record. Same lesson as the obsolescence
        # `declining` rule -- accuse on actuals, never on the forecast alone.
        if trailing is not None and float(trailing.get(row[col_product], 0) or 0) > 0:
            return "Review - forecast zero, sold recently"
        return "Review - no demand"
    plan["recommended_action"] = plan.apply(_action, axis=1)

    for c in ("forecast_monthly_units", "p50_monthly_units", "p90_monthly_units",
              "months_of_cover_on_hand", "p90_vs_formula_ratio"):
        if c in plan.columns:
            plan[c] = plan[c].round(1)

    # Explicit column order. Several columns are derived late (the delta after
    # rounding, the baht values after that), and pandas appends a new column at
    # the end -- which pushed "safety stock vs current rule" away from the two
    # columns it is the difference of. Anything not listed keeps its place at
    # the end, so a future column can't be silently dropped.
    order = [col_product, "forecast_horizon_units", "forecast_monthly_units",
             "protection_period_months", "expected_demand_protection",
             "safety_stock", "reorder_point", "order_up_to_level",
             "legacy_safety_stock", "safety_stock_delta_vs_legacy",
             "p50_monthly_units", "p90_monthly_units", "safety_stock_p90", "p90_vs_formula_ratio",
             "on_hand", "suggested_order_qty", "months_of_cover_on_hand",
             "list_price", "safety_stock_value_thb", "suggested_order_value_thb",
             "recommended_action"]
    plan = plan[[c for c in order if c in plan.columns] + [c for c in plan.columns if c not in order]]
    return plan


def obsolescence_flags(forecast_df, history_df, col_product, horizon_months=12,
                       production_status=None, list_price=None, on_hand=None,
                       dormant_months=24, decline_ratio=0.4, min_trailing_units=12,
                       stock_cover_months=5.0):
    """Flags SKUs at risk of becoming dead / obsolete stock -- the "inventory
    unsold > 5 years -> write-off" pain point. Deliberately conservative so
    the list stays actionable rather than flagging the whole sparse tail.
    Reasons stack:
      * status_discontinued -- Production Status == 'Discontinue'
      * dormant             -- sold historically, but nothing in the last
                               `dormant_months` months
      * declining           -- had real volume (>= min_trailing_units over the
                               trailing horizon) and its ACTUAL sales fell below
                               decline_ratio x the preceding equal-length window.
                               Measured actual-vs-actual, deliberately NOT against
                               the forecast: on demand this sparse the model
                               outputs zero for most intermittent items, so a
                               forecast-based test flags "sparse" rather than
                               "dying". Items whose first ever sale falls inside
                               the trailing window are skipped -- they have no
                               prior period to have declined from. The forecast
                               enters only as a VETO, never as a trigger: an item
                               the model still expects to outsell its own last 12
                               months is not dead stock. That asymmetry is the
                               point -- a zero forecast (the model's failure mode
                               on sparse data) can no longer manufacture a flag,
                               while a credible rebound can clear one. Never
                               set on a discontinued item.
      * overstocked         -- on-hand covers > 3 years of forecast demand
    Returns only flagged SKUs with reasons and (if list_price / on_hand
    given) the baht value exposed, ranked by value at risk.

    Everything is priced at `stock_cover_months` of cover (lead time + review
    period), because a year of *demand* is not a year of *stock on the shelf*.
    The quantity behind that comes from, in order:
      1. on_hand          -- real stock, if supplied. Used AS-IS: it is already a
                             measured stock level, so scaling it would be wrong.
      2. forecast_next    -- next-horizon forecast, scaled to the cover window
      3. trailing_actual  -- last `horizon_months` of real sales, scaled likewise
    Step 3 exists because the items most at risk -- discontinued ones -- are
    excluded from the forecast by construction, so steps 1 and 2 both give zero
    and they would price up at nothing at all. Steps 2 and 3 share the same
    scaling so the total is not a blend of two different time windows. The
    `value_basis` column records which step produced each figure, so a proxy is
    never mistaken for a measured one."""
    hist = history_df.copy()
    hist["t"] = hist["FiscalYear"] * 12 + hist["FiscalMonth"]
    last_t = hist["t"].max()
    trailing = hist[hist["t"] > last_t - horizon_months].groupby(col_product)["Units"].sum()
    # The equal-length window immediately before the trailing one. This is the
    # comparison period for a genuine decline -- real sales against real sales.
    prior = hist[(hist["t"] > last_t - 2 * horizon_months)
                 & (hist["t"] <= last_t - horizon_months)].groupby(col_product)["Units"].sum()
    # First month the item ever actually sold. Anything starting inside the
    # trailing window is too new to have a trend, and would otherwise be scored
    # against a period in which it did not exist.
    first_sale_t = hist[hist["Units"] > 0].groupby(col_product)["t"].min()
    recent = hist[hist["t"] > last_t - dormant_months].groupby(col_product)["Units"].sum()
    # "mattered once" = sold at least min_trailing_units in its best 12 months
    hist_by_year = hist.groupby([col_product, "FiscalYear"])["Units"].sum()
    peak_year = hist_by_year.groupby(level=0).max()
    fc_next = forecast_df.groupby(col_product)["Final_Forecast"].sum()

    products = sorted(set(fc_next.index) | set(hist[col_product].unique()))
    status = pd.Series(production_status) if production_status is not None else pd.Series(dtype=object)
    lp = pd.Series(list_price) if list_price is not None else pd.Series(dtype=float)
    oh = pd.Series(on_hand) if on_hand is not None else pd.Series(dtype=float)

    rows = []
    for p in products:
        reasons = []
        mattered = float(peak_year.get(p, 0)) >= min_trailing_units
        discontinued = str(status.get(p, "")).strip().lower() == "discontinue"
        if discontinued:
            reasons.append("status_discontinued")
        # A discontinued item is flagged as discontinued only, never also dormant.
        if mattered and not discontinued and float(recent.get(p, 0)) == 0:
            reasons.append(f"dormant_{dormant_months}m")
        t_actual, f_next = float(trailing.get(p, 0)), float(fc_next.get(p, 0))
        p_actual = float(prior.get(p, 0))
        is_new = float(first_sale_t.get(p, last_t + 1)) > last_t - horizon_months
        # A discontinued item's fall in sales is the discontinuation itself, not
        # a second reason -- it is flagged as discontinued only.
        if (not discontinued and t_actual >= min_trailing_units and p_actual > 0 and not is_new
                and t_actual < decline_ratio * p_actual
                and f_next < t_actual):          # forecast vetoes, never triggers
            reasons.append("declining_demand")
        monthly = f_next / max(horizon_months, 1)
        on_hand_p = float(oh.get(p, np.nan))
        if not np.isnan(on_hand_p) and monthly > 0 and on_hand_p / monthly > 36:
            reasons.append("overstocked_>3yr")
        if not reasons:
            continue
        # A year of demand is not a year of stock -- both demand-derived
        # quantities are scaled to the cover you would realistically hold (lead
        # time + review period), so the column totals one consistent window.
        # Real on-hand stock is exempt: it is already a measured stock level.
        _cover = max(float(stock_cover_months), 0.0) / max(horizon_months, 1)
        if not np.isnan(on_hand_p):
            qty, basis = on_hand_p, "On hand"
        elif f_next > 0:
            qty, basis = f_next * _cover, f"{stock_cover_months:g}-mo cover (forecast)"
        else:
            qty, basis = t_actual * _cover, f"{stock_cover_months:g}-mo cover (last 12m)"
        value = qty * float(lp.get(p, 0) or 0)
        rows.append({col_product: p, "reasons": ", ".join(reasons),
                     "trailing_actual_units": round(t_actual, 1),
                     "forecast_next_units": round(f_next, 1),
                     "value_at_risk_thb": round(value, 2),
                     "value_basis": basis})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("value_at_risk_thb", ascending=False).reset_index(drop=True)
    return out


def plan_for_rules(forecast_df, col_product, inputs, rules):
    """The inventory plan and the obsolescence list for one set of stocking
    rules: `rules` = dict(lead_time, review, service, legacy) in months / as a
    fill rate. `inputs` is everything else they need (horizon, sigma, price,
    p50/p90, trailing sales, production status, the full-catalog history),
    stored once by the build. The app calls this again whenever a rule changes,
    so a plan made in the build and one re-made for new rules cannot differ in
    method."""
    price = inputs["price"] or None
    plan = build_inventory_plan(
        forecast_df, col_product, horizon_months=inputs["horizon"],
        lead_time_months=rules["lead_time"], review_period_months=rules["review"],
        service_level=rules["service"], sigma_by_product=inputs["sigma"],
        list_price=price, legacy_safety_months=rules["legacy"],
        monthly_p50_by_product=inputs["p50"], monthly_p90_by_product=inputs["p90"],
        trailing_actual_by_product=inputs["trailing"],
    )
    obs = obsolescence_flags(
        forecast_df, inputs["history"], col_product, horizon_months=inputs["horizon"],
        production_status=inputs["status"] or None, list_price=price,
        stock_cover_months=float(rules["lead_time"]) + float(rules["review"]),
    )
    return plan, obs


# =============================================================================
#  Service-level simulation -- does the policy actually prevent stockouts?
# =============================================================================
# WAPE says how wrong the forecast was. It does not say whether the resulting
# stocking policy would have met demand, which is the question MELFT actually
# asked ("we keep stocking out"). This replays a periodic-review (R, S) policy
# month by month against REAL held-out demand and reports the fill rate, so the
# forecast can be judged in the units the business uses.
#
# Mechanics, spelled out because they are assumptions rather than facts:
#   * Review every `review_period_months`; at a review, order up to S.
#   * Orders arrive `lead_time_months` later (an order placed in month 1 with a
#     2-month lead lands in month 3).
#   * Unmet demand is LOST, not backordered -- for a distributor the customer
#     buys elsewhere, and treating it as a backlog would flatter the fill rate.
#   * Stock starts at S with an empty pipeline. A true steady state would have
#     orders already in transit, so the opening months are slightly pessimistic
#     -- equally so for every policy compared, which is what matters here.
# Lead time and review period are rounded to whole months, since the demand
# series is monthly.


def simulate_policy(actual_monthly, col_product, order_up_to, lead_time_months=2,
                    review_period_months=3, month_col="FiscalMonth"):
    """Replays one (R, S) policy against actual demand.

    actual_monthly: long frame of REAL demand with [col_product, month_col,
                    'Actual'] -- `res_df` from a backtest is exactly this shape.
    order_up_to:    {product -> S}. Products missing from it are given S = 0.

    Returns one row per product: [demand, served, unmet, stockout_months,
    review_cycles, stockout_cycles, avg_on_hand, order_up_to]."""
    lead = max(int(round(lead_time_months)), 0)
    review = max(int(round(review_period_months)), 1)

    d = actual_monthly.pivot_table(index=col_product, columns=month_col,
                                   values="Actual", aggfunc="sum").fillna(0.0)
    months = sorted(d.columns)
    products = d.index.to_numpy()
    demand = d[months].to_numpy(dtype=float)          # (items x months)
    n, T = demand.shape

    S = pd.Series(order_up_to, dtype="float64").reindex(products).fillna(0.0).to_numpy()
    on_hand = S.copy()
    # Arrivals, indexed by the month slot they land in. Sized T + lead so an
    # order placed in the final months has somewhere to go (it just never
    # arrives within the horizon, which is correct).
    pipeline = np.zeros((n, T + lead + 1))
    short = np.zeros((n, T), dtype=bool)              # did month t go short?
    served = np.zeros(n)
    unmet = np.zeros(n)
    on_hand_sum = np.zeros(n)
    cycles = 0

    for t in range(T):
        on_hand += pipeline[:, t]
        if t % review == 0:
            cycles += 1
            # Inventory position = on hand + everything still in transit, so a
            # review doesn't re-order stock that is already on its way.
            position = on_hand + pipeline[:, t + 1:].sum(axis=1)
            pipeline[:, t + lead] += np.maximum(0.0, S - position)
        got = np.minimum(on_hand, demand[:, t])
        missed = demand[:, t] - got
        served += got
        unmet += missed
        short[:, t] = missed > 1e-9
        on_hand -= got
        on_hand_sum += on_hand

    # A review cycle counts as a stockout if ANY month inside it went short --
    # the classic cycle-service-level definition, reported alongside the fill
    # rate because the two answer different questions ("how often did we let
    # someone down" vs "how much of the demand did we actually ship").
    stockout_cycles = sum(short[:, s:s + review].any(axis=1)
                          for s in range(0, T, review))

    return pd.DataFrame({
        col_product: products,
        "demand": demand.sum(axis=1),
        "served": served,
        "unmet": unmet,
        "stockout_months": short.sum(axis=1),
        "review_cycles": cycles,
        "stockout_cycles": stockout_cycles,
        "avg_on_hand": on_hand_sum / max(T, 1),
        "order_up_to": S,
    })


def service_level_comparison(actual_monthly, col_product, policies, lead_time_months=2,
                             review_period_months=3, list_price=None, month_col="FiscalMonth",
                             demand_only=True):
    """Runs `simulate_policy` for each {label: {product -> S}} in `policies`
    and returns (summary, {label: per_item_frame}).

    demand_only: score only products that actually had demand in the replayed
    year. A product with zero demand can never stock out, so including the
    ~80% of the catalog that sold nothing pushes every policy's fill rate
    towards 100% and hides the difference between them -- the same
    whole-catalog-versus-active trap as the p90 agreement stat.

    summary: [Policy, Fill rate, Cycle service level, Items stocked out,
    Items scored, Stockout months, Avg inventory (units), Avg inventory value,
    Demand, Unmet]."""
    lp = pd.Series(list_price, dtype="float64") if list_price is not None else None
    rows, per_item = [], {}
    for label, S in policies.items():
        r = simulate_policy(actual_monthly, col_product, S, lead_time_months,
                            review_period_months, month_col=month_col)
        per_item[label] = r
        scored = r[r["demand"] > 0] if demand_only else r
        dem = float(scored["demand"].sum())
        n_cycles = float(scored["review_cycles"].max()) if len(scored) else 0.0
        total_cycles = n_cycles * len(scored)
        inv_value = np.nan
        if lp is not None and len(scored):
            inv_value = float((scored["avg_on_hand"]
                               * scored[col_product].map(lp).fillna(0.0)).sum())
        rows.append({
            "Policy": label,
            "Fill rate": (float(scored["served"].sum()) / dem) if dem > 0 else np.nan,
            "Cycle service level": (1 - float(scored["stockout_cycles"].sum()) / total_cycles)
                                    if total_cycles else np.nan,
            "Items stocked out": int((scored["unmet"] > 1e-9).sum()),
            "Items scored": int(len(scored)),
            "Stockout months": int(scored["stockout_months"].sum()),
            "Avg inventory (units)": float(scored["avg_on_hand"].sum()),
            "Avg inventory value": inv_value,
            "Demand": dem,
            "Unmet": float(scored["unmet"].sum()),
        })
    return pd.DataFrame(rows), per_item


FAMILY_MODEL = "Forecast-driven safety stock"
FAMILY_FLAT = "Flat rule"


def policy_frontier(actual_monthly, forecast_df, col_product, sigma_by_product,
                    lead_time_months=2, review_period_months=3,
                    service_levels=(0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99),
                    flat_months=(1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12),
                    list_price=None, trailing_actual_by_product=None, horizon_months=12, fallback_cv=0.7):
    """Fill rate against stock budget for both policy families, so they can be
    compared at the SAME budget rather than at whatever budget each happens to
    choose. A higher service level buys a higher fill rate with more stock, and
    so does a flat rule with more months -- the question worth asking is which
    one gets more service per baht.

    One row per policy: sigma-based safety stock at each of `service_levels`, and
    the flat 'N months of average demand' rule at each of `flat_months`, all
    ordering to the same forecast (`forecast_df`: [col_product, Final_Forecast]).
    Order-up-to levels are computed with exactly build_inventory_plan's formula
    and rounding -- order_up_to_level for the sigma policy,
    expected_demand_protection + legacy_safety_stock for the flat rule -- without
    building a full plan per policy (18 policies replay in ~0.5s, about the cost
    of one full plan build). Returns [Policy, Family, Setting] + service_level_comparison's columns."""
    monthly_s = forecast_df.groupby(col_product)["Final_Forecast"].sum() / max(horizon_months, 1)
    products = monthly_s.index
    monthly = monthly_s.to_numpy(dtype=float)
    sigma = _resolve_sigma(products, monthly, sigma_by_product, fallback_cv)
    dormant = _dormant_mask(products, monthly, trailing_actual_by_product)
    protection = float(lead_time_months) + float(review_period_months)

    policies, meta = {}, []
    for sl in sorted(set(service_levels)):
        safety = np.where(dormant, 0.0, _z(sl) * sigma * np.sqrt(max(protection, 0.0)))
        label = f"Forecast-driven, {sl:.0%} target"
        policies[label] = dict(zip(products, np.round(monthly * protection + safety)))
        meta.append((label, FAMILY_MODEL, float(sl)))
    base = np.round(monthly * protection)
    for k in sorted(set(flat_months)):
        label = f"Flat rule, {k:g} months"
        policies[label] = dict(zip(products, base + np.round(monthly * k)))
        meta.append((label, FAMILY_FLAT, float(k)))

    summary, _ = service_level_comparison(actual_monthly, col_product, policies, lead_time_months,
                                          review_period_months, list_price=list_price)
    return pd.DataFrame(meta, columns=["Policy", "Family", "Setting"]).merge(summary, on="Policy")


def fill_rate_at_budget(frontier, budget, family=FAMILY_FLAT, value_col="Avg inventory value"):
    """The fill rate `family` would reach at average stock `budget` (in
    `value_col`: baht by default, or 'Avg inventory (units)' when there are no
    prices), linearly interpolated along its frontier. NaN outside the swept
    range -- never extrapolated."""
    f = frontier[frontier["Family"] == family].dropna(subset=[value_col]).sort_values(value_col)
    if len(f) < 2 or pd.isna(budget) or not (f[value_col].min() <= budget <= f[value_col].max()):
        return np.nan
    return float(np.interp(budget, f[value_col], f["Fill rate"]))
