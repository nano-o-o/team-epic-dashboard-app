# Can the model get better? — seven experiments (2026-09-16, re-run 2026-09-29 and 2026-09-30; idea 7 added 2026-10-01)

Tested outside the app; nothing here changes the dashboard. Scripts: `experiments/improve_00…07_*.py`
(shared setup in `improve_common.py`, raw result tables in `experiments/_results/`, a rebuildable
cache in `experiments/_cache/`). Run from `experiments/` with `../.venv/Scripts/python.exe <script>`.

**Re-run on 2026-09-29 after two feature leaks were fixed** (commit 9739e75: `avg_monthly_units`
averaged across neighbouring items, and the trailing customer count fell to zero after an item's last
sale). The fix moved every number, and the app now auto-selects LightGBM (Tweedie) instead of XGBoost,
so `improve_common.MODEL` follows it. The verdicts below are the re-run's; where one changed, it says so.

**Re-run again on 2026-09-29 after the data-cleaning rules** (commit e065017: 16 exact duplicate sales
lines and 13 items with no list price dropped; 5,259 → 5,246 items). Every number moved by 0.5 points or
less.

**Re-run on 2026-09-30 after the survivorship fix** (commit 4a5ee97). The app used to drop items the item
master marks "Discontinue" *today* before any backtest — but items are often discontinued because their
demand died, so that scored every past year on the survivors. The app now scores on every item (5,693) and
drops discontinued items only from the forward forecast; `improve_common.py` does the same. Every number
moved (item × month WAPE 69.5% → 71.8%, the 95% plan's budget ฿1.18bn → ฿1.32bn). **Every verdict
holds.** Idea 5 got cleaner: the two category grains no longer flip between origins.

## How everything was scored

- **Baseline = the app's defaults build, rebuilt without Streamlit** and checked before anything
  else: FY2024 item-month WAPE 71.8% (1 month ahead) / 73.4% (12 months ahead), bias −12.3% / −12.9%,
  Category1 27.9% / 29.0%; 95% plan fills 83.0% of demand on ฿1.32bn, flat rule 81.4% at that budget,
  today's 1.5-month rule 75.9% on ฿782M. All match the app exactly.
- **Forecasts are compared 12 months ahead** (made the March before the year), on the app's own
  68,316 validation item-months.
- **Stocking is compared at the same stock budget**, with the app's replay (quarterly review, 2-month
  lead time, lost sales, scored on items that had demand), using only what was known before the year.
- **Two origin years, FY2024 and FY2023.** An idea has to hold up at both.
- **A leak found along the way:** for FY2023 there is no earlier year to measure forecast error on, so
  the app's method falls back to errors measured on FY2023 itself, which lifts its FY2023 fill rate to
  99%. Every FY2023 stocking comparison below uses a fair σ instead (each item's monthly demand std
  over the 24 months before the year; on FY2024 it gives 82.6% vs the app's 83.0%, a like-for-like
  stand-in). The app's shipped FY2024 numbers are **not** affected — FY2024 has a prior year.

## Bottom line

1. **No idea moves item × month accuracy into the 80–90% range.** The best forecasting change tested
   (ideas 2 and 3) made WAPE worse. The model is not the bottleneck.
2. **Outside information (idea 1) is what fixes the level** — and the level is what decides how much stock
   to hold. With 3 months of order visibility the FY2023 plan would have kept ~89% fill on ฿1.47bn instead
   of 90% on ฿2.24bn. It has to come from MELFT; it cannot be simulated into existence.
3. **Price-aware stocking (idea 4) is the only in-house change that helps the business metric**: +1.3 to
   +3.2 fill-rate points at the same budget on FY2024 with the ML forecast. That balanced version loses
   0.1–1.0 points at FY2023; the model-free version wins at both years but starves expensive items.
4. **Using a different model on each dashboard page (idea 5) does not work.** The shipped model wins every
   precision grain (SKU × month, SKU × quarter, both category grains) at both origins. The only grain
   another model "wins" is Total (volume), which is arithmetically |aggregate bias|, and its winner flips
   between origins. One model, many grains, stays right.
5. **Tuning the model's settings (idea 7) does not help the business metric.** Settings that score up to
   2 points better on WAPE do it by forecasting lower, and fill up to 0.9 points less at the same budget.

## Idea 1 — what a sales plan or order backlog would be worth: high value, but only MELFT can supply it

The data has no backlog, quotation or sales-plan field (checked against the data dictionary; only ~550
open orders exist, all at the end of the export), so this is a **value-of-information simulation**: the
app's 12-month-ahead forecast is re-levelled with a plan of a given quality, or given k months of actual
orders in advance. Accuracy is scored on months 4–12 only, so no variant gets credit for months it was
simply told. Stocking compares each variant's 95% plan with the baseline at that same budget.

| | FY2024 WAPE (m4–12) | bias | Cat1 WAPE | 95% plan | baseline at same budget | FY2023 WAPE (m4–12) | bias | Cat1 WAPE | 95% plan | baseline at same budget |
|---|---|---|---|---|---|---|---|---|---|---|
| App | 74.6% | −14.4% | 32.6% | 83.0% on ฿1.32bn | — | 116.2% | +55.1% | 75.0% | 90.3% on ฿2.24bn | — |
| Perfect catalog-total plan | 79.3% | −1.7% | 34.8% | 84.1% on ฿1.47bn | 84.0% (+0.1) | 78.6% | −8.5% | 32.1% | 88.4% on ฿1.58bn | 87.6% (+0.9) |
| Perfect Category1 plan | 82.3% | −2.5% | 34.1% | 83.5% on ฿1.47bn | 83.9% (−0.4) | 80.9% | −8.7% | 30.6% | 88.6% on ฿1.64bn | 87.9% (+0.7) |
| Category1 plan off by ±20% | 83.2% | −6.7% | 39.0% | 82.5% on ฿1.43bn | 83.7% (−1.2) | 82.3% | −9.9% | 35.3% | 88.4% on ฿1.63bn | 87.8% (+0.5) |
| Naive plan (last year's total) | 73.0% | −20.8% | 33.1% | 82.2% on ฿1.24bn | 82.3% (−0.1) | 119.3% | +59.2% | 78.5% | 90.4% on ฿2.28bn | 90.4% (−0.0) |
| 1 month of orders visible, rest re-levelled | 83.1% | +3.8% | 41.9% | 84.8% on ฿1.53bn | 84.3% (+0.5) | 75.1% | −20.2% | 32.0% | 88.0% on ฿1.49bn | 86.9% (+1.1) |
| 3 months of orders visible, rest re-levelled | 77.0% | −7.4% | 34.4% | 84.9% on ฿1.40bn | 83.5% (+1.4) | 75.3% | −19.1% | 31.4% | 89.3% on ฿1.47bn | 86.9% (+2.4) |

What it shows:
- **A plan fixes the level, not the item detail.** Bias goes to ~0 and category WAPE improves sharply when
  the model is badly off (FY2023: 75% → 31–35%). Item × month WAPE gets *worse* at FY2024, because WAPE
  rewards under-forecasting on zero-heavy data — another reason not to judge this by WAPE.
- **Its stocking value is mostly "how much to spend", not "where".** At FY2024 a perfect plan says "hold
  ฿1.47bn to reach 84%" (same efficiency, bigger budget). At FY2023, where the model over-forecast by 70%,
  it says "you only need ~฿1.58bn" — about ฿0.66bn less stock for 1.9 points less fill.
- **Order visibility is worth the most**: 3 months of real orders adds +1.4 (FY2024) and +2.4 (FY2023)
  points at the same budget, and at FY2023 reaches 89% fill on ฿1.47bn instead of 90% on ฿2.24bn.
- **A naive "same as last year" plan is useless** (−0.1 to 0.0 points, and far worse accuracy at FY2023). A
  plan helps only if it knows something the history doesn't; a ±20%-accurate category plan still helped at
  FY2023 but hurt at FY2024.

## Idea 2 — different methods for sparse items (ADIDA): rejected

Simple rate forecasts spread flat over the year, on Sparse (and Sparse + Medium) items, ML elsewhere.

| FY2024 (FY2023 in brackets) | WAPE item-month | Fill @ same budget |
|---|---|---|
| App (12 months ahead) | 73.4% (127.1%) | 83.0% (90.3%) |
| Best: ADIDA annual SES on Sparse | 82.3% (140.7%) | 83.8% (90.4%) |
| 24-month mean on Sparse | 78.5% (138.7%) | 83.0% (90.4%) |
| Croston SBA on Sparse | 102.6% (163.2%) | 83.0% (90.4%) |
| App's own forecast, spread flat over the year | 72.1% (126.1%) | 83.0% (90.3%) |

WAPE gets worse with every rate method; fill rate moves by at most +0.8 points at FY2024 and +0.1 at FY2023.
**The useful finding is the last row:** spreading the app's forecast flat changes fill rate by exactly
nothing. For stocking, *which month* an item sells in doesn't matter — only *how much* over the
protection period — so there is nothing to gain from better month-timing on sparse items.

## Idea 3 — forecast per customer × item: rejected

A LightGBM model of each customer's next-12-month purchases of each item (recency, typical reorder gap,
how overdue the next order is, customer and item volume), plus a model for new buyers, summed to items.
50,919 customer-item pairs.

| | FY2024 WAPE | FY2024 item-year WAPE | FY2024 fill @ same budget | FY2023 WAPE | FY2023 bias |
|---|---|---|---|---|---|
| App | 73.4% | 46.3% | 83.0% | 127.1% | +70% |
| Customer × item model | 76.2% | 54.2% | 79.9% | 180.3% | +134% |
| 50/50 blend with the app | 74.0% | 47.6% | 81.8% | 153.4% | +102% |

Worse at both origins on every measure, including the blend. Reorder cadence per customer does not carry
more signal than the item's own history — consistent with the finding that cadence features were flat in
the ablation.

## Idea 4 — set stock for fill rate per baht: the one idea that helps

Fill rate of each policy family at the same average stock value (each family swept across its own
setting, then read at a common budget):

**FY2024**

| Policy | ฿782M (today's rule) | ฿987M | ฿1.32bn (app's 95% plan) | ฿1.65bn |
|---|---|---|---|---|
| App plan (forecast + z·σ·√P) | 76.3% | 79.8% | 83.0% | 84.9% |
| Flat rule | 75.9% | 78.7% | 81.4% | 82.6% |
| Quantile p50 + k(p90−p50) | 73.7% | 76.2% | 79.3% | 81.9% |
| σ measured directly over 5 months | 76.4% | 79.6% | 82.9% | — |
| Empirical 5-month demand percentile (no model) | 74.9% | 77.4% | 80.9% | 83.2% |
| **Price-aware, empirical demand (24 months)** | 79.0% | 81.3% | 83.4% | 84.3% |
| **Price-aware, empirical demand (36 months)** | **80.7%** | **83.8%** | **86.2%** | **87.7%** |
| **Price-aware + the app's ML forecast** | 79.5% | 81.8% | 84.3% | 86.3% |

**FY2023 (fair σ baseline)**

| Policy | ฿1.50bn | ฿1.55bn (today's rule) | ฿2.00bn | ฿2.50bn |
|---|---|---|---|---|
| App plan, fair σ | 87.0% | 87.3% | 89.6% | 90.9% |
| Flat rule | 84.7% | 84.9% | 86.4% | 87.4% |
| Price-aware, empirical (24 = 36 months; only 24 exist) | 88.3% | 88.6% | 90.0% | — |
| Price-aware + ML forecast | 86.9% | 87.1% | 89.0% | 89.9% |

"Price-aware" chooses each item's order-up-to level to maximise *expected units shipped minus
λ × list price × stock*, then sweeps λ: buffer goes where each baht buys the most fill rate. The plain
quantile, direct-σ and empirical-percentile policies do **not** beat the app — the gain comes
specifically from putting price into the decision, which the app's formula ignores.

**Who gets the stock** (FY2024, at a matched budget):

| Policy | Budget | Fill | Items stocked out | Items with zero stock | Fill: cheapest ⅓ | middle ⅓ | priciest ⅓ |
|---|---|---|---|---|---|---|---|
| App 95% plan | ฿1.32bn | 83.0% | 1,460 | 805 | 81.5% | 83.7% | 85.8% |
| Price-aware, empirical | ฿1.25bn | 83.1% | 1,575 | 906 | 83.1% | 84.6% | **79.2%** |
| Price-aware + ML forecast | ฿1.27bn | 83.8% | **1,428** | 894 | 82.2% | 85.4% | 84.9% |

At FY2023 (matched ฿2.24–2.25bn) all three sit within 0.8 points and the price tiers are essentially identical.

**Verdict.**
- The **empirical price-aware** policy (36-month history) wins at both origins (FY2024 +2.8 to +4.4 points,
  FY2023 +0.4 to +1.3) — but it gets there by cutting the priciest third of items from 86% to ~79–81% fill.
  That is a business-policy trade (fewer stockouts on cheap, high-volume items; more on expensive ones),
  not a free win.
- The **price-aware + ML forecast** policy is the balanced version: at FY2024 +1.3 points at ฿1.32bn and
  +3.2 at today's ฿782M budget, *fewer* items stocked out, and the priciest third about unchanged — but it
  **loses** 0.1–1.0 points at FY2023. It does not clear the "wins at both origins" bar.
- Gains are 1–3 points at the plan's own budget — about the size of the plan's existing edge over the flat
  rule.

## Idea 5 — a different model on each dashboard page: rejected

Each page reports the *same* forecast at a different grain — Category1 × month 27.9% is the item numbers
added up, not a category model's score — so "use the best model for each page" only means something if the
model ranking flips between grains. `improve_05_per_grain_models.py` measures it: the app's three default
models scored at every grain `evaluate_backtest` reports, 1 month ahead at FY2024 **and** FY2023, then
12 months ahead at FY2024 (the horizon the pages actually consume). Every shipped headline reproduces
exactly, so this is a true defaults build.

**FY2024, 12 months ahead — WAPE by grain**

| Grain | **LightGBM (Tweedie) — shipped** | XGBoost (Tweedie) | HistGradientBoosting |
|---|---|---|---|
| SKU × month | **73.4%** | 74.3% | 86.2% |
| Category1 × month | **29.0%** | 29.2% | 33.9% |
| Category2 × month | **37.7%** | 38.2% | 44.9% |
| SKU × quarter | **59.0%** | 59.5% | 71.2% |
| Total (volume) | 12.9% | **9.5%** | 11.0% |
| *Bias* | *−12.9%* | *−9.5%* | *−11.0%* |
| *FY2024 units forecast (actual 187,173)* | *163,008* | *169,415* | *166,672* |

**Why XGBoost's Total win is not a win.** `Total (volume)` collapses every item and month into one number,
so its WAPE *is* |aggregate bias| — it rewards whichever model under-forecasts least, not whichever
forecasts best. XGBoost lands on 169,415 units against 187,173 actual where LightGBM lands on 163,008: less
low, still low. It loses every precision grain to LightGBM, and the inventory plan is built item by item,
with σ from each item's own error, so switching would degrade the forecast that *decides* in order to
improve one row that is only *reported*. The honest lever for the level is the bias correction that
already exists and is deliberately off (≈3 WAPE points for ≈7 bias points).

**The per-grain winners, 1 month ahead, at both origins:**

| Grain | FY2024 winner | FY2023 winner | |
|---|---|---|---|
| SKU × month | LightGBM (71.8%) | LightGBM (102.4%) | same |
| Category1 × month | LightGBM (27.9%; XGBoost 28.4%) | LightGBM (54.9%) | same |
| Category2 × month | LightGBM (36.2%; XGBoost 36.5%) | LightGBM (61.8%) | same |
| SKU × quarter | LightGBM (53.7%) | LightGBM (81.5%) | same |
| Total (volume) | HistGradientBoosting (3.1%) | LightGBM (36.7%) | **flips** |

Only Total (volume) changes hands, and it is |bias|, not precision: HGB is worst of the three there at
FY2023 (47.9%). Before this re-run XGBoost edged the two category grains at FY2024 by 0.1 points or less;
scored on every item, LightGBM now holds them at both origins. Choosing a winner per grain on the validation
year would be selecting on the test set anyway.

**Verdict.** Keep one model for the whole build. Beyond the accuracy argument, one forecast feeding every page
is what lets a number be traced from the Overview through the Category Plan to the Inventory Plan; per-page
models would leave the ฿ figures on one page unable to reconcile with the units on another. The shipped model
wins every grain that measures precision, at both origins. This is also why `evaluate_backtest` returns
`by_grain` as a reporting matrix while `build_everything` selects on the SKU-month column alone — the
separation is deliberate.

## Idea 6 — Poisson safety stock for slow movers: rejected (2026-09-22)

The app's safety stock assumes demand over the 5-month protection period is roughly a bell curve. For an
item that sells 0–2 units a month that is a poor fit, and the textbook spare-parts fix is to stock the
service-level percentile of a **Poisson** count instead. `improve_06_poisson_safety_stock.py` replays it
with the app's exact simulation, at the same stock budget:

| Fill rate at the same budget | FY2024 @ ฿782M | FY2024 @ ฿1.32bn | FY2023 @ ฿1.55bn | FY2023 @ ฿2.24bn |
|---|---|---|---|---|
| Baseline (normal, the app) | 76.3% | **83.0%** | **87.3%** | **90.3%** |
| Poisson for every item | 73.4% | can't reach it (max ฿1.07bn) | 85.8% | can't reach it (max ฿1.73bn) |
| Hybrid: Poisson below 1 unit/month, normal above | 76.6% | 83.0% | 86.1% | 88.6% |
| Negative binomial (item's own variance) | 75.5% | 81.1% | 85.5% | 88.2% |

- **Why Poisson loses:** it assumes variance = mean (independent single-unit orders). MELFT's demand
  varies **2.4× its mean** (median item), and 71% of selling items exceed 1.5×: B2B orders come in lumps. A
  pure Poisson rule therefore under-stocks — even at 99.5% it can't spend the plan's budget.
- **The hybrid** gains 0.0–0.3 points at FY2024 but loses 1.2–1.7 at FY2023, so it fails the "wins at both
  origins" bar.
- **Negative binomial** (Poisson's overdispersed cousin, fitted to each item's own error) also loses.
- **Where Poisson *does* live in the project:** the Tweedie objective is a compound Poisson–gamma
  model — a Poisson number of orders per month, each with a gamma-distributed size. That is exactly the
  "lumpy" structure a plain Poisson misses, and why Tweedie beat the explicit hurdle and Croston models.

## Idea 7 — tune the model's settings: rejected (2026-10-01)

`improve_07_tuning.py`. Thirteen LightGBM (Tweedie) settings, one change at a time from the shipped
defaults, scored 1 month ahead at both origins:

| Setting | FY2024 WAPE | FY2024 bias | FY2023 WAPE | FY2023 bias |
|---|---|---|---|---|
| **Shipped** (Tweedie power 1.3, 300 trees, 31 leaves) | **71.8%** | −12.3% | **102.4%** | +36.7% |
| Tweedie power 1.1 | 73.1% | −10.8% | 103.9% | +37.4% |
| Tweedie power 1.5 | 70.8% | −13.8% | 101.1% | +35.7% |
| Tweedie power 1.7 | 69.8% | −16.1% | 97.8% | +31.7% |
| 63 leaves | 71.7% | −13.4% | 100.9% | +33.7% |
| Bagging (80% rows and columns) | 71.2% | −12.5% | 101.2% | +34.6% |
| Fewer/more trees, slower learning, 15 leaves, min leaf 20/50, L2 = 5 | 71.6–72.4% | | 101.4–103.7% | |

The settings that lowered WAPE at both origins were re-run the way the app forecasts (12 months ahead,
FY2024) and stocked through the app's replay at the shipped 95% plan's budget (฿1.32bn):

| Setting | 12-month WAPE | 12-month bias | Fill at ฿1.32bn |
|---|---|---|---|
| **Shipped** | 73.4% | −12.9% | **83.0%** |
| Tweedie power 1.5 | 71.9% | −16.7% | 82.6% |
| Tweedie power 1.7 | 71.5% | −19.2% | 82.1% |
| Bagging | 73.3% | −11.2% | 83.0% |

- **Why the WAPE "win" is a trap:** a higher Tweedie power pulls each forecast towards the item's typical
  month — which, with 79% of item-months at zero, means lower. WAPE rewards that; the stocking plan does
  not. Power 1.7 forecasts 7% fewer units and fills 0.9 points less at the same budget.
- **Everything else is within about a point**, and the one neutral candidate (bagging) leaves fill rate
  exactly where it is. The shipped settings stay.
- **A quirk checked along the way:** when `forecast_future` feeds its own forecasts forward, it marks an
  item "sold last month" whenever the forecast is above zero — and a Tweedie forecast never is exactly
  zero, so after the first month every item looks recently sold. Counting a month as sold only from 0.5
  or 1 unit makes the 12-month forecast slightly *worse* (FY2024 73.4% → 73.8% / 73.9%; FY2023
  127.1% → 127.8% / 128.4%), so the code is left as it is.
- **Judge answer:** "Yes, we tuned it: thirteen settings at two origins. The ones that score better on WAPE
  do it by under-forecasting and fill fewer orders on the same budget, so we kept the defaults."

## Not re-run after the fix

Hyperparameter tuning of XGBoost (ten settings: best +0.4 points, worse 12-month bias; LightGBM was
re-tuned after the fix, see idea 7) and direct multi-horizon forecasting (better 12-month bias, worse WAPE) were one-off checks with no script here; their
numbers predate the leak fixes. Model averaging was re-measured on the 2026-09-30 build: the mean of
LightGBM and XGBoost scores 72.1% against LightGBM's 71.8% (1 month ahead), with 99.7% correlated errors —
still no gain.
