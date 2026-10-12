# Project glossary — words that make us faster

Two lists. **Part A** is the forecasting / inventory vocabulary this project runs on —
use these in the deck and when talking to mentors. **Part B** is how to describe a
code change to me (Claude) so I change the right thing on the first try.

---

## Part A — Forecasting & inventory terms

### The data

| Term | Meaning | In this project |
|---|---|---|
| **SKU** | "Stock-Keeping Unit" — one distinct product you can order | Keyed on `Item Internal ID`, ~5,700 of them |
| **Grain** (or *granularity*) | How finely the data is sliced: per-SKU-per-month is *fine* grain, per-category-per-year is *coarse* grain | Our headline metric is at **SKU × month** grain — the hardest one |
| **Intermittent / lumpy demand** | A series that is mostly zeros with occasional spikes | ~78% of our SKU-month cells are zero |
| **Sparsity** | Share of months an item sold in at all | `sparsity_coverage` feature; ~75% of the catalog is "sparse" (<25%) |
| **Fiscal year (FY)** | A 12-month accounting year not starting in January | MELFT: **April → March**. "FY2024" = Apr 2024–Mar 2025 |
| **Aggregate** | Add up many series into one total | ETS/SARIMA only work on the aggregate, not per-SKU |

### Model types

| Term | Meaning |
|---|---|
| **Baseline** | The dumb-but-honest forecast you must beat. Ours is **seasonal-naive**: "same month last year, times this item's growth rate" |
| **Gradient boosting** (LightGBM, XGBoost, HistGradientBoosting) | A model that builds many small decision trees, each fixing the last one's mistakes. Our main models |
| **Tweedie objective** | A setting that tells the model "expect mostly zeros with occasional positive spikes" — fits B2B demand far better than default regression |
| **Croston / SBA** | The classic intermittent-demand method: forecast *size of an order* and *gap between orders* separately. We tested it; it lost |
| **Hurdle (2-stage)** | One model predicts *will it sell at all?*, a second predicts *how much, given a sale*. Multiply them. Tested; didn't beat plain LightGBM |
| **ETS / Holt-Winters** | "Exponential smoothing" — a weighted average that leans on recent months and repeats a seasonal shape. Aggregate-level cross-check |
| **SARIMA** | A statistical time-series model with seasonal terms. Aggregate-level cross-check; the "why not SARIMA?" answer |
| **Ensemble** | Averaging several models' forecasts together. Tested here: no gain — our models' errors are 97–99.8% correlated |

### Accuracy metrics (all: lower = better, except where noted)

| Term | Meaning | Watch out for |
|---|---|---|
| **WAPE** | Weighted Absolute Percentage Error = total units we were off ÷ total actual units. Our headline | Volume-weighted, so big items dominate |
| **MAPE** | Mean Absolute Percentage Error — averages the % error of each row equally | Explodes when actual is near zero; not great for sparse data |
| **MAE** | Mean Absolute Error — average units off, in plain units | Not comparable between a big item and a small one |
| **RMSE** | Root Mean Squared Error — like MAE, but squaring the misses makes a few big ones weigh more | Always ≥ MAE; a big gap means a few large misses. Same unit problem as MAE |
| **MASE** | MAE ÷ (MAE of the naive forecast). **< 1 means you genuinely beat naive** | The fair yardstick for lumpy demand |
| **Forecast bias** | Are we systematically too high (+) or too low (−)? sum(forecast)/sum(actual) − 1 | Matters more than error *size* for stocking |
| **Pinball loss** | The score for a quantile forecast (p10, p90): misses on the wrong side of the line cost more — a p90 is punished hard for demand above it, lightly for demand below | Lower is better; only comparable between models at the same quantile |
| **Backtest** | Train on old years, predict a year you hid, score against what really happened | Our FY2021–23 → predict FY2024 |
| **1 month ahead / 12 months ahead** | Two ways to score a backtest: predict each month from the *real* month before it, or make the whole year's forecast at the start and build on your own predictions | The app reports both; 12 months ahead is how the forecast is actually used |
| **Hold-out** | The hidden year in a backtest | = "validation year" in the app |
| **Walk-forward / rolling-origin** | Backtest repeatedly, moving the cutoff forward each time (train→22 predict 23; train→23 predict 24) | Shows whether accuracy is *stable*, not a fluke |
| **Leakage** | Accidentally letting the model see the future (e.g. using this month's sales to build this month's feature) | Every feature here is shifted to use only *prior* months |
| **Feature ablation** | Turn feature groups on one at a time and re-score, to see what each is actually worth | Our finding: one feature (trailing distinct-customer count) does most of the work |

### Features

| Term | Meaning |
|---|---|
| **Feature** | An input column the model learns from |
| **Lag feature** | "The value N months ago" (`lag_1`, `lag_12`) |
| **Rolling mean** | Trailing average over the last N months (`roll_mean_3`) |
| **One-hot encoding** | Turning a category ("PLC", "Robot") into yes/no columns, one per value |
| **Seasonality encoding** | Month-of-year expressed as sine/cosine so "December" and "January" are close |
| **Leading indicator** | A feature that moves *before* demand does (e.g. "regular buyers have gone quiet") |
| **Experimental B2B features** | Reorder cadence (which regular buyers are due to order again) and customer mix (VIP/wholesale share, buyer industries). Built from the customer history, **off by default**: in testing they did not improve accuracy |

### Inventory

| Term | Meaning |
|---|---|
| **Lead time** | Weeks/months between placing a supplier order and the stock arriving |
| **Review period** | How often you check stock and decide whether to reorder (MELFT: quarterly) |
| **Safety stock** | Extra buffer stock held to cover demand being higher than forecast during the lead time |
| **Service level / fill rate** | Target probability you *don't* run out (95% = out-of-stock ~1 cycle in 20) |
| **Reorder point (ROP)** | Stock level that triggers a new order |
| **Order-up-to level** | The target stock level you top back up to when you order |
| **Same-budget comparison** | Compare two stocking policies at the *same* average stock value, so "we stock out less" can't just mean "we hold more" |
| **Obsolescence / dead stock** | Inventory that will likely never sell — discontinued, dormant, or sharply declining |
| **S&OP** | "Sales & Operations Planning" — the monthly meeting where demand and supply plans are reconciled, usually at category level |

---

## Part B — Talking to me about code changes

Being specific in these ways saves a round-trip:

| Instead of… | Say… | Why it helps |
|---|---|---|
| "the chart looks wrong" | "on **Product Detail**, the forecast line for revenue mode is ~10× too high" | Names the page + mode + direction of the error |
| "add a filter" | "add a **Category 1 dropdown** to the **Forecast Table** page that narrows the rows" | Names the widget, the page, and what it acts on |
| "it's slow" | "**Build Forecast** takes ~2 min on my real file; the file is 265k rows CSV" | Tells me *which* step and gives me a number to beat |
| "change the wording" | "in the **Model Accuracy** help tooltip, 'WAPE' should read '…'" | Points at the exact string |
| "make it look better" | "match the **corner radius / border** of the chart card to the KPI cards" | Concrete visual property, not a vibe |

**Useful words:**

- **Page / view** — one entry in the left sidebar (Overview, Model Accuracy, …).
- **Widget** — an interactive control: a `selectbox` (dropdown), `radio` (pill buttons), `checkbox`, `slider`, `multiselect`, `text_input`, `button`, `expander` (collapsible section).
- **Card** — a bordered white box. KPI cards, table frames, and now the chart containers.
- **Tooltip / `help=`** — the little "?" you hover for an explanation.
- **Module** — one `.py` file. `dashboard_app.py` = the UI; `forecast_core.py` = the modelling; `inventory.py` = stocking maths; `classical.py` = ETS/SARIMA.
- **Function signature** — a function's name + its arguments, e.g. `build_inventory_plan(forecast_df, col_product, lead_time_months=2.0, …)`.
- **Refactor** — restructure code *without changing what it does* (e.g. "pull this into a helper").
- **Regression** — something that used to work and now doesn't, caused by a recent change.
- **Edge case** — an unusual input that breaks things: an item with zero history, a single-month customer, a category that's all null.
- **Smoke test** — a quick "does every page still load without an error" check. I run these headlessly with `AppTest` after changes.
- **Hard-coded** — a value written directly in the code rather than read from a setting or the data (e.g. `lead_time = 2.0`).
- **Deterministic** — same input always gives the same output (our `random_state=42` everywhere keeps the models deterministic).
- **In place** — modify the existing file/object rather than making a copy.

**When you want options, not action:** say *"assess / recommend / what would you do"* — I'll lay out choices and wait. Say *"do it / go ahead / implement"* when you want the change made.
