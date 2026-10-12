# Team Epic — MELFT Demand Forecasting & Inventory Planning Dashboard

A Streamlit app that forecasts monthly demand for ~5,700 B2B SKUs with intermittent demand and turns the forecast into an inventory plan (safety stock, reorder quantities, obsolescence flags).

## Run it

Requires **Python 3.12+**.

**Windows:** double-click `run_dashboard.bat`. The first run creates `.venv` and installs the packages.

**Any OS:**

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip
.venv/bin/python -m streamlit run dashboard_app.py
```

The app opens at http://localhost:8501.

## Data

MELFT's data is included in `data/`. To build the forecast on it, go to **Build Forecast** and upload **`data/cleaned_data.csv`** in step 1, then run the build (about 1–2 minutes). The item and customer master files in `data/` load automatically. If you run the build without uploading anything, the app uses generated sample data instead.

| File | Contents |
|---|---|
| `Sale Order History.csv` | Raw sales-order export. |
| `cleaned_data.csv` | The same orders after `data_cleaning.ipynb`; the file the app reads. |
| `Item_master.xlsx` | Product categories, production status and list prices. |
| `Customer_master.xlsx` | Customer types and industries. |
| `features.parquet`, `features_dictionary.csv` | The model's feature table, written by `build_features.py`, and a description of every feature. |

## What each file does

### The app

| File | Role |
|---|---|
| `dashboard_app.py` | Starts the Streamlit dashboard: the sidebar, the shared page header and the page switch. |
| `ui/` | The dashboard itself: one file per page in `ui/pages/`, plus shared settings, styles, chart helpers and the cached build (`ui/pipeline.py`). `ui/__init__.py` lists what each file holds. No modelling logic; it calls the three modules below. |
| `forecast_core.py` | The forecasting engine: fiscal calendar (April–March), monthly aggregation, features (built from past months only, so no leakage), models (LightGBM, XGBoost, hurdle model, baselines), backtests and accuracy metrics (WAPE, MASE, bias). |
| `inventory.py` | Turns the forecast into a stocking plan: safety stock from lead time, review period and each item's own forecast error; obsolescence flags; and a replay of past years to compare fill rate at the same stock budget. |
| `classical.py` | ETS and SARIMA forecasts at total and category level, used as a cross-check for the machine-learning model. |
| `data_cleaning.ipynb` | Cleans the raw sales export (dates, duplicates, unpriced items) into the file the app reads. |

### Supporting scripts

| File | Role |
|---|---|
| `build_features.py` | Writes the full feature table to a file for inspection, using the same functions as the app. |
| `check_app.py` | Runs every page and tab of the app headlessly and can compare its outputs against a saved reference, to catch regressions. |
| `export_results.py` | Re-creates the app's CSV downloads into `results/` without clicking through the app. |
| `verify_exports.py` | Re-derives the numbers in the exported CSVs from first principles and checks they add up. |

These read the files in `data/`.

### `tests/`

Small checks written with plain `assert` statements, each run directly (for example `python tests/test_feature_leaks.py`). They use synthetic data.

| File | What it checks |
|---|---|
| `test_feature_leaks.py` | No feature reads another item's data or a later month. |
| `test_cleaning_drops.py` | Exact duplicate sales lines and items with no list price are dropped. |
| `test_order_rhythm.py` | The order-rhythm (reorder cadence) feature is calculated correctly. |

### `experiments/`

Research run outside the app to test whether the model or the stocking policy could be improved. **Results and verdicts are in [`experiments/IMPROVEMENT_RESULTS.md`](experiments/IMPROVEMENT_RESULTS.md)**; the raw output of each run is in `experiments/_results/`.

| File | Question it answers |
|---|---|
| `improve_common.py` | Shared setup: rebuilds the app's default forecast without the UI. |
| `harness.py` | Shared data loader for the older experiments. |
| `improve_00_baseline.py` | Does the rebuild reproduce the app's numbers exactly? (Run first, so the other comparisons can be trusted.) |
| `improve_01_plan_value.py` | How much would a sales plan or order backlog improve accuracy? |
| `improve_02_sparse_methods.py` | Should rarely-sold items use a simpler rate forecast? |
| `improve_03_customer_item.py` | Is forecasting per customer × item and adding up better? |
| `improve_04_stocking_policies.py` | Which stocking rule gives the best fill rate per baht of stock? |
| `improve_05_per_grain_models.py` | Should each dashboard page use a different model? |
| `improve_06_poisson_safety_stock.py` | Does count-based (Poisson) safety stock work better for slow movers? |
| `improve_07_tuning.py` | Does tuning LightGBM's settings help? |
| `feature_ablation.py` | Does each group of features earn its place? |
| `hierarchical_reconciliation.py` | Does forecasting categories first and splitting down beat forecasting each item? |
| `new_features_and_inventory.py` | Early test of the customer-mix and price features and the inventory layer. |

### Other

| File | Role |
|---|---|
| `GLOSSARY.md` | Vocabulary used in the app. |
| `assets/` | Logo and navigation icons. |
| `.streamlit/config.toml` | Theme (colours, font). |
| `run_dashboard.bat` | One-click launcher for Windows. |
