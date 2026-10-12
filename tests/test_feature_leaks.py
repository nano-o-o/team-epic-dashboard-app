"""Plain-assert check that per-item features never read another item or a later month.
Run from the project root: .venv/Scripts/python.exe tests/test_feature_leaks.py"""
import sys, pathlib
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from forecast_core import add_features, compute_distinct_customers_trailing, compute_reorder_cadence_feature

# Two items, shuffled index: item A sells 100/month, item B sells 0.
df = pd.DataFrame({
    "P": ["A"] * 4 + ["B"] * 4,
    "FiscalYear": [2024] * 8,
    "FiscalMonth": [1, 2, 3, 4] * 2,
    "Units": [100.0] * 4 + [0.0] * 4,
}).sample(frac=1, random_state=42)
f = add_features(df, "P", 4).set_index(["P", "FiscalMonth"])
assert pd.isna(f.loc[("B", 1), "avg_monthly_units"])  # no prior months of B: nothing to average
assert (f.loc["B"].loc[2:, "avg_monthly_units"] == 0).all()  # A's 100s never reach B
assert (f.loc["A"].loc[2:, "avg_monthly_units"] == 100).all()
assert f.loc[("A", 4), "roll_mean_3"] == 100 and f.loc[("B", 4), "roll_mean_3"] == 0

# One customer buys in fiscal month 1 only; the count must hold for the next 12 months.
raw = pd.DataFrame({"P": ["A"], "C": ["x"], "D": pd.to_datetime(["2024-04-15"])})
c = compute_distinct_customers_trailing(raw, "P", "C", "D", 4)
c = c.set_index(["FiscalYear", "FiscalMonth"])["distinct_customers_trailing_12m"]
assert c.loc[(2024, 1)] == 0  # current month excluded
assert (c.loc[(2024, 2):(2025, 1)] == 1).all() and len(c.loc[(2024, 2):(2025, 1)]) == 12
# Orders in Apr, May, Jun 2024, then Nov 2027 (3.5 years later). Counting the buyer as
# active beyond horizon_months after June would reveal that they come back.
raw = pd.DataFrame({"P": ["A"] * 4, "C": ["x"] * 4,
                    "D": pd.to_datetime(["2024-04-15", "2024-05-15", "2024-06-15", "2027-11-15"])})
r = compute_reorder_cadence_feature(raw, "P", "C", "D", 4, horizon_months=18)
r_t = r["FiscalYear"] * 12 + r["FiscalMonth"]
assert r_t.max() - (2024 * 12 + 3) <= 18 or r_t[r_t > 2024 * 12 + 3 + 18].min() > 2027 * 12 + 8, sorted(r_t)

print("feature leaks: ok")
