"""Plain-assert check for forecast_core.order_rhythm. Run from the project root:
.venv/Scripts/python.exe tests/test_order_rhythm.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from forecast_core import order_rhythm

assert order_rhythm([0, 0, 0]) is None

one = order_rhythm([0, 5, 0])
assert one["gap"] is None and one["next_idx"] is None and one["last_vs_typical"] is None
assert one["last_idx"] == 1 and one["last_size"] == 5

# Sells every 3 months; last sale is the biggest ever.
two = order_rhythm([3, 0, 5, 0])  # one gap is not a rhythm
assert two["gap"] is None and two["next_idx"] is None

r = order_rhythm([4, 0, 0, 4, 0, 0, 4, 0, 0, 20, 0])
assert r["gap"] == 3 and r["last_idx"] == 9 and r["next_idx"] == 12
assert r["typical_size"] == 4 and r["last_vs_typical"] == "big"
assert r["sale_months"] == 4 and r["months"] == 11

assert order_rhythm([10, 10, 10, 1])["last_vs_typical"] == "small"
assert order_rhythm([2, 6, 4])["last_vs_typical"] == "typical"
print("order_rhythm: ok")
