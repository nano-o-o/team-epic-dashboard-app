"""Plain-assert check: exact duplicate lines and items with no list price are dropped."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import forecast_core as fc

raw = pd.DataFrame({"id": ["a", "a", "a", "b", "c", "d"], "qty": [1, 1, 2, 5, 3, 4]})
master = pd.DataFrame({"ID": ["a", "b", "c"], "List price": [10, None, "x"]})   # d: not in master, kept
out, n = fc.drop_duplicate_lines(raw)
assert n == 1 and len(out) == 5, (n, out)
out, n = fc.drop_unpriced_items(out, "id", master, "ID", "List price")
assert n == 2 and set(out["id"]) == {"a", "d"}, (n, out)
print("ok")
