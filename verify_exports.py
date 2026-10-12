"""
=============================================================================
 verify_exports.py -- invariant checks on the CSVs the dashboard exports
=============================================================================
Run this after downloading files from the app. It re-derives every number it
can from the file itself, from `Item_master.xlsx`, and from `cleaned_data.csv`,
and reports anything that doesn't add up.

Why this exists: the dashboard formats numbers with pandas Stylers, which only
affect what is drawn on screen -- `to_csv()` ships the underlying values. Every
bug this script checks for was a real one found by reading an exported file:
a baht value priced off an unrounded quantity, a projection exported at full
float precision, a "no demand" label on an item that had just sold. Screen
review cannot catch those, because on screen they look fine.

Usage:
    .venv/Scripts/python.exe verify_exports.py                  # checks ./results
    .venv/Scripts/python.exe verify_exports.py --dir some/dir
    .venv/Scripts/python.exe verify_exports.py --lead 2 --review 3

Exit code is 0 when every check passes, 1 otherwise, so it can gate a commit.
Files that aren't present are skipped, not failed.
=============================================================================
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

# Tolerances. Quantities are exported as integers so they must match exactly;
# baht values carry 2 dp in the obsolescence export, hence the 1-unit slack.
QTY_TOL = 0.5
THB_TOL = 1.0

_checks = []


def check(name, ok, detail=""):
    _checks.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def info(msg):
    print(f"         {msg}")


def _load_master(root):
    p = os.path.join(root, "data", "Item_master.xlsx")
    if not os.path.exists(p):
        return None
    im = pd.read_excel(p)
    im.columns = [c.strip() if isinstance(c, str) else c for c in im.columns]
    im["Item Internal ID"] = im["Item Internal ID"].astype(str)
    return im.drop_duplicates("Item Internal ID").set_index("Item Internal ID")


def verify_forecast_pivot(path, master):
    print(f"\n=== {os.path.basename(path)} ===")
    d = pd.read_csv(path)
    months = [m for m in ["Apr", "May", "Jun", "Jul", "Aug", "Sep",
                          "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"] if m in d.columns]
    total_col = next((c for c in d.columns if c.endswith(("total", "total (units)"))), None)  # older exports lack "(units)"
    info(f"{len(d):,} products x {len(months)} months")

    check("12 fiscal months present", len(months) == 12, f"found {len(months)}")
    if total_col:
        diff = (d[total_col] - d[months].sum(axis=1)).abs()
        check("annual total == sum of the 12 months",
              diff.max() <= QTY_TOL, f"max diff {diff.max():.2f}")
        info(f"grand total {d[total_col].sum():,.0f} units")
    check("no negative forecasts", (d[months] < 0).sum().sum() == 0)
    check("no missing values", d.isna().sum().sum() == 0)
    check("item IDs unique", not d.iloc[:, 0].duplicated().any())
    check("all monthly values are whole units", (d[months] % 1 != 0).sum().sum() == 0)

    if master is not None and "Item Code" in d.columns and "Item Code" in master.columns:
        want = d.iloc[:, 0].astype(str).map(master["Item Code"])
        mism = want.notna() & (want.astype(str).str.strip() != d["Item Code"].astype(str).str.strip())
        check("item codes agree with Item_master.xlsx", mism.sum() == 0, f"{mism.sum()} disagree")

    zero = (d[total_col] == 0).mean() if total_col else np.nan
    info(f"{zero:.1%} of products forecast zero for the whole year "
         f"(expected to be high -- ~78% of SKU-months are zero in the source data)")
    return d


def verify_inventory(path, master, lead, review):
    print(f"\n=== {os.path.basename(path)} ===")
    d = pd.read_csv(path)
    info(f"{len(d):,} rows")
    ss, price = "Safety stock", "List price (฿)"
    val = "Safety stock value (฿)"
    mo, yr = "Forecast / month", "Forecast / year"

    if ss in d and price in d and val in d:
        exp = (d[ss] * d[price]).round()
        bad = (d[val] - exp).abs() > THB_TOL
        check("safety-stock value == safety stock x list price",
              bad.sum() == 0,
              f"{bad.sum():,} rows disagree; total {d[val].sum():,.0f} vs {exp.sum():,.0f}")
        z = (d[ss] == 0) & (d[val] > 0)
        check("no baht value against a zero quantity", z.sum() == 0,
              f"{z.sum():,} rows price a zero safety stock at {d.loc[z, val].sum():,.0f} THB")
        info(f"total safety-stock value {d[val].sum():,.0f} THB")

    if mo in d and yr in d:
        off = (d[yr] - (d[mo] * 12).round()).abs()
        # 'Forecast / month' is exported at 1 dp, so 12x it can drift by up to 6.
        check("annual forecast consistent with the monthly average",
              off.max() <= 6, f"max diff {off.max():.0f}")

    if {mo, ss, "Reorder point", "Order-up-to level"} <= set(d.columns):
        rp = (d[mo] * lead + d[ss]).round()
        oo = (d[mo] * (lead + review) + d[ss]).round()
        check(f"reorder point == monthly x lead({lead}) + safety stock",
              (d["Reorder point"] - rp).abs().le(1).all(),
              f"max diff {(d['Reorder point'] - rp).abs().max():.0f}")
        check(f"order-up-to == monthly x (lead+review)({lead + review}) + safety stock",
              (d["Order-up-to level"] - oo).abs().le(1).all(),
              f"max diff {(d['Order-up-to level'] - oo).abs().max():.0f}")

    if {ss, "Current-rule safety stock", "Safety stock vs current rule"} <= set(d.columns):
        delta = d[ss] - d["Current-rule safety stock"]
        check("delta vs current rule == safety stock - current-rule stock",
              (d["Safety stock vs current rule"] - delta).abs().le(QTY_TOL).all())

    if "Action" in d:
        info("action mix: " + ", ".join(f"{k} {v:,}" for k, v in d["Action"].value_counts().items()))
        if ss in d:
            # "no demand" must mean no demand: no forecast AND nothing sold
            # recently. An item carrying stock under that label contradicts
            # itself, which is what the old forecast-only rule produced.
            contra = (d["Action"] == "Review - no demand") & (d[ss] > 0)
            check("'no demand' items carry no safety stock",
                  contra.sum() == 0,
                  f"{contra.sum():,} rows hold {d.loc[contra, ss].sum():,.0f} units "
                  f"({d.loc[contra, val].sum():,.0f} THB) under a 'no demand' label"
                  if contra.sum() else "")
    return d


def verify_obsolescence(path, master, cover_months):
    print(f"\n=== {os.path.basename(path)} ===")
    d = pd.read_csv(path)
    info(f"{len(d):,} flagged items")
    info("reasons: " + ", ".join(f"{k} {v:,}" for k, v in d["Flagged for"].value_counts().items()))
    val, basis = "Value at risk (฿)", "Valued from"
    trail, fnext = "Units sold (last 12m)", "Forecast next 12m"

    if master is None or "List price" not in master.columns:
        info("Item_master.xlsx not available -- skipping the value re-derivation")
    elif val in d and basis in d:
        lp = d.iloc[:, 0].astype(str).map(master["List price"])
        scale = cover_months / 12.0
        for label, src in (("last 12m", trail), ("forecast", fnext)):
            m = d[basis].str.contains(label, na=False)
            if not m.any() or src not in d:
                continue
            exp = d.loc[m, src] * lp[m] * scale
            bad = (d.loc[m, val] - exp).abs() > THB_TOL
            check(f"'{label}' basis: value == units x list price x {cover_months}/12",
                  bad.sum() == 0, f"{m.sum():,} rows on this basis, {bad.sum():,} disagree")
        info(f"total value at risk {d[val].sum():,.0f} THB "
             f"({(d[val] == 0).sum():,} rows valued at zero -- dormant items with no trailing sales)")

    dec = d[d["Flagged for"].str.contains("Declining", na=False)]
    if len(dec) and {trail, fnext} <= set(d.columns):
        # The forecast may only VETO a declining flag, never raise one -- so no
        # flagged item may have a forecast at or above its own last-12m sales.
        viol = dec[fnext] >= dec[trail]
        check("declining flags respect the forecast veto (forecast < last-12m sales)",
              viol.sum() == 0, f"{viol.sum()} of {len(dec)} declining rows violate it")
        check("declining flags are backed by real recent volume",
              (dec[trail] > 0).all(), f"{(dec[trail] <= 0).sum()} rows flagged with no trailing sales")
    return d


def verify_projection(path):
    print(f"\n=== {os.path.basename(path)} ===")
    d = pd.read_csv(path)
    info(f"{len(d):,} categories")
    num = [c for c in d.columns if d[c].dtype.kind in "if"]
    frac = {c: int((d[c] % 1 != 0).sum()) for c in num}
    offenders = {c: n for c, n in frac.items() if n}
    check("unit columns exported as whole numbers", not offenders,
          f"fractional values in {offenders}" if offenders else "")

    trend = next((c for c in d.columns if "Trend" in c), None)
    last = next((c for c in d.columns if c.startswith("FY") and "units" in c), None)
    proj = next((c for c in d.columns if "Projected" in c), None)
    if trend and last and proj:
        mult = 1 + d[trend].astype(str).str.rstrip("%").astype(float) / 100
        # Two sources of legitimate slack: the trend is displayed rounded to a
        # whole percent, and the projection is exported rounded to whole units.
        # The second dominates on tiny rows -- 1 unit x 0.5 = 0.5 rounds to 0, a
        # "50% error" that is really rounding -- so allow the larger of 1% and
        # one unit, in absolute units.
        abs_diff = (d[proj] - d[last] * mult).abs()
        allowed = np.maximum(0.01 * d[last].abs(), 1.0)
        bad = abs_diff > allowed
        check("projection == last-year units x (1 + trend)",
              not bad.any(),
              f"{int(bad.sum())} rows off by more than max(1%, 1 unit); max diff {abs_diff.max():.1f} units")
        info(f"{d[last].sum():,.0f} units last year -> {d[proj].sum():,.0f} projected "
             f"({d[proj].sum() / max(d[last].sum(), 1) - 1:+.1%})")
        floor = (d[trend].astype(str).str.rstrip("%").astype(float) <= -50).sum()
        if floor:
            info(f"{floor} categor{'y' if floor == 1 else 'ies'} pinned at the -50% clip floor "
                 f"(the multiplier is clipped to [0.5, 2.0])")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results", help="directory holding the downloaded CSVs (default: results)")
    ap.add_argument("--root", default=".", help="project root, for Item_master.xlsx (default: .)")
    ap.add_argument("--lead", type=float, default=2.0, help="lead time in months used at build time")
    ap.add_argument("--review", type=float, default=3.0, help="review period in months used at build time")
    args = ap.parse_args()

    d = args.dir
    if not os.path.isdir(d):
        print(f"No such directory: {d}")
        return 1
    master = _load_master(args.root)
    print(f"Checking exports in {os.path.abspath(d)}")
    print(f"Assuming lead time {args.lead} mo + review {args.review} mo "
          f"(pass --lead/--review if you built with different values)")
    if master is None:
        print("Item_master.xlsx not found -- value re-derivations will be skipped.")

    found = False
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        n = os.path.basename(f).lower()
        try:
            if "forecast" in n and "pivot" in n:
                verify_forecast_pivot(f, master); found = True
            elif "inventory" in n:
                verify_inventory(f, master, args.lead, args.review); found = True
            elif "obsolescence" in n:
                verify_obsolescence(f, master, args.lead + args.review); found = True
            elif "projection" in n:
                verify_projection(f); found = True
            else:
                print(f"\n=== {os.path.basename(f)} ===")
                info("no checks defined for this file name -- skipped")
        except Exception as e:
            check(f"{os.path.basename(f)} readable", False, f"{type(e).__name__}: {e}")

    if not found:
        print("\nNo recognised export files found. Expected names containing "
              "'forecast_pivot', 'inventory', 'obsolescence' or 'projection'.")
        return 1

    failed = [n for n, ok, _ in _checks if not ok]
    print(f"\n{'=' * 70}")
    print(f"{len(_checks) - len(failed)} of {len(_checks)} checks passed.")
    if failed:
        print("Failed:")
        for n in failed:
            print(f"  - {n}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
