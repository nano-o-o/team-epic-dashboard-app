"""One-command check of the dashboard. Run from the project root:

    .venv/Scripts/python.exe check_app.py                      # demo + real data, every page and tab
    .venv/Scripts/python.exe check_app.py --save ref.pkl        # ...and keep the outputs as a reference
    .venv/Scripts/python.exe check_app.py --compare ref.pkl     # ...and fail if any number/table/caption moved

--compare lists EVERY difference it finds (with a one-line summary of which pages moved), and
ignores the two strings that differ between two runs of the same code: the build clock and the
global <style> block.
    .venv/Scripts/python.exe check_app.py --demo-only           # skip the slow real-data build

Steps: (1) headless AppTest of every page and Model Accuracy tab on demo and on real data
(`cleaned_data.csv`), failing on any st.error / exception or a missing build; (2) the saved-build
restore: a fresh session must get the previous build back without rebuilding; (3) verify_exports.py
on results/. It replaces the ad-hoc snapshot scripts used before.
"""
import argparse
import os
import pathlib
import pickle
import re
import subprocess
import sys
import tempfile

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = pathlib.Path(__file__).resolve().parent
PAGES = ["Overview", "Category Plan", "Product Detail", "Customer Detail", "Forecast Table", "Inventory Plan"]
TABS = ["Business impact", "Model accuracy", "Diagnostics"]
REAL_LOADER = '''def generate_fake_raw_data():
    _r = pd.read_csv("data/cleaned_data.csv", low_memory=False)
    _r.columns = _r.columns.str.strip()
    return clean_dates(_r, "Sales Order Date")[0]

'''
ANCHOR = 'if "_redirect_to" in st.session_state:'


def app_copy(mode, tmp):
    """The app itself for demo; for real, a copy whose demo loader reads cleaned_data.csv."""
    src = (ROOT / "dashboard_app.py").read_text(encoding="utf-8")
    if mode == "real":
        assert src.count(ANCHOR) == 1, "anchor for the real-data loader not found"
        src = src.replace(ANCHOR, REAL_LOADER + ANCHOR)
    path = pathlib.Path(tmp) / f"_check_{mode}.py"
    path.write_text(src, encoding="utf-8")
    return str(path)


def assert_clean(at, where):
    assert not at.exception, (where, [e.value for e in at.exception])
    assert not at.error, (where, [e.value for e in at.error])


def run_pages(path):
    """Build, then visit every page and tab. Returns the outputs to compare."""
    st.cache_data.clear()
    st.cache_resource.clear()
    at = AppTest.from_file(path, default_timeout=1500)
    at.run()
    assert_clean(at, "Build Forecast")
    res = at.session_state["results"]
    assert res is not None, "build produced no results"
    pages = {}
    visits = [(p, None) for p in PAGES] + [("Model Accuracy", t) for t in TABS]
    for page, tab in visits:
        at.sidebar.radio(key="nav_page").set_value(page)
        if tab:
            at.session_state["acc_tab"] = tab
        at.run()
        name = page + (f" / {tab}" if tab else "")
        assert_clean(at, name)
        pages[name] = ([d.value for d in at.dataframe], [c.value for c in at.caption],
                       [m.value for m in at.markdown],
                       # Vega-Lite specs; their dataset names are hashes of the data drawn.
                       [c.proto.spec for c in at.get("vega_lite_chart")])
    check_live_rules(at, res)
    check_category_scope(at, res)
    check_open_top(at)
    return {"res": res[:-1], "meta": dict(res[-1]), "pages": pages}


def check_category_scope(at, res):
    """Category 1 in the Scope popover narrows the product pages to that
    category, narrows Category 2's options, and survives a whole-catalog page."""
    history = res[0]
    cats = history["Category1"].dropna().astype(str).value_counts() if "Category1" in history else []
    if len(cats) < 2:   # demo data has no categories
        return
    at.sidebar.radio(key="nav_page").set_value("Forecast Table")
    at.run()
    n_all = len(res[1]["Item Internal ID"].unique())
    cat = cats.index[1]   # not the biggest
    at.multiselect(key="scope_Category1").set_value([cat]).run()
    assert_clean(at, "Scope / Category 1")
    # history keeps discontinued items (backtests score them); Scope offers forecast items only.
    _fc_items = set(res[1]["Item Internal ID"])
    want = history.loc[(history["Category1"].astype(str) == cat) & history["Item Internal ID"].isin(_fc_items),
                       "Item Internal ID"].nunique()
    text = " ".join(str(m.value) for m in at.sidebar.markdown)
    assert f"{want:,} of {n_all:,} selected" in text, (want, text)
    c2 = set(history.loc[history["Category1"].astype(str) == cat, "Category2"].fillna("").astype(str))
    assert set(at.multiselect(key="scope_Category2").options) <= c2 | {"(unknown)"}
    at.sidebar.radio(key="nav_page").set_value("Category Plan")
    at.run()
    at.sidebar.radio(key="nav_page").set_value("Overview")
    at.run()
    assert at.multiselect(key="scope_Category1").value == [cat], "category Scope lost on navigating away"
    at.multiselect(key="scope_Category1").set_value([]).run()
    assert_clean(at, "Scope / cleared")


def check_open_top(at):
    """Overview's #1 product / customer buttons open that item on its detail page,
    and it is row 1 of that page's top 10."""
    for i, (page, focus) in enumerate([("Product Detail", "pd_focus"), ("Customer Detail", "cd_focus")]):
        at.sidebar.radio(key="nav_page").set_value("Overview")
        at.run()
        btn = [b for b in at.button if b.key == f"_ov_top_{i}"]
        if not btn:   # demo data: no forecast / no customers
            continue
        btn[0].click().run()
        assert_clean(at, f"Overview -> {page}")
        assert at.sidebar.radio(key="nav_page").value == page, page
        assert at.selectbox(key=focus).value == at.session_state[focus + "_clicked"], page


def check_live_rules(at, res):
    """Stocking rules set on Inventory Plan re-plan every page without a
    rebuild, survive navigating away, and the defaults give back the built plan."""
    def shown():
        return [m.value for m in at.markdown] + [c.value for c in at.caption]

    at.sidebar.radio(key="nav_page").set_value("Inventory Plan")
    at.run()
    base = shown()
    at.slider(key="inv_sl_pct").set_value(90).run()
    at.number_input(key="inv_lt").set_value(3.0).run()
    assert_clean(at, "Inventory Plan / new rules")
    moved = shown()
    if res[1]["Final_Forecast"].sum() > 0:   # demo forecasts ~0, so nothing to move
        assert moved != base, "Inventory Plan did not follow the new rules"
    at.sidebar.radio(key="nav_page").set_value("Model Accuracy")
    at.session_state["acc_tab"] = "Business impact"
    at.run()
    assert_clean(at, "Model Accuracy / new rules")
    at.sidebar.radio(key="nav_page").set_value("Inventory Plan")
    at.run()
    assert shown() == moved, "stocking rules lost on navigating away"
    at.slider(key="inv_sl_pct").set_value(95).run()
    at.number_input(key="inv_lt").set_value(2.0).run()
    assert shown() == base, "default rules do not give back the built plan"


def check_restore(path, tmp):
    """A new session (a browser reload) must get the last build back, on the page it was on."""
    os.environ["MELFT_LAST_BUILD"] = str(pathlib.Path(tmp) / "last_build.pkl")
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
        first = AppTest.from_file(path, default_timeout=1500)
        first.run()
        assert_clean(first, "restore: first build")
        # A slow page result must travel with the build. Open its tab before clicking:
        # a click on a tab that isn't drawn is lost.
        first.sidebar.radio(key="nav_page").set_value("Model Accuracy")
        first.session_state["acc_tab"] = "Model accuracy"
        first.run()
        next(b for b in first.button if "walk-forward" in b.label.lower()).click().run()
        assert_clean(first, "restore: walk-forward")
        wf_key = f"_wf_result_v{first.session_state['build_version']}"
        assert wf_key in first.session_state, "walk-forward produced no result"
        # Stocking rules set on Inventory Plan must travel with the build too.
        first.sidebar.radio(key="nav_page").set_value("Inventory Plan")
        first.run()
        first.slider(key="inv_sl_pct").set_value(90).run()
        # A fresh AppTest is a fresh session. Clearing cache_resource as well forces
        # the disk copy to be read, which is what a server restart does.
        st.cache_data.clear()
        st.cache_resource.clear()
        again = AppTest.from_file(path, default_timeout=60)
        again.query_params["page"] = "Inventory Plan"
        again.run()
        assert_clean(again, "restore: new session")
        assert again.session_state["results"] is not None, "no results in the new session"
        assert "_restored_at" in again.session_state, "the new session rebuilt instead of restoring"
        assert again.sidebar.radio(key="nav_page").value == "Inventory Plan", "page was not restored"
        assert wf_key in again.session_state, "walk-forward result was not restored"
        assert again.slider(key="inv_sl_pct").value == 90, "stocking rules were not restored"
    finally:
        os.environ["MELFT_LAST_BUILD"] = ""


_CLOCK = re.compile(r"built \d{1,2} \w{3} \d{2}:\d{2}")


def _norm(x):
    """Two runs of the SAME code differ in two strings that mean nothing: the
    build clock in the sidebar card ("built 20 Sep 13:07"), and the global
    <style> block, which is rebuilt on every edit to the app. Before this,
    every --compare aborted on the clock -- on the first page it reached, so
    the other eight were never checked at all. CSS is not covered here either
    way: broken selectors fail silently and only the browser catches them.
    Only the <style> element collapses, so a page title that carries one
    (FC_SCALE_CSS) still has its text compared."""
    if isinstance(x, str):
        x = re.sub(r"<style.*?</style>", "<style>", x, flags=re.S).strip()
        return _CLOCK.sub("built <clock>", x)
    return x


def _page_of(where):
    """'real['pages']['Inventory Plan'][1][3]' -> 'real Inventory Plan', for the
    one-line summary of which pages moved."""
    m = re.match(r"(\w+)\['pages'\]\['([^']+)'\]", where)
    return f"{m.group(1)} {m.group(2)}" if m else f"{where.split('[')[0]} (build outputs)"


def _brief(v):
    v = repr(v)
    return v if len(v) <= 200 else v[:200] + "..."


def same(a, b, where, diffs):
    """Collect every difference into `diffs` instead of raising on the first,
    so one run names all of them."""
    a, b = _norm(a), _norm(b)
    if type(a) is not type(b):
        diffs.append((where, f"type {type(a).__name__} -> {type(b).__name__}"))
    elif isinstance(a, (pd.DataFrame, pd.Series)):
        eq = pd.testing.assert_frame_equal if isinstance(a, pd.DataFrame) else pd.testing.assert_series_equal
        try:
            eq(a, b, check_exact=True, obj=where)
        except AssertionError as e:
            # pandas' first line is only "DataFrame are different"; the useful
            # part ("shape mismatch", the column, the first differing values)
            # is the lines under it.
            lines = [ln.strip() for ln in str(e).splitlines() if ln.strip()]
            diffs.append((where, "  |  ".join(lines[1:4]) or lines[0]))
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            diffs.append((where, f"keys differ: {sorted(set(a) ^ set(b))}"))
        for k in a.keys() & b.keys():
            same(a[k], b[k], f"{where}[{k!r}]", diffs)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            diffs.append((where, f"length {len(a)} -> {len(b)}"))
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{where}[{i}]", diffs)
    elif type(a).__module__.split(".")[0] in ("sklearn", "lightgbm", "xgboost", "statsmodels"):
        pass  # fitted models: their outputs are compared instead
    elif not (a == b or (a != a and b != b)):
        diffs.append((where, f"{_brief(a)}  ->  {_brief(b)}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save")
    ap.add_argument("--compare")
    ap.add_argument("--demo-only", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    os.environ["MELFT_LAST_BUILD"] = ""  # page checks always build; restore is tested on its own
    modes = ["demo"] if args.demo_only else ["demo", "real"]
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for mode in modes:
            out[mode] = run_pages(app_copy(mode, tmp))
            print(f"[ok] {mode}: {len(out[mode]['pages'])} pages/tabs, forecast total "
                  f"{out[mode]['res'][1]['Final_Forecast'].sum():,.0f}")
        check_restore(app_copy("demo", tmp), tmp)
        print("[ok] saved build restored in a new session")
    if args.save:
        pickle.dump(out, open(args.save, "wb"))
        print(f"[ok] reference saved to {args.save}")
    if args.compare:
        ref = pickle.load(open(args.compare, "rb"))
        diffs = []
        for mode in modes:
            same(ref[mode], out[mode], mode, diffs)
        if diffs:
            # Which pages moved first, then the detail: on a deliberate change
            # the summary alone usually answers "did anything else move?".
            print(f"\n[FAIL] {len(diffs)} difference(s) vs {args.compare}")
            print("       changed: " + ", ".join(sorted({_page_of(w) for w, _ in diffs})))
            for where, what in diffs[:40]:
                print(f"  {where}\n      {what}")
            if len(diffs) > 40:
                print(f"  ... and {len(diffs) - 40} more")
            sys.exit(1)
        print(f"[ok] identical to {args.compare}")
    subprocess.run([sys.executable, "verify_exports.py"], check=True)


if __name__ == "__main__":
    main()
