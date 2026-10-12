"""Re-export the app's CSVs into results/ from a real-data build, for verify_exports.py.

Runs check_app's real-data copy of the app headlessly and writes the payload of every
CSV download button drawn on Forecast Table, Inventory Plan and Customer Detail into
results/ under the button's own file name -- byte-for-byte what a click would download.

    .venv/Scripts/python.exe export_results.py
"""
import os
import pathlib
import tempfile

os.environ["MELFT_LAST_BUILD"] = ""   # always build; never read or overwrite the saved build

import streamlit as st
from streamlit.testing.v1 import AppTest

import check_app

PAGES = ["Forecast Table", "Inventory Plan", "Customer Detail"]
OUT = check_app.ROOT / "results"
HOOK = '''
import streamlit as _st, pathlib as _pl
from streamlit.delta_generator import DeltaGenerator as _DG
_orig_dl = _DG.download_button
def _dl(self, label, data, file_name=None, *a, **k):
    if file_name and str(file_name).endswith(".csv") and isinstance(data, (str, bytes)):
        _pl.Path(r"%s", str(file_name)).write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
    return _orig_dl(self, label, data, file_name, *a, **k)
_DG.download_button = _dl
_st.download_button = _st._main.download_button
'''


def main():
    OUT.mkdir(exist_ok=True)
    path = pathlib.Path(check_app.app_copy("real", tempfile.mkdtemp()))
    src = path.read_text(encoding="utf-8")
    path.write_text(src.replace(check_app.ANCHOR, HOOK % OUT + check_app.ANCHOR, 1), encoding="utf-8")

    at = AppTest.from_file(str(path), default_timeout=1500)
    at.run()
    check_app.assert_clean(at, "Build Forecast")
    assert at.session_state["results"] is not None, "build produced no results"
    for page in PAGES:
        at.sidebar.radio(key="nav_page").set_value(page)
        at.run()
        check_app.assert_clean(at, page)
    print("results/:", ", ".join(sorted(p.name for p in OUT.glob("*.csv"))))


if __name__ == "__main__":
    main()
