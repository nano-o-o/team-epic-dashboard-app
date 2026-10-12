"""Keeps the last build across browser reloads and server restarts (memory + a pickle
file), so a reload never forces a rebuild.
"""
import functools
import glob
import hashlib
import os
import pickle
import sys
import time

import streamlit as st


# ============================= SAVED BUILD ========================================
# A browser reload starts a new Streamlit session, which used to wipe `results` and
# cost a 90-second rebuild mid-demo. The last build is kept in memory and in a file,
# so a reload -- or a server restart -- gets it straight back, on the same page.
# MELFT_LAST_BUILD moves the file; setting it to "" switches all of this off.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _last_build_path():
    """Read on every call, not at import: check_app.py changes MELFT_LAST_BUILD between runs."""
    return os.environ.get("MELFT_LAST_BUILD", os.path.join(".streamlit", "last_build.pkl"))


@functools.cache  # an edit re-imports this module (Streamlit's file watcher), so it is recomputed then
def _code_fingerprint():
    """Changes whenever the app or pipeline code does, so an edit never serves a stale build."""
    h = hashlib.sha256()
    app_files = [os.path.join(_ROOT, "dashboard_app.py")] + sorted(
        glob.glob(os.path.join(_ROOT, "ui", "**", "*.py"), recursive=True))
    for path in app_files + [sys.modules[m].__file__ for m in ("forecast_core", "inventory", "classical")]:
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


@st.cache_resource(show_spinner=False)
def _saved_build():
    """The last build, shared by every session of this server ({} if none)."""
    # ponytail: one slot for the whole server -- right for one presenter on a local app;
    # key it per user before hosting the app for several people at once.
    try:
        with open(_last_build_path(), "rb") as f:
            saved = pickle.load(f)  # only ever written by _write_saved_build below
        if saved.get("code") == _code_fingerprint():
            return saved
    except Exception:
        pass  # no file, unreadable, or from older code: the next build replaces it
    return {}


def _write_saved_build():
    try:
        path = _last_build_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path + ".tmp", "wb") as f:
            pickle.dump(_saved_build(), f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(path + ".tmp", path)
    except Exception as e:
        st.toast(f"Couldn't save the build for page reloads: {e}")


def remember_build():
    """After a successful build: this session's build becomes the saved one."""
    st.session_state.pop("_restored_at", None)
    if not _last_build_path():
        return
    saved = _saved_build()
    saved.clear()
    saved.update(
        code=_code_fingerprint(), token=time.time_ns(), built=time.strftime("%d %b %H:%M"),
        state={k: st.session_state[k] for k in ("results", "is_fake", "build_version", "build_config")},
        # The feature and model toggles the build used, the live stocking rules, and page results added
        # later by remember_result. Without the model ones a reloaded session
        # would show Build Forecast's "changes vs the last run" reporting a
        # model change nobody made.
        extras={k: v for k, v in st.session_state.items() if k.startswith(("feat_", "mdl_", "inv_rules"))},
        # The live dict, so targets typed later travel too (to disk on the next write).
        ov_targets=st.session_state.setdefault("_ov_targets", {}),
    )
    st.session_state["_saved_token"] = saved["token"]
    _write_saved_build()


def remember_result(key):
    """Saves a slow page result (walk-forward, ablation, cross-check, workbook) with the build."""
    saved = _saved_build() if _last_build_path() else {}
    if saved and saved["token"] == st.session_state.get("_saved_token"):
        saved["extras"][key] = st.session_state[key]
        _write_saved_build()


def restore_build():
    """First run of a new session: pick up the saved build, if there is one."""
    saved = _saved_build() if _last_build_path() else {}
    if "results" in st.session_state or not saved or saved["code"] != _code_fingerprint():
        return False
    st.session_state.update(saved["state"])
    st.session_state.update(saved["extras"])
    st.session_state.update(_ov_targets=saved["ov_targets"], _saved_token=saved["token"],
                            _restored_at=saved["built"])
    return True
