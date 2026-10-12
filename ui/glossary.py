"""The Glossary dialog, read from GLOSSARY.md.
"""
import html
import re

import streamlit as st

from ui.components import empty_state


@st.cache_data(show_spinner=False)
def glossary_sections(path="GLOSSARY.md"):
    """Part A of GLOSSARY.md (the forecasting / inventory vocabulary; Part B is about
    working on the code) as [(heading, header cells, rows of cells)], one per '###' table.
    A heading's '(...)' aside is dropped, as on the board."""
    try:
        with open(path, encoding="utf-8") as f:
            part_a = f.read().split("\n## Part B")[0]
    except OSError:
        return []
    sections = []
    for block in part_a.split("\n### ")[1:]:
        lines = block.splitlines()
        table = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines if ln.startswith("|")]
        if len(table) > 2:
            sections.append((lines[0].split(" (")[0].strip(), table[0], table[2:]))
    return sections


def _md_inline(s):
    """GLOSSARY.md's inline Markdown (**bold**, *italic*, `code`) as escaped HTML."""
    s = html.escape(s, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"\*(.+?)\*", r"<em>\1</em>", s)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", s)


def _heading_id(title):
    """A glossary card's anchor id: 'Model types' -> 'gl-model-types'."""
    return "gl-" + "-".join(re.findall(r"[a-z0-9]+", title.lower()))


@st.dialog("Glossary", width="large")
def show_glossary(sections):
    """The board's Glossary page as a pop-up: search and "On this page" jump links on
    the left, one card per GLOSSARY.md section on the right."""
    n_terms = sum(len(rows) for _, _, rows in sections)
    st.markdown('<div class="page-subtitle gl-subtitle">Words that make the forecast and the stocking plan easier to talk about</div>',
                unsafe_allow_html=True)
    aside, body = st.container(key="plain_glossary_detail").columns([300, 900], gap="medium")
    with aside, st.container(border=True):
        q = st.text_input(f"Search {n_terms} terms", placeholder="For example WAPE or safety stock",
                          key="glossary_q").strip().lower()
        shown = [(h, head, [r for r in rows if q in re.sub(r"[*`]", "", " ".join(r)).lower()])
                 for h, head, rows in sections]
        shown = [s for s in shown if s[2]]
        st.markdown('<div class="gl-kicker">On this page</div>' + "".join(
            f'<a class="gl-link" href="#{_heading_id(h)}"><span>{html.escape(h)}</span><span>{len(rows)}</span></a>'
            for h, _, rows in shown), unsafe_allow_html=True)
    with body:
        st.caption("The forecasting and inventory vocabulary this project runs on. Use these terms in the deck and with mentors.")
        if not shown:
            empty_state(f"No term matches “{html.escape(q)}”.")
        for h, head, rows in shown:
            cols = "220px minmax(0, 1fr)" + (" 300px" if len(head) > 2 else "")
            lines = [f'<div class="gl-row gl-head">{"".join(f"<div>{html.escape(c)}</div>" for c in head)}</div>']
            lines += [f'<div class="gl-row">{"".join(f"<div>{_md_inline(c)}</div>" for c in r)}</div>' for r in rows]
            with st.container(border=True):
                st.markdown(f'<div class="gl-title" id="{_heading_id(h)}">{html.escape(h)}</div>'
                            f'<div class="gl-table" style="--gl-cols: {cols};">{"".join(lines)}</div>',
                            unsafe_allow_html=True)
