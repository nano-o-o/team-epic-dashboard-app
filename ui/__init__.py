"""The dashboard's UI, split out of dashboard_app.py (the app shell and entry point).

    config.py       fixed settings: build defaults, feature toggles, labels, colours, navigation
    styles.py       the CSS (global stylesheet + the per-page type scale)
    components.py   small helpers every page uses: formatting, pills, KPI cards, titles, tables
    charts.py       Altair helpers: axes, period bucketing, trend chart, fill-rate frontier
    pipeline.py     the cached build (build_everything) and the slow on-demand extras
    saved_build.py  keeps the last build across reloads and restarts
    glossary.py     the Glossary dialog
    pages/          one module per page; each has render(**build data it needs)

Imports only flow downwards: config <- styles <- components <- charts / pipeline <- pages.
"""
