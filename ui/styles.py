"""The app's CSS: the global stylesheet (Forecast Console design system) and the
type/spacing scale that page_title() adds on the redesigned pages.
"""
import streamlit as st

from ui.config import (
    BG, BORDER_CONTROL, BORDER_SUBTLE, CARD, CARD_RADIUS, NAVY, NAVY_DARK, NAV_ACTIVE,
    NAV_GROUP_STARTS, ORANGE, TEXT_MAIN, TEXT_MUTED,
)


# Full light theme: .streamlit/config.toml sets Streamlit's own base theme
# to light and matches these same colors, so native components (tables,
# tooltips, inputs) already render with correctly-contrasted text
# automatically -- this CSS only adds custom card/sidebar styling on top,
# it does NOT need to fight or re-assert Streamlit's own text color choices.
# (This is the same principle that fixed the earlier dark-theme migration:
# align with Streamlit's theme system instead of overriding it piecemeal.)
# Build Forecast shows one step at a time. The steps that are shut still
# RENDER their widgets -- Streamlit drops the state of a widget a run does not
# instantiate, and the build reads all three steps' values in one run -- so they
# are hidden here rather than skipped. This runs before the stylesheet because an
# Edit or Back/Next click is handled as a callback, i.e. before the script body,
# so the rule below is already right for the step that click opened.
def inject_global_css():
    """The global stylesheet, written once per run (and the Build Forecast step hiding)."""
    st.session_state["_bf_step"] = min(st.session_state.get("_bf_step", 1), 3)
    _STEP_HIDE_CSS = "".join(
        f'[class*="st-key-plain_body{_n}"] {{ display: none !important; }}'
        for _n in (1, 2, 3) if _n != st.session_state.get("_bf_step", 1))

    st.markdown(f"""
    <style>
        {_STEP_HIDE_CSS}
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
        .stApp {{ background-color: {BG}; }}
        .stApp, .stApp p, .stApp label, .stApp input, .stApp textarea, .stApp button p {{ font-family: "IBM Plex Sans", system-ui, sans-serif; }}
        header[data-testid="stHeader"] {{ background: transparent; }}
        /* Title close to the top of the page, level with the sidebar logo. */
        div[data-testid="stMainBlockContainer"] {{ padding-top: 2.5rem; }}

        /* Sidebar as a floating white card on the grey page. The section itself
           goes transparent and gets an inset; its content box carries the white
           fill and the rounded corners. */
        section[data-testid="stSidebar"] {{
            background-color: transparent !important; border-right: none; box-shadow: none;
            padding: 20px 0 20px 20px;
        }}
        section[data-testid="stSidebar"] > div[data-testid="stSidebarContent"] {{
            background-color: {CARD}; border-radius: 20px;
        }}
        section[data-testid="stSidebar"] div[data-testid="stSidebarHeader"] {{ padding-top: 22px; }}
        /* Streamlit leaves 96px of empty padding under the sidebar content, enough
           to push the 8 nav rows into a scrollbar on a 900px-tall screen. */
        section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {{ padding-bottom: 24px; }}
        .nav-divider {{ border-top: 3px solid {NAV_ACTIVE}; margin: 0 0 24px 0; }}
        .sidebar-wordmark {{ font-size: 22px; font-weight: 800; letter-spacing: -0.3px; color: {TEXT_MAIN}; margin: 4px 0 18px 0; }}
        /* Sidebar nav: plain rows with an icon, a grey pill on the current page,
           and the radio circle hidden -- it reads as a menu, not a form control. */
        section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 6px; width: 100%; }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label {{
            width: 100%; background-color: transparent; border: none; border-radius: 10px;
            padding: 3px 14px; margin: 0; transition: background-color 0.15s ease;
        }}
        /* Streamlit 1.63 option markup: label > span(hidden input) > div > div >
           [div(circle), div(stMarkdownContainer)]. The icon is a role="img" span. */
        section[data-testid="stSidebar"] div[role="radiogroup"] > label > div > div > div:first-child:not([data-testid]) {{ display: none; }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label p {{ font-size: 16px; font-weight: 500; color: {TEXT_MAIN}; }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label p [role="img"] {{ font-size: 22px; margin-right: 12px; vertical-align: middle !important; color: {TEXT_MAIN}; }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{ background-color: #F5F5F5; }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-selected="true"] {{ background-color: {NAV_ACTIVE}; }}

        /* Page title -- the large bold heading at the top of every page. */
        .page-title {{ font-size: 28px; font-weight: 600; line-height: 34px; letter-spacing: -0.01em; color: {TEXT_MAIN}; }}
        .page-subtitle {{ color: {TEXT_MUTED}; font-size: 13px; margin-top: 4px; }}

        /* Cards -- white on the grey page, borderless, rounded. */
        /* The card is a size container so the number can scale to the card's own
           width: full 30px in a 3-card row, smaller in Inventory Plan's 4-card row,
           where "฿588,091,558" at 30px used to break in the middle of the number.
           word-break normal lets text values wrap at spaces, never inside a number. */
        .kpi-card {{ background-color: {CARD}; border-radius: {CARD_RADIUS}; padding: 22px 24px; min-height: 150px;
            border: 1px solid {BORDER_SUBTLE}; box-shadow: 0 1px 2px rgba(22, 24, 29, 0.06);
            container-type: inline-size; }}
        .kpi-label {{ color: {TEXT_MUTED}; font-size: 13px; font-weight: 600; margin-bottom: 8px; }}
        .kpi-value {{ color: {TEXT_MAIN}; font-size: min(30px, 12.5cqi); font-weight: 700; line-height: 1.1;
            overflow-wrap: normal; word-break: normal; }}
        /* The design system's KpiCard delta: status tones, not the chart variance colours. */
        .kpi-delta-up {{ color: #0A6B7A; font-size: 13px; font-weight: 600; margin-top: 8px; }}
        .kpi-delta-down {{ color: #B42318; font-size: 13px; font-weight: 600; margin-top: 8px; }}
        /* A row of KPI cards shares one height: a card whose note wraps no longer stands
           taller than its neighbours. Only for columns holding just the card -- where a
           caption follows, stretching the card would push the captions out of line. */
        div[data-testid="stColumn"]:has(.kpi-card):not(:has([data-testid="stCaptionContainer"])) > div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] {{ flex: 1 1 auto; }}
        div[data-testid="stColumn"]:has(.kpi-card):not(:has([data-testid="stCaptionContainer"])) :is([data-testid="stMarkdown"], [data-testid="stMarkdown"] > div) {{ height: 100%; }}
        /* +1rem undoes Streamlit's -1rem bottom margin on Markdown, which otherwise leaves the
           stretched cards 1rem shorter than the tallest one. The target card gets the same
           margin so the Overview row measures the same way. */
        div[data-testid="stColumn"]:has(.kpi-card):not(:has([data-testid="stCaptionContainer"])) [data-testid="stMarkdownContainer"] {{ height: calc(100% + 1rem) !important; }}
        div[data-testid="stColumn"]:has(.kpi-card):not(:has([data-testid="stCaptionContainer"])) .kpi-card {{ min-height: max(150px, 100%); box-sizing: border-box; }}
        /* The comparison a KPI's arrow is measured against, or its exact value. */
        .kpi-note {{ color: {TEXT_MUTED}; font-size: 12px; font-weight: 500; margin-top: 4px; }}
        /* Digits of equal width, so numbers in a row of cards line up. */
        .kpi-value, .profile-value, .build-card-value {{ font-variant-numeric: tabular-nums; }}
        /* Sidebar build summary: what every page is showing. */
        .build-card {{ border-top: 3px solid {NAV_ACTIVE}; padding: 6px 14px 0 14px; margin-top: 8px; }}
        .build-card-label {{ color: {TEXT_MUTED}; font-size: 11px; font-weight: 700; letter-spacing: 0.6px;
            text-transform: uppercase; margin-top: 10px; }}
        .build-card-value {{ color: {TEXT_MAIN}; font-size: 13px; font-weight: 600; line-height: 1.35;
            overflow-wrap: anywhere; }}
        .build-card-flag {{ color: {ORANGE}; }}
        .build-card-note {{ color: {TEXT_MUTED}; font-size: 12px; line-height: 16px; margin-top: 2px; }}
        /* Info / success notes in the app's own navy and grey, not Streamlit's blue and
           green. Warnings and errors keep their colours -- those need attention. */
        div[data-testid="stAlertContentInfo"], div[data-testid="stAlertContentSuccess"] {{ color: {TEXT_MAIN}; }}
        div[data-testid="stAlert"]:has(div[data-testid="stAlertContentInfo"]) > div,
        div[data-testid="stAlert"]:has(div[data-testid="stAlertContentSuccess"]) > div {{
            background-color: {NAV_ACTIVE}; border: none; border-radius: 8px; }}
        .side-card {{ background-color: {CARD}; border-radius: {CARD_RADIUS};
            border: 1px solid {BORDER_SUBTLE}; box-shadow: 0 1px 2px rgba(22, 24, 29, 0.06);
            padding: 22px 24px; color: {TEXT_MAIN}; margin-bottom: 14px; }}
        .side-card-label {{ color: {TEXT_MUTED}; font-size: 11px; font-weight: 700; letter-spacing: 0.8px; margin-bottom: 10px; }}
        /* Compact profile card -- one fact (customer type, industry, category...).
           Same white fill, border and radius as the KPI cards, but auto height and
           a text-sized value: these hold words, not big numbers, and a long FA Sub
           Industry name must wrap rather than be clipped. min-height (not height:100%,
           which Streamlit columns do not stretch) keeps a row of cards level when one
           value wraps to a second line. */
        .profile-card {{ background-color: {CARD}; border-radius: {CARD_RADIUS};
            border: 1px solid {BORDER_SUBTLE};
            padding: 14px 18px; min-height: 88px; }}
        .profile-label {{ color: {TEXT_MUTED}; font-size: 11px; font-weight: 700; letter-spacing: 0.6px;
            text-transform: uppercase; margin-bottom: 4px; }}
        .profile-value {{ color: {TEXT_MAIN}; font-size: 16px; font-weight: 600; line-height: 1.3;
            overflow-wrap: anywhere; }}
        .profile-note {{ color: {TEXT_MUTED}; font-size: 12px; font-weight: 500; margin-top: 2px; }}
        /* Inset variant: the same tile INSIDE a white card (the per-model error row on
           Model Accuracy), where a white tile would be invisible. The accent value is
           orange -- the app's "this is the one we use" colour. */
        .profile-card-inset {{ background-color: {BG}; min-height: 74px; }}
        .profile-value-accent {{ color: {ORANGE}; }}
        /* st.container(border=True) card -- white fill, no visible border and the
           same radius as the KPI cards. Streamlit 1.63 renders a bordered container
           as a stVerticalBlock directly inside a stLayoutWrapper. The previous
           selector targeted the old stVerticalBlockBorderWrapper test id, which no
           longer exists -- so every chart card was silently left transparent on the
           grey page background. */
        /* Unbordered containers share this markup; the ones the app makes are keyed
           "plain_..." (the sidebar card slot, and the container inside each page fragment,
           whose unkeyable wrapper is recognised by that child) and stay transparent.
           Only the fragment bodies ("plain_..._detail") mark a wrapper: a real card whose
           first child is a plain_ container (Build Forecast's step bodies) stays white. */
        div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:not([class*="st-key-plain_"]):not(:has(> div[data-testid="stLayoutWrapper"] > [class*="st-key-plain_"]:where([class*="_detail"]))) {{
            background-color: {CARD} !important;
            border: 1px solid {BORDER_SUBTLE} !important;
            box-shadow: 0 1px 2px rgba(22, 24, 29, 0.06);
            border-radius: {CARD_RADIUS} !important;
            padding: 1.25rem 1.5rem !important;
        }}
        /* Every expander is the design system's Disclosure: a hairline above, a bold
           summary with +/- in place of the chevron, the body muted and indented. */
        div[data-testid="stExpander"] details {{
            background-color: transparent !important; border: none !important;
            border-top: 1px solid {BORDER_SUBTLE} !important; border-radius: 0 !important;
        }}
        div[data-testid="stExpander"] summary {{ padding: 12px 0 0 !important; }}
        div[data-testid="stExpander"] summary:hover {{ background: transparent !important; }}
        div[data-testid="stExpander"] summary p {{ font-weight: 600; }}
        div[data-testid="stExpander"] summary span:has(> :is([data-testid="stExpanderIcon"], [data-testid="stIconMaterial"])) {{ display: none; }}
        div[data-testid="stExpander"] summary::before {{ content: "+"; width: 16px; margin-right: 8px;
            text-align: center; font-weight: 600; color: {TEXT_MAIN}; }}
        div[data-testid="stExpander"] details[open] > summary::before {{ content: "−"; }}
        div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {{ padding: 8px 0 4px 24px !important; }}
        /* The Overview FY-target card is a real container (it holds a number input),
           so it does not get .kpi-card's min-height from the HTML cards beside it.
           With a target set it holds a label, the input and two lines of gap text,
           which at the usual 150px ran into its bottom edge -- so this one KPI row is
           170px, the three HTML cards included (scoped with :has, so every other
           page's KPI row keeps 150px). st.container(key=...) renders the key as an
           "st-key-" class. */
        .st-key-ov_target_card [data-testid="stWidgetLabel"] p {{ color: {TEXT_MUTED};
            font-size: 13px; font-weight: 600; }}
        .target-gap-note {{ color: {TEXT_MUTED}; font-size: 12px; font-weight: 500; margin-top: 2px; }}

        /* --- Forecast Console: status pills, target bar, attention cards -----------
           Tones come from the design system's tokens (--status-*): teal for good, amber
           for warn, red for bad. They are deliberately NOT the app's GREEN/RED, which
           mark a variance being up or down, not a state needing attention. */
        .fc-pill {{ display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px;
            border-radius: 999px; font-size: 12px; line-height: 16px; font-weight: 600; white-space: nowrap; }}
        .fc-pill--good {{ background: #E0F0F3; color: #0A6B7A; }}
        .fc-pill--warn {{ background: #FBEED3; color: #7A4D00; }}
        .fc-pill--bad {{ background: #FBE3E0; color: #B42318; }}
        .fc-pill--neutral {{ background: {NAV_ACTIVE}; color: {TEXT_MUTED}; }}

        /* The board's identity line above a detail page's KPI row: the item or
           customer, its status pill and one muted line of facts. */
        .ident-row {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
            margin-bottom: 16px; }}
        .ident-name {{ font-size: 24px; line-height: 32px; font-weight: 600; color: {TEXT_MAIN};
            letter-spacing: -0.01em; }}
        .ident-meta {{ font-size: 14px; line-height: 20px; color: {TEXT_MUTED}; }}
        /* Model Accuracy, "Monthly miss by model": one small card per model, the chart's legend. */
        .miss-tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px;
            margin-bottom: 8px; }}
        .miss-tile {{ border: 1px solid {BORDER_SUBTLE}; border-radius: 8px; padding: 12px 16px; background: {CARD}; }}
        .miss-tile--on {{ border-color: {ORANGE}; }}
        .miss-name {{ display: flex; align-items: center; gap: 8px; font-size: 12px; line-height: 16px;
            font-weight: 600; color: {TEXT_MUTED}; }}
        .miss-sw {{ flex-shrink: 0; width: 18px; height: 3px; border-radius: 2px; }}
        .miss-val {{ font-size: 24px; line-height: 28px; font-weight: 600; color: {TEXT_MAIN}; margin-top: 8px;
            font-variant-numeric: tabular-nums; }}
        .miss-tile--on .miss-val {{ color: {ORANGE}; }}
        /* Customer Detail: each category's all-years total and share, the chart's legend. */
        /* No 1rem pull-up under the list, so the stretched chart beside it ends where it does. */
        [data-testid="stMarkdownContainer"]:has(> .cd-share) {{ margin-bottom: 0; }}
        .cd-share-h {{ font-size: 12px; line-height: 16px; font-weight: 600; color: {TEXT_MUTED}; margin-bottom: 4px; }}
        .cd-share-row {{ display: flex; align-items: center; gap: 8px; padding: 6px 0;
            border-top: 1px solid {BORDER_SUBTLE}; font-size: 13px; line-height: 18px; color: {TEXT_MAIN}; }}
        .cd-sw {{ flex-shrink: 0; width: 12px; height: 12px; border-radius: 2px; }}
        .cd-share-name {{ flex-grow: 1; min-width: 0; overflow-wrap: anywhere; }}
        .cd-share-val {{ font-weight: 600; font-variant-numeric: tabular-nums; white-space: nowrap; }}
        .cd-share-val em {{ font-style: normal; font-weight: 500; color: {TEXT_MUTED}; display: inline-block;
            min-width: 38px; text-align: right; }}

        /* Target bar: the forecast as a filled track, its p10-p90 range as a band on top
           of it, and the target as a dashed tick. The 40px/28px margins make room for the
           tick's label above the track. */
        .tgt-value {{ color: {TEXT_MAIN}; font-size: 36px; line-height: 40px; font-weight: 600;
            letter-spacing: -0.01em; font-variant-numeric: tabular-nums; }}
        .tgt-status {{ display: flex; align-items: baseline; gap: 10px; margin-top: 8px; flex-wrap: wrap; }}
        .tgt-status .kpi-note {{ margin-top: 0; }}
        .tgt-track {{ position: relative; height: 16px; margin: 40px 0 28px; background: {NAV_ACTIVE};
            border-radius: 4px; }}
        .tgt-fill {{ position: absolute; left: 0; top: 0; bottom: 0; background: {NAVY};
            border-radius: 4px 0 0 4px; }}
        .tgt-range {{ position: absolute; top: 4px; bottom: 4px; background: rgba(184, 84, 0, 0.15);
            border: 1px solid {ORANGE}; border-radius: 2px; }}
        .tgt-tick {{ position: absolute; top: -10px; bottom: -10px; width: 0;
            border-left: 2px dashed {TEXT_MAIN}; }}
        .tgt-tick span {{ position: absolute; top: -20px; transform: translateX(-50%); font-size: 12px;
            font-weight: 600; white-space: nowrap; color: {TEXT_MAIN}; }}

        /* Attention cards: pill, big count, title, body, then the button. space-between
           pins the button to the bottom of the card (the board's .fc-attn rule), so three
           cards of different text lengths still line their buttons up. */
        div[data-testid="stVerticalBlock"][class*="st-key-attn_card_"] {{ justify-content: space-between; }}
        /* Secondary button, as on the board: white with a hairline, not the navy of a
           primary action. Scoped to these cards so no other button changes. */
        div[class*="st-key-attn_card_"] .stButton button, .st-key-_ov_biz .stButton button {{
            background-color: {CARD} !important; color: {TEXT_MAIN} !important;
            border: 1px solid {BORDER_CONTROL} !important; width: auto !important; }}
        div[class*="st-key-attn_card_"] .stButton button:hover, .st-key-_ov_biz .stButton button:hover {{
            background-color: {NAV_ACTIVE} !important; color: {TEXT_MAIN} !important; }}
        /* The section's own link sits at the right of its heading row. */
        .st-key-_ov_biz .stButton {{ text-align: right; }}
        .attn-count {{ color: {TEXT_MAIN}; font-size: 24px; line-height: 28px; font-weight: 600;
            margin-top: 8px; font-variant-numeric: tabular-nums; }}
        .attn-title {{ color: {TEXT_MAIN}; font-size: 16px; line-height: 24px; font-weight: 600; }}
        .attn-body {{ color: {TEXT_MUTED}; font-size: 13px; line-height: 20px; margin-top: 8px; }}
        /* --- Build Forecast: the three-step strip and its step cards ---------------
           The strip sits on the page background as it does on the board; each step
           card repeats its numbered badge so a card still says which step it is when
           the strip has scrolled away. Teal = done, navy = the step you are on. */
        .fc-steps {{ display: flex; gap: 16px; margin: 0 0 36px; padding: 0; list-style: none; }}
        .fc-step {{ flex: 1 1 0; display: flex; align-items: center; gap: 12px;
            padding: 4px 4px 18px !important; margin: 0 !important;
            border-bottom: 3px solid transparent; }}
        .fc-step--on {{ border-bottom-color: {NAVY}; }}
        .fc-num {{ flex-shrink: 0; width: 32px; height: 32px; border-radius: 50%; display: flex;
            align-items: center; justify-content: center; font-size: 14px; font-weight: 600;
            background: {NAV_ACTIVE}; color: {TEXT_MUTED}; }}
        .fc-num--done {{ background: #0A6B7A; color: #FFFFFF; }}
        .fc-num--on {{ background: {NAVY}; color: #FFFFFF; }}
        .fc-step-name {{ display: block; font-size: 14px; font-weight: 600; color: {TEXT_MAIN}; }}
        .fc-step-sub {{ display: block; font-size: 12px; line-height: 16px; color: {TEXT_MUTED}; }}
        /* A step card's header: badge, title, one-line state, and its pill at the right. */
        .fc-hdr {{ display: flex; align-items: center; gap: 14px; margin-bottom: 14px; }}
        /* A section inside a step card. Streamlit renders ##### at 20px, the same
           size as the step title, which flattens the hierarchy -- the board sets
           these one level down at 16px. */
        .fc-sec {{ font-size: 16px; line-height: 24px; font-weight: 600; color: {TEXT_MAIN};
            margin: 8px 0 12px; }}
        .fc-hdr-title {{ font-size: 20px; line-height: 28px; font-weight: 600; color: {TEXT_MAIN}; }}
        .fc-hdr-title--sm {{ font-size: 16px; line-height: 24px; }}
        .fc-hdr-sub {{ font-size: 13px; line-height: 18px; color: {TEXT_MUTED}; }}
        /* Small uppercase group heading, inside a card (feature groups, master-file
           sections). Same treatment as .profile-label, which is tied to that tile. */
        .fc-grp {{ font-size: 11px; line-height: 16px; font-weight: 600; letter-spacing: 0.06em;
            text-transform: uppercase; color: {TEXT_MUTED}; margin: 20px 0 6px; }}
        /* A step's name and state are one line each on the board; the main column is
           narrower here, and "Inventory rules" was wrapping the strip out of line. */
        .fc-step-name, .fc-step-sub {{ white-space: nowrap; }}
        /* Fiscal-year tiles: the board draws them white with a hairline, not the grey
           inset the model tiles on Model Accuracy use. */
        div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"][class*="st-key-fy_tiles"][class*="st-key-fy_tiles"][class*="st-key-fy_tiles"][class*="st-key-fy_tiles"] {{
            padding: 0 !important; margin-bottom: 16px !important;
            background-color: transparent !important; border: 0 !important; box-shadow: none !important; }}
        .st-key-fy_tiles .profile-card-inset {{ background-color: {CARD};
            border: 1px solid {BORDER_SUBTLE}; }}
        /* One bordered row per model, navy once it is ticked, its one-line note inside
           at 12px and indented under the label -- the board's "Models to compare". */
        div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"][class*="st-key-mdlopt_"][class*="st-key-mdlopt_"][class*="st-key-mdlopt_"][class*="st-key-mdlopt_"] {{
            padding: 10px 16px !important; border: 1px solid {BORDER_SUBTLE} !important;
            border-radius: 8px !important; gap: 0 !important; }}
        div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"][class*="st-key-mdlopt_"][class*="st-key-mdlopt_"][class*="st-key-mdlopt_"][class*="st-key-mdlopt_"]:has(label[data-selected="true"]) {{
            border-color: {NAVY} !important; }}
        div[class*="st-key-mdlopt_"] [data-testid="stCaptionContainer"] {{ margin-left: 28px; }}
        div[class*="st-key-mdlopt_"] [data-testid="stCaptionContainer"] p {{ font-size: 12px;
            line-height: 16px; }}
        /* Select all / Deselect all are secondary buttons and Reset a ghost, each at its
           own width -- not three navy blocks stretched across the card. */
        div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"].st-key-feat_btns {{
            padding: 0 !important; }}
        .st-key-feat_btns .stButton button {{ background-color: {CARD} !important;
            color: {TEXT_MAIN} !important; border: 1px solid {BORDER_CONTROL} !important; }}
        .st-key-feat_btns .stButton button:hover {{ background-color: {NAV_ACTIVE} !important; }}
        .st-key-feat_btns .st-key-_feat_btn_reset button {{ border-color: transparent !important; }}
        /* Step 2's footer: back and next as the board draws them. They are jump links to
           the other step cards, which are all on the page at once here. */
        /* The open step's footer: a hairline, then a quiet Back at the left and the
           navy Next at the right. A shut step's Edit button gets the same quiet
           treatment, and neither stretches across the card, as on the board. */
        /* go_to()'s scroll-to-top script: runs, takes no space. */
        div[data-testid="stLayoutWrapper"]:has(> .st-key-plain_scroll_top) {{ display: none; }}
        .st-key-plain_step_nav {{ margin-top: 24px; padding: 18px 0 16px;
            border-top: 1px solid {BORDER_SUBTLE}; }}
        /* The block is a flex column whose children shrink to their content, so the
           Next button is pushed over by aligning the column, not its child. */
        [class*="st-key-plain_nav_right"] {{ align-items: flex-end; }}
        /* Two classes, not one: the page's own ".stButton button" rule comes later in
           this sheet, so a single-class selector loses to it and these stayed navy. */
        [class*="st-key-_bf_back"] .stButton button, [class*="st-key-_bf_edit"] .stButton button {{
            background-color: {CARD} !important; color: {TEXT_MAIN} !important;
            border: 1px solid {BORDER_CONTROL} !important; }}
        [class*="st-key-_bf_back"] .stButton button:hover, [class*="st-key-_bf_edit"] .stButton button:hover {{
            background-color: {NAV_ACTIVE} !important; color: {TEXT_MAIN} !important; }}
        /* Master files: one small bordered card per file inside the Data step. */
        div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"][class*="st-key-mastercard_"][class*="st-key-mastercard_"][class*="st-key-mastercard_"][class*="st-key-mastercard_"] {{
            padding: 16px !important; border: 1px solid {BORDER_SUBTLE} !important; box-shadow: none;
            border-radius: 8px !important; gap: 12px !important; }}
        /* Both master cards keep their buttons on the card's bottom edge, level with each other. */
        [data-testid="stLayoutWrapper"]:has(> [class*="st-key-plain_mst_btns_"]) {{ margin-top: auto; }}
        /* Forecast chart legend (chart_legend), as the design system's ForecastChart draws it. */
        .fc-legend {{ display: flex; gap: 24px; flex-wrap: wrap; font-size: 13px; line-height: 20px; color: {TEXT_MAIN};
                      margin-bottom: 8px; padding-left: 86px; }}  /* 86px = the y axis (fixed in trend_chart) */
        .fc-legend span {{ display: inline-flex; align-items: center; gap: 8px; }}
        .fc-legend svg {{ width: 28px; height: 10px; }}
        .fc-mst-head {{ display: flex; align-items: center; gap: 8px; }}
        .fc-mst-title {{ flex-grow: 1; font-size: 15px; line-height: 20px; font-weight: 600; color: {TEXT_MAIN}; }}
        .fc-mst-src {{ font-size: 13px; line-height: 18px; color: {TEXT_MUTED}; margin-top: 12px; }}
        .fc-mst-what {{ font-size: 13px; line-height: 18px; color: {TEXT_MAIN}; margin-top: 12px; }}
        /* Streamlit pulls a markdown block up by 1rem, which ran the text into the checkbox. */
        [class*="st-key-mastercard_"] [data-testid="stMarkdownContainer"]:has(.fc-mst-head, .fc-mst-note) {{
            margin-bottom: 0; }}
        [class*="st-key-mastercard_"] [data-testid="stCheckbox"] p {{ font-size: 14px; line-height: 20px; }}
        /* The checkbox's note, under its label as on the board. */
        .fc-mst-note {{ font-size: 12px; line-height: 16px; color: {TEXT_MUTED}; margin: -12px 0 0 24px; }}
        /* The design system's small buttons: Replace file secondary, Column names ghost. */
        div[class*="st-key-mastercard_"] [data-testid="stPopover"] button {{ min-height: 30px; height: 30px;
            padding: 0 12px; border-radius: 8px; }}
        div[class*="st-key-mastercard_"] [data-testid="stPopover"] button p {{ font-size: 13px; font-weight: 600; }}
        div[class*="st-key-mastercard_"] [data-testid="stPopover"] button[data-testid="stBaseButton-secondary"] {{
            background: {CARD}; border: 1px solid {BORDER_CONTROL}; color: {TEXT_MAIN}; }}
        div[class*="st-key-mastercard_"] [data-testid="stPopover"] button:hover {{ background: {NAV_ACTIVE}; color: {TEXT_MAIN}; }}
        /* The footer's right-hand note on the Run step. */
        .fc-nav-note {{ font-size: 13px; line-height: 18px; color: {TEXT_MUTED}; text-align: right; }}
        /* Inventory Plan, "Stocking rules": one small bordered card per rule with
           its input kept to a short width. */
        div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"][class*="st-key-rulecard_"][class*="st-key-rulecard_"][class*="st-key-rulecard_"][class*="st-key-rulecard_"] {{
            padding: 16px !important; border: 1px solid {BORDER_SUBTLE} !important;
            border-radius: 8px !important; gap: 6px !important; }}
        div[class*="st-key-rulecard_"] [data-testid="stNumberInput"] {{ max-width: 200px; }}
        div[class*="st-key-rulecard_"] [data-testid="stWidgetLabel"] p {{ font-size: 13px; font-weight: 600; }}
        div[class*="st-key-rulecard_"] [data-testid="stCaptionContainer"] p {{ font-size: 12px; line-height: 16px; }}
        /* Run step, "Forecast output options": one bordered row per option, navy once
           ticked, the title at 15px, the explanation under it and a pill at the right. */
        /* A horizontal container is a stHorizontalBlock, which the card rule never matches. */
        div[data-testid="stHorizontalBlock"][class*="st-key-runopt_"] {{
            padding: 16px; border: 1px solid {BORDER_SUBTLE}; border-radius: 8px; }}
        div[data-testid="stHorizontalBlock"][class*="st-key-runopt_"]:has(label[data-selected="true"]) {{
            border-color: {NAVY}; }}
        div[class*="st-key-runopt_"] [data-testid="stCheckbox"] label p {{ font-size: 15px; font-weight: 600; }}
        div[class*="st-key-runopt_"] [data-testid="stCaptionContainer"] {{ margin-left: 28px; }}
        div[class*="st-key-runopt_"] [data-testid="stCaptionContainer"] p {{ font-size: 13px; line-height: 18px; }}
        /* "Review your settings": label, value and a ghost Edit per row, hairlines between. */
        .st-key-plain_review {{ gap: 0 !important; border-bottom: 1px solid {BORDER_SUBTLE}; }}
        [class*="st-key-plain_rv_"] {{ border-top: 1px solid {BORDER_SUBTLE}; padding: 5px 0; gap: 12px !important;
            min-height: 41px; }}
        .fc-rv-k {{ display: block; width: 120px; font-size: 14px; line-height: 20px; color: {TEXT_MUTED}; }}
        .fc-rv-v {{ font-size: 14px; line-height: 20px; font-weight: 600; color: {TEXT_MAIN}; }}
        .fc-rv-was {{ font-weight: 500; color: {TEXT_MUTED}; }}
        [class*="st-key-_bf_rv_"] .stButton button {{ background-color: transparent !important;
            color: {TEXT_MAIN} !important; min-height: 30px; height: 30px; padding: 0 12px;
            border-radius: 8px !important; }}
        [class*="st-key-_bf_rv_"] .stButton button p {{ font-size: 13px; }}
        [class*="st-key-_bf_rv_"] .stButton button:hover {{ background-color: {NAV_ACTIVE} !important;
            color: {TEXT_MAIN} !important; }}
        /* Nobody enters a sales target by stepping 1,000 units or 1M baht at a time,
           and the two step buttons took 64px from a narrow input: a 10-digit baht
           figure was clipped to "2600000C" at a 1440px-wide window. */
        .st-key-ov_target_card [data-testid="stNumberInputStepDown"],
        .st-key-ov_target_card [data-testid="stNumberInputStepUp"] {{ display: none; }}
        /* Streamlit's "Press Enter to apply" hint is drawn INSIDE the input and sat on
           top of the typed number in this narrow box. The empty-state caption below
           says the same thing, and Tab / clicking away also applies the value. */
        .st-key-ov_target_card [data-testid="InputInstructions"] {{ display: none; }}
        /* The design system's small input and button under the bar: 36px tall, a control border. */
        .st-key-ov_target_card [data-testid="stNumberInputContainer"] {{ height: 36px; }}
        .st-key-_ov_set_target button {{ min-height: 36px; height: 36px; padding: 0 16px; }}
        /* Streamlit's chart theme writes the page background (config.toml
           backgroundColor, #EEEEEE) as an INLINE style on every Vega <svg>. An
           !important stylesheet rule beats an inline style, so paint it the card
           white rather than leaving a grey rectangle inside a white card. */
        [data-testid="stVegaLiteChart"] svg.marks {{ background-color: {CARD} !important; }}
        /* Dividers -- Streamlit's default hr carries a big vertical margin that reads
           as dead space (e.g. above the "experimental features" toggle). Tighten it. */
        hr, div[data-testid="stDivider"] {{ margin: 0.4rem 0 !important; }}
        /* Except the sidebar's one rule, between the build summary and the build notes:
           tightened to 0.4rem the notes read as the card's last line. !important because
           the rule above has it too -- without it this is silently overridden. */
        section[data-testid="stSidebar"] hr {{ margin: 22px 0 18px !important;
            border: 0 !important; border-top: 3px solid {NAV_ACTIVE} !important; background: none !important; }}
        h1, h2, h3, h4, h5, h6 {{ color: {TEXT_MAIN} !important; }}
        /* Streamlit gives h4/h5 no bottom margin, so a section or card title sat
           directly on its first line of content. One rule, so every page's "####
           page section" and "##### card title" get the same breathing room. */
        [data-testid="stMarkdownContainer"] :is(h4, h5) {{ margin-bottom: 12px !important; }}
        /* Buttons and download buttons -- dark navy instead of orange, guaranteed
           contrast regardless of any theme quirk (this has been a repeated
           trouble spot in earlier passes). */
        .stButton button, .stDownloadButton button {{
            background-color: {NAVY} !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }}
        /* Download and run buttons (keyed act_...): one padding and height, and a long
           label wraps inside the button instead of spilling out of it. */
        [data-testid="stMain"] :is(.stDownloadButton, [class*="st-key-act_"]) button {{
            min-height: 40px; height: auto !important; padding: 8px 16px !important; }}
        [data-testid="stMain"] :is(.stDownloadButton, [class*="st-key-act_"]) button p {{
            white-space: normal !important; overflow-wrap: anywhere; text-align: center; }}
        .stButton button:hover, .stDownloadButton button:hover {{
            background-color: {NAVY_DARK} !important;
            color: #FFFFFF !important;
        }}
        /* The sidebar's Glossary button reads as one more menu row, not an action. */
        section[data-testid="stSidebar"] .stButton button {{
            background-color: transparent !important; color: {TEXT_MAIN} !important; font-weight: 500 !important;
        }}
        /* Icon and label lined up with the nav rows above (measured in the browser). */
        section[data-testid="stSidebar"] .stButton button > div {{ width: 100%; justify-content: flex-start; }}
        section[data-testid="stSidebar"] .stButton button p {{ font-size: 16px; }}
        section[data-testid="stSidebar"] .stButton button [data-testid="stIconMaterial"] {{ font-size: 22px; }}
        section[data-testid="stSidebar"] .stButton button span:has(> [data-testid="stIconMaterial"]) {{ margin-right: 14px; }}
        section[data-testid="stSidebar"] .stButton button:hover {{ background-color: #F5F5F5 !important; }}
        /* SegmentedControl (design system): a sunken track with a control-grey
           outline, 28px segments, the selected one navy with white text. Every view
           switch (Units / Revenue, Monthly / Quarterly, grain) is an st.segmented_control. */
        [data-testid="stButtonGroup"] > [role="radiogroup"]:has(> button[data-variant="segmented_control"]) {{
            display: inline-flex; flex-wrap: nowrap; gap: 2px; padding: 2px; width: auto;
            background: {NAV_ACTIVE}; border: 1px solid #737884; border-radius: 8px; }}
        button[data-variant="segmented_control"] {{ height: 28px; min-height: 28px; padding: 0 12px;
            margin: 0 !important; border: 0 !important; border-radius: 6px !important;
            background: transparent !important; color: {TEXT_MAIN} !important; box-shadow: none !important; }}
        button[data-variant="segmented_control"] p {{ font-size: 14px; font-weight: 500; }}
        button[data-variant="segmented_control"]:hover {{ background: {CARD} !important; }}
        button[data-variant="segmented_control"][aria-checked="true"] {{ background: {NAVY} !important; }}
        button[data-variant="segmented_control"][aria-checked="true"] p {{ color: #FFFFFF !important; font-weight: 600; }}
        button[data-variant="segmented_control"]:focus-visible {{ outline: 2px solid #1A56DB; outline-offset: 2px; }}
        /* Redesign: the current page is marked by its pill and bold text (no red square: the user's call). */
        section[data-testid="stSidebar"] div[role="radiogroup"] > label {{ position: relative; }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label p {{ font-weight: 400; }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-selected="true"] p {{ font-weight: 600; }}
        /* Glossary is a pop-up, not a page: a footer item under a hairline, outside the nav groups. */
        section[data-testid="stSidebar"] .st-key-glossary_btn {{ border-top: 3px solid {NAV_ACTIVE}; padding-top: 16px; margin-top: 8px; }}
        /* Redesign: the board's nav group labels (Plan / Explore / Model), drawn above each
           group's first row so the radio -- and nav_page -- stay one widget. */
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:is({", ".join(f":nth-of-type({n})" for n in NAV_GROUP_STARTS)}) {{ margin-top: 34px; }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(1) {{ margin-top: 16px; }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label::after {{
            position: absolute; left: 14px; bottom: 100%; margin-bottom: 2px; font-size: 11px; line-height: 16px;
            font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: {TEXT_MUTED}; }}
        {" ".join(f'section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type({n})::after {{ content: "{g}"; }}' for n, g in NAV_GROUP_STARTS.items())}
        /* Glossary pop-up (the board's Glossary page): grey body so its white cards read as on a page,
           term tables as grids, jump links as the board's "On this page" rows. */
        div[role="dialog"]:has(.gl-table) {{ background: {BG}; }}
        /* Title and subtitle as page_title() draws them on the pages: 28/34 semibold, subtitle 4px below. */
        [data-testid="stDialog"]:has(.gl-table) h2, [data-testid="stDialog"]:has(.gl-table) h2 p {{ font-size: 28px; line-height: 34px; }}
        [data-testid="stDialog"] .gl-subtitle {{ margin-top: -28px; }}
        .gl-title {{ font-size: 20px; line-height: 28px; font-weight: 600; color: {TEXT_MAIN}; margin-bottom: 16px; }}
        .gl-table {{ font-size: 14px; line-height: 20px; color: {TEXT_MAIN}; }}
        .gl-row {{ display: grid; grid-template-columns: var(--gl-cols); column-gap: 24px; padding: 12px 0;
            border-top: 1px solid {BORDER_SUBTLE}; }}
        .gl-row.gl-head {{ padding: 0 0 8px; border-top: 0; font-size: 11px; line-height: 16px; font-weight: 600;
            letter-spacing: .06em; text-transform: uppercase; color: {TEXT_MUTED}; }}
        .gl-row > div:nth-child(1) {{ font-weight: 600; }}
        .gl-row > div:nth-child(3) {{ color: {TEXT_MUTED}; }}
        .gl-head > div {{ font-weight: 600 !important; }}
        .gl-table code {{ font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12px; color: {TEXT_MAIN};
            background: {NAV_ACTIVE}; padding: 1px 4px; border-radius: 4px; }}
        .gl-kicker {{ font-size: 11px; line-height: 16px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
            color: {TEXT_MUTED}; padding: 0 12px 4px; }}
        a.gl-link {{ display: flex; justify-content: space-between; gap: 8px; padding: 8px 12px; border-radius: 8px;
            font-size: 14px; color: {TEXT_MAIN} !important; text-decoration: none !important; }}
        a.gl-link:hover {{ background: {NAV_ACTIVE}; }}
        a.gl-link span:last-child {{ color: {TEXT_MUTED}; }}
    </style>
    """, unsafe_allow_html=True)


# The Forecast Console scale (design system tokens) for the redesigned pages:
# 16px between cards (space-4), 24px card padding (space-5), 48px from a band's
# cards to the next "#### section" (space-7: the <br> before it is sized to make
# that), a 12px step from a title to its content; headings 20/28, captions 12/16
# muted, KPI values 36/40. Emitted with the page title, so it lives only while
# one of these pages renders and takes no block of its own (a lone <style>
# element would still take a 16px gap). The card rule repeats the global
# selector so it wins by order, and still loses to the keyed special cards.
FC_SCALE_PAGES = ("Build Forecast", "Overview", "Inventory Plan", "Model Accuracy",
                  "Product Detail", "Customer Detail", "Category Plan", "Forecast Table")
FC_SCALE_CSS = f"""<style>
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] :is(h4, h5) {{ font-size: 20px !important;
        line-height: 28px !important; padding: 0 !important; margin: 0 0 11px !important; }}
    [data-testid="stMain"] [data-testid="stCaptionContainer"] p {{ font-size: 12px; line-height: 16px;
        color: {TEXT_MUTED}; }}
    div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:not([class*="st-key-plain_"]):not(:has(> div[data-testid="stLayoutWrapper"] > [class*="st-key-plain_"]:where([class*="_detail"]))) {{
        padding: 24px !important; }}
    div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"][class*="st-key-rulecard_"][class*="st-key-rulecard_"][class*="st-key-rulecard_"][class*="st-key-rulecard_"] {{
        gap: 8px !important; }}
    [data-testid="stMain"] [data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] [data-testid="stMarkdownContainer"] > br:only-child) {{
        height: 15px; }}
    /* A title beside controls (a 40px button or radio row): centred on them, its
       12px step taken by the row's own height, and a section row pulled up by the
       centring offset so the band still measures 48px to the title. */
    /* Only a title placed straight in a column: a title inside a card in a column
       must not centre the whole page row. */
    [data-testid="stMain"] [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] [data-testid="stMarkdownContainer"] :is(h4, h5) {{
        margin-bottom: 0 !important; }}
    [data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] :is(h4, h5)) {{
        align-items: center !important; }}
    [data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] h4) {{
        margin-top: -15px; }}
    /* A segmented control (34px) is shorter than a button, so its row adds the
       4px that keeps the title 12px off what follows. */
    [data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] :is(h4, h5)):has([data-testid="stButtonGroup"]) {{
        margin-bottom: 4px; }}
    /* A card title alone in its column beside a switch or button: Markdown's -1rem
       bottom margin left a 12px box for the 28px title, so centring that box set the
       title 8px low. (A title with a caption under it is not centred; left as is.) */
    [data-testid="stMain"] [data-testid="stHorizontalBlock"]:has([data-testid="stButtonGroup"], [data-testid="stButton"]) > [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:only-child [data-testid="stMarkdownContainer"]:has(h5) {{
        margin-bottom: 0 !important; }}
    /* The KPI cards stretch 1rem past their row to undo Markdown's -1rem margin,
       which ran them into the card below; the row reserves that 1rem. */
    [data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.kpi-card) {{ margin-bottom: 1rem; }}
    [data-testid="stMain"] .kpi-card {{ padding: 24px; }}
    [data-testid="stMain"] .kpi-label {{ font-size: 12px; line-height: 16px; margin-bottom: 8px; }}
    [data-testid="stMain"] .kpi-value {{ font-size: min(36px, 20cqi); line-height: 40px; font-weight: 600;
        letter-spacing: -0.01em; }}
    [data-testid="stMain"] .kpi-note {{ font-size: 12px; line-height: 16px; margin-top: 8px; }}
    [data-testid="stMain"] :is(.stButton, .stDownloadButton) button {{ border-radius: 8px !important; }}
    /* Product Detail: 24px between the page's sections, as the board's main column
       spaces them (16px block gap + 8px). */
    :is(.st-key-pd_rhythm, .st-key-pd_chart, .st-key-pd_monthly, .st-key-cd_chart, .st-key-cd_projection,
        [class*="st-key-cp_"], [class*="st-key-ft_"]) {{ margin-top: 8px; }}
    /* Order rhythm tiles in the design system's terms: white with a hairline
       (sunken fill is for table headers and control tracks), a sentence-case
       12px label, the value at the 16px subheading size, a 12px muted note. */
    /* The tiles' Markdown runs 1rem past their row, as the KPI cards do; the row
       reserves it so the card keeps its 24px bottom padding. */
    .st-key-pd_rhythm [data-testid="stHorizontalBlock"] {{ margin-bottom: 1rem; }}
    .st-key-pd_rhythm .profile-card-inset {{ background-color: {CARD}; border: 1px solid {BORDER_SUBTLE};
        padding: 16px; min-height: 0; }}
    .st-key-pd_rhythm .profile-label {{ font-size: 12px; line-height: 16px; font-weight: 600;
        letter-spacing: 0; text-transform: none; margin-bottom: 8px; }}
    .st-key-pd_rhythm .profile-value {{ font-size: 16px; line-height: 24px; }}
    .st-key-pd_rhythm .profile-note {{ font-size: 12px; line-height: 16px; font-weight: 400; margin-top: 4px; }}
</style>"""
