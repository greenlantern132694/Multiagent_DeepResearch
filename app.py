"""
DeepInsight - Streamlit UI for the multi-agent research pipeline.

Place this file next to agents.py, tools.py and pipeline.py, then run:
    streamlit run app.py

The UI re-uses the exact same agents and the same four steps as
pipeline.run_research_pipeline(), but runs them one at a time so the page can
show live progress while the work happens in the background.
"""

import time
import random
from datetime import datetime

import streamlit as st

from agents import create_reader_agent, create_search_agent, writer_chain, critic_chain

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="DeepInsight",
    page_icon="🔭",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------
# Styling
# Palette: Midnight #0E1328 | Panel #161D3A | Signal teal #3DD6C6
#          Lantern gold #F2C46D | Mist #AAB3D1 | Paper #EEF1FA
# --------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');

:root{
  --bg:#0E1328; --panel:#161D3A; --line:#2A3358;
  --teal:#3DD6C6; --gold:#F2C46D; --mist:#AAB3D1; --paper:#EEF1FA;
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"]{
  background: var(--bg); color: var(--paper);
  font-family: 'Sora', system-ui, sans-serif;
}
[data-testid="stHeader"]{ background: transparent; }
.block-container{ max-width: 780px; padding-top: 3rem; padding-bottom: 5rem; }
#MainMenu, footer{ visibility:hidden; }

.di-hero h1{
  font-size: 3.1rem; font-weight: 700; letter-spacing:-0.03em;
  margin: 0 0 .4rem 0; color: var(--paper);
}
.di-hero p{
  font-family:'Source Serif 4', Georgia, serif; font-size:1.2rem;
  color: var(--mist); line-height:1.55; margin:0 0 1.6rem 0; max-width: 34em;
}
.di-trust{ color: var(--mist); font-size:.82rem; margin-top:.9rem; }
.di-trust b{ color: var(--teal); font-weight:600; }

/* input */
[data-testid="stTextInput"] input{
  background: var(--panel); color: var(--paper);
  border: 1.5px solid var(--line); border-radius: 14px;
  padding: .95rem 1.1rem; font-size: 1.05rem; font-family:'Sora',sans-serif;
}
[data-testid="stTextInput"] input:focus{
  border-color: var(--teal); box-shadow: 0 0 0 3px rgba(61,214,198,.18);
}
[data-testid="stTextInput"] label{ display:none; }

/* buttons */
.stButton > button{
  background: var(--panel); color: var(--paper);
  border: 1px solid var(--line); border-radius: 999px;
  font-family:'Sora',sans-serif; font-size:.85rem; padding:.45rem 1rem;
  transition: border-color .15s, color .15s;
}
.stButton > button:hover{ border-color: var(--teal); color: var(--teal); }
.stButton > button:focus-visible{ outline: 2px solid var(--teal); outline-offset: 2px; }
.stButton > button[kind="primary"]{
  background: var(--teal); color:#07201D; border:none; font-weight:700;
  font-size:1rem; padding:.7rem 1.6rem; border-radius:14px; width:100%;
}
.stButton > button[kind="primary"]:hover{ background:#63e6d8; color:#07201D; }

/* stepper */
.di-steps{ display:flex; flex-direction:column; gap:.55rem; margin:1.2rem 0 .4rem 0; }
.di-step{
  display:flex; gap:.9rem; align-items:flex-start; padding:.85rem 1rem;
  border:1px solid var(--line); border-radius:14px; background: var(--panel);
  opacity:.5;
}
.di-step.active{ opacity:1; border-color: var(--teal); }
.di-step.done{ opacity:1; }
.di-dot{
  flex:0 0 auto; width:26px; height:26px; border-radius:50%;
  display:flex; align-items:center; justify-content:center;
  font-size:.8rem; font-weight:700; border:1.5px solid var(--line); color: var(--mist);
}
.di-step.active .di-dot{ border-color: var(--teal); color: var(--teal); animation: pulse 1.4s ease-in-out infinite; }
.di-step.done .di-dot{ background: var(--teal); border-color: var(--teal); color:#07201D; }
.di-step h4{ margin:0; font-size:.95rem; font-weight:600; color: var(--paper); }
.di-step p{ margin:.15rem 0 0 0; font-size:.82rem; color: var(--mist); }
@keyframes pulse{ 0%,100%{ box-shadow:0 0 0 0 rgba(61,214,198,.45);} 50%{ box-shadow:0 0 0 7px rgba(61,214,198,0);} }
@media (prefers-reduced-motion: reduce){ .di-step.active .di-dot{ animation:none; } }

.di-tip{
  border-left: 3px solid var(--gold); padding:.4rem .9rem; margin:.8rem 0 0 0;
  color: var(--mist); font-size:.86rem; font-family:'Source Serif 4',serif; font-style: italic;
}

/* result */
.di-done{
  background: linear-gradient(135deg, rgba(61,214,198,.12), rgba(242,196,109,.10));
  border:1px solid var(--line); border-radius:16px; padding:1.1rem 1.3rem; margin:1.4rem 0 1rem 0;
}
.di-done h3{ margin:0 0 .25rem 0; font-size:1.15rem; color: var(--gold); }
.di-done p{ margin:0; color: var(--mist); font-size:.9rem; }
.di-stats{ display:flex; gap:1.6rem; margin-top:.8rem; flex-wrap:wrap; }
.di-stats div{ font-size:.78rem; color: var(--mist); }
.di-stats b{ display:block; font-size:1.25rem; color: var(--paper); }

[data-testid="stTabs"] button[role="tab"]{ color: var(--mist); font-family:'Sora',sans-serif; }
[data-testid="stTabs"] button[aria-selected="true"]{ color: var(--teal); }
.di-report{ font-family:'Source Serif 4', Georgia, serif; font-size:1.06rem; line-height:1.75; }
.di-report h1,.di-report h2,.di-report h3{ font-family:'Sora',sans-serif; letter-spacing:-.01em; }
[data-testid="stDownloadButton"] button{
  background: var(--panel); color: var(--paper); border:1px solid var(--line); border-radius:12px;
}
[data-testid="stDownloadButton"] button:hover{ border-color: var(--gold); color: var(--gold); }
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Content
# --------------------------------------------------------------------------
STEPS = [
    ("Searching the web", "A search agent is finding recent, reliable sources on your topic."),
    ("Reading the best source", "A reader agent is opening the most relevant page for depth."),
    ("Writing your report", "A writer is combining everything into a structured draft."),
    ("Reviewing for quality", "A critic is checking the draft and scoring it honestly."),
]

TIPS = [
    "Specific topics give sharper reports. 'Solid-state batteries in EVs' beats 'batteries'.",
    "Every claim in your report traces back to sources gathered in step 1 and 2.",
    "A second pair of eyes helps. The critic reads your report before you do.",
    "Most people skim first, then read deeply. Your report is built for both.",
    "Good research is mostly good questions. You already asked one.",
]

EXAMPLES = [
    "Future of quantum computing",
    "AI regulation in the EU",
    "Solid-state batteries",
    "Rise of AI coding agents",
]

# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
if "topic" not in st.session_state:
    st.session_state.topic = ""
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: topic, state, seconds, at
if "current" not in st.session_state:
    st.session_state.current = None


def set_topic(value: str):
    st.session_state.topic = value


def render_stepper(placeholder, active: int, durations: dict):
    """Draw the 4-step progress list. active = index running; steps before it are done."""
    html = ['<div class="di-steps">']
    for i, (title, desc) in enumerate(STEPS):
        if i < active:
            cls, mark = "done", "✓"
            sub = f"Done in {durations.get(i, 0):.1f}s"
        elif i == active:
            cls, mark, sub = "active", str(i + 1), desc
        else:
            cls, mark, sub = "", str(i + 1), desc
        html.append(
            f'<div class="di-step {cls}"><div class="di-dot">{mark}</div>'
            f"<div><h4>{title}</h4><p>{sub}</p></div></div>"
        )
    html.append("</div>")
    placeholder.markdown("".join(html), unsafe_allow_html=True)


def run_pipeline_live(topic: str):
    """Same logic as pipeline.run_research_pipeline, with live UI updates."""
    state, durations = {}, {}
    stepper = st.empty()
    bar = st.progress(0, text="Getting started...")
    tip = st.empty()
    tip.markdown(f'<div class="di-tip">{random.choice(TIPS)}</div>', unsafe_allow_html=True)

    def start(i):
        render_stepper(stepper, i, durations)
        bar.progress(int(i / len(STEPS) * 100), text=f"Step {i + 1} of {len(STEPS)}: {STEPS[i][0]}")
        return time.time()

    def finish(i, t0):
        durations[i] = time.time() - t0

    t_all = time.time()

    # Step 1 - search
    t0 = start(0)
    search_agent = create_search_agent()
    search_result = search_agent.invoke(
        {"messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]}
    )
    state["search_results"] = search_result["messages"][-1].content
    finish(0, t0)

    # Step 2 - read
    t0 = start(1)
    tip.markdown(f'<div class="di-tip">{random.choice(TIPS)}</div>', unsafe_allow_html=True)
    reader_agent = create_reader_agent()
    reader_result = reader_agent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"Based on the following search results about '{topic}', "
                    f"pick the most relevant URL and scrape it for deeper content.\n\n"
                    f"Search Results:\n{state['search_results'][:800]}",
                )
            ]
        }
    )
    state["scraped_content"] = reader_result["messages"][-1].content
    finish(1, t0)

    # Step 3 - write
    t0 = start(2)
    tip.markdown(f'<div class="di-tip">{random.choice(TIPS)}</div>', unsafe_allow_html=True)
    research_combined = (
        f"SEARCH RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )
    state["report"] = writer_chain.invoke({"topic": topic, "research": research_combined})
    finish(2, t0)

    # Step 4 - critic
    t0 = start(3)
    tip.markdown(f'<div class="di-tip">{random.choice(TIPS)}</div>', unsafe_allow_html=True)
    state["feedback"] = critic_chain.invoke({"report": state["report"]})
    finish(3, t0)

    render_stepper(stepper, len(STEPS), durations)
    bar.progress(100, text="Complete")
    tip.empty()
    return state, time.time() - t_all


def build_markdown_export(topic: str, state: dict) -> str:
    return (
        f"# {topic}\n\n{state['report']}\n\n---\n\n## Critic review\n\n{state['feedback']}\n"
    )


def render_result(topic: str, state: dict, seconds: float, celebrate: bool = False):
    words = len(str(state["report"]).split())
    sources = str(state["search_results"]).count("http")

    st.markdown(
        f"""
<div class="di-done">
  <h3>Your report is ready</h3>
  <p>{topic}</p>
  <div class="di-stats">
    <div><b>{words:,}</b>words</div>
    <div><b>{max(sources, 1)}</b>sources found</div>
    <div><b>{seconds:.0f}s</b>total time</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    if celebrate:
        st.toast("Report complete", icon="✅")

    tab_report, tab_critic, tab_sources, tab_raw = st.tabs(
        ["Report", "Critic's review", "Sources", "Deep read"]
    )
    with tab_report:
        st.markdown(f'<div class="di-report">\n\n{state["report"]}\n\n</div>', unsafe_allow_html=True)
    with tab_critic:
        st.caption("An independent review of the report above, so you know where to trust it and where to double-check.")
        st.markdown(state["feedback"])
    with tab_sources:
        st.caption("What the search agent found.")
        st.markdown(state["search_results"])
    with tab_raw:
        st.caption("The full page content the reader agent pulled for extra depth.")
        st.markdown(state["scraped_content"])

    slug = "".join(c if c.isalnum() else "_" for c in topic.lower())[:40].strip("_") or "report"
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download report (.md)",
            data=build_markdown_export(topic, state),
            file_name=f"deepinsight_{slug}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Download report (.txt)",
            data=build_markdown_export(topic, state),
            file_name=f"deepinsight_{slug}.txt",
            mime="text/plain",
            use_container_width=True,
        )


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
st.markdown(
    """
<div class="di-hero">
  <h1>DeepInsight</h1>
  <p>Name a topic. Four AI agents search, read, write and review,
  then hand you a sourced report in a couple of minutes.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.text_input(
    "Research topic",
    key="topic",
    placeholder="What do you want to understand today?",
)

# Quick-start chips: lower the effort of the first step
st.caption("Not sure where to start? Try one:")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    col.button(ex, key=f"ex_{ex}", on_click=set_topic, args=(ex,), use_container_width=True)

go = st.button("Start research", type="primary", use_container_width=True)

st.markdown(
    '<div class="di-trust"><b>Sourced</b> from the live web &nbsp;|&nbsp; '
    "<b>Reviewed</b> by a critic agent &nbsp;|&nbsp; <b>Yours</b> to download</div>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Run / show results
# --------------------------------------------------------------------------
if go:
    topic = st.session_state.topic.strip()
    if not topic:
        st.warning("Type a topic or pick one of the examples above to begin.")
    else:
        st.divider()
        st.markdown(f"#### Researching: {topic}")
        try:
            state, seconds = run_pipeline_live(topic)
            entry = {"topic": topic, "state": state, "seconds": seconds, "at": datetime.now()}
            st.session_state.history.insert(0, entry)
            st.session_state.current = entry
            render_result(topic, state, seconds, celebrate=True)
        except Exception as e:  # keep the UI friendly if a tool/API call fails
            st.error(
                "Something went wrong while researching. Check your API keys and "
                "internet connection, then try again."
            )
            with st.expander("Technical details"):
                st.exception(e)
elif st.session_state.current:
    st.divider()
    cur = st.session_state.current
    render_result(cur["topic"], cur["state"], cur["seconds"])

# Past research: lets people return to earlier work without re-running it
if len(st.session_state.history) > 1:
    st.divider()
    st.markdown("#### Your earlier research")
    for i, h in enumerate(st.session_state.history[1:], start=1):
        label = f"{h['topic']}  ({h['at'].strftime('%H:%M')})"
        if st.button(label, key=f"hist_{i}"):
            st.session_state.current = h
            st.rerun()