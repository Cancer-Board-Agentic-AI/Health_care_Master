from __future__ import annotations

import html
import json
import os

import httpx
import streamlit as st

API_URL = os.getenv("MEDICAL_API_URL", "http://localhost:8000")

SPECIALISTS = {
    "radiology": ("◉", "Radiology"),
    "pathology": ("⬡", "Pathology & Molecular"),
    "surgery": ("✚", "Surgical Oncology"),
    "radiation": ("✦", "Radiation Oncology"),
    "medical_oncology": ("◆", "Medical Oncology"),
    "supportive": ("●", "Supportive Care"),
}

PLAN_STAGES = (
    ("pretreatment_confirmation", "1", "Pretreatment confirmation", "🔬"),
    ("initial_treatment_decision", "2", "Initial treatment", "🧭"),
    ("breast_and_axillary_surgery", "3", "Breast and axillary surgery", "🏥"),
    ("postoperative_systemic_treatment", "4", "Postoperative systemic treatment", "💊"),
    ("radiation", "5", "Radiation", "☢️"),
    ("supportive_and_access", "6", "Supportive and access planning", "🤝"),
)


def _compact(items: list[str], limit: int = 2) -> str:
    values = [html.escape(str(item)) for item in (items or [])[:limit]]
    return "<br>".join(f"• {item}" for item in values) or "—"


def _agent_card(
    name: str, result: dict | None = None, error: str = "", phase: str = "initial",
) -> str:
    icon, label = SPECIALISTS[name]
    if error:
        status, status_class, body = "Needs review", "error", html.escape(error)
    elif result is None:
        status, status_class, body = "Analysing…", "working", "Reviewing the case in parallel"
    else:
        confidence = float(result.get("confidence", 0) or 0)
        if phase == "deliberating":
            status, status_class = "Reviewing peer opinions…", "working"
        elif phase in {"deliberation", "evidence_review"}:
            label = "Evidence-reviewed" if phase == "evidence_review" else "Deliberated"
            status, status_class = f"{label} · {confidence:.0%}", "complete"
        else:
            status, status_class = f"Initial opinion · {confidence:.0%}", "complete"
        reasoning = html.escape(str(result.get("reasoning", ""))[:500]) or "—"
        body = (
            f'<div class="agent-label">Findings</div>{_compact(result.get("key_findings", []))}'
            f'<div class="agent-label">Recommendation</div>{_compact(result.get("recommended_plan", []))}'
            f'<div class="agent-label">Reasoning summary</div>{reasoning}'
        )
    return f'''<div class="agent-card {status_class}">
        <div class="agent-head"><span class="agent-orb">{icon}</span>
        <span><strong>{label}</strong><br><small>{status}</small></span></div>
        <div class="agent-body">{body}</div>
    </div>'''



def _split_report(markdown: str) -> tuple[str, str]:
    """Return (main report, evidence) so references can use a native expander."""
    heading = "## Evidence & References"
    if heading not in markdown:
        return markdown, ""
    main, tail = markdown.split(heading, 1)
    limitation = "## Limitations & Disclaimer"
    if limitation in tail:
        evidence, remainder = tail.split(limitation, 1)
        main = main.rstrip() + "\n\n" + limitation + remainder
    else:
        evidence = tail
    return main.strip(), evidence.strip()


def _without_case_preamble(markdown: str) -> str:
    """Remove title/case already shown above the live specialist dashboard."""
    consensus = "## Consensus Confidence Index"
    if consensus in markdown:
        return consensus + markdown.split(consensus, 1)[1]
    return markdown


def _without_text_recommendation(markdown: str) -> tuple[str, str]:
    """Split around the Markdown recommendation when a visual plan is available."""
    heading = "## Final Recommendation"
    if heading not in markdown:
        return markdown, ""
    before, recommendation = markdown.split(heading, 1)
    trailing_headings = ("## Recommended Tests / Investigations", "## Limitations & Disclaimer")
    positions = [recommendation.find(item) for item in trailing_headings if item in recommendation]
    if not positions:
        return before.strip(), ""
    position = min(value for value in positions if value >= 0)
    return before.strip(), recommendation[position:].strip()


def _render_board_plan(plan: dict) -> None:
    st.markdown("## Final Recommendation")
    summary = plan.get("case_summary")
    if summary:
        st.info(f"Tumor-board conclusion: {summary}", icon="📋")

    columns = st.columns(2)
    for index, (key, number, title, icon) in enumerate(PLAN_STAGES):
        with columns[index % 2]:
            with st.container(border=True):
                st.markdown(f"### {icon} {number}. {title}")
                items = plan.get(key, []) or ["No recommendation recorded."]
                for item in items:
                    st.markdown(f"- {item}")

    dependencies = plan.get("decision_dependencies", [])
    if dependencies:
        with st.expander("⏳ Decisions pending additional results", expanded=True):
            for item in dependencies:
                st.markdown(f"- {item}")

    disagreements = plan.get("unresolved_disagreements", [])
    if disagreements:
        disagreement_list = "\n".join(f"- {item}" for item in disagreements)
        st.warning(f"Unresolved board disagreements:\n\n{disagreement_list}")


def _render_report(markdown: str, *, omit_case: bool = False, board_plan: dict | None = None) -> None:
    main, evidence = _split_report(markdown)
    if board_plan:
        before, after = _without_text_recommendation(main)
        st.markdown(_without_case_preamble(before) if omit_case else before)
        _render_board_plan(board_plan)
        if after:
            st.markdown(after)
    else:
        st.markdown(_without_case_preamble(main) if omit_case else main)
    if evidence:
        with st.expander("Evidence & References", expanded=False):
            st.markdown(evidence)


st.set_page_config(page_title="Local Medical AI", page_icon="⚕️", layout="wide")
st.title("⚕️ Local Medical AI")
st.caption("Offline collaborative clinical decision support. Not a substitute for professional care.")
st.markdown("""
<style>
.agent-card {border:1px solid #d9e2ec; border-radius:16px; padding:14px; height:270px;
             box-sizing:border-box; overflow:hidden; margin-bottom:14px;
             background:rgba(250,252,255,.75);}
.agent-card.working {border-color:#8bb8e8; box-shadow:0 0 0 2px rgba(60,130,200,.08);}
.agent-card.complete {border-color:#65a984;}
.agent-card.error {border-color:#d97777;}
.agent-head {display:flex; align-items:center; gap:10px; margin-bottom:10px;}
.agent-orb {display:inline-flex; width:42px; height:42px; border-radius:50%; align-items:center;
            justify-content:center; background:#e7f1fb; color:#185a91; font-size:20px; flex:none;}
.agent-card.working .agent-orb {animation:pulse 1.5s infinite;}
.agent-card.complete .agent-orb {background:#e4f4ea; color:#23764a;}
.agent-label {font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.04em;
              margin-top:8px; color:#52606d;}
.agent-body {font-size:.84rem; line-height:1.35; height:198px; overflow-y:auto;
             overflow-x:hidden; overflow-wrap:anywhere; padding-right:6px; scrollbar-gutter:stable;}
.agent-body::-webkit-scrollbar {width:6px;}
.agent-body::-webkit-scrollbar-thumb {background:#b8c5d1; border-radius:6px;}
@keyframes pulse {0%,100%{opacity:1} 50%{opacity:.45}}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Session")
    model = st.selectbox("Unified medical model", ["medgemma:27b", "medgemma:4b"])
    vision_model = model
    st.caption("The selected MedGemma model is used for both clinical reasoning and image analysis.")
    upload = st.file_uploader("Add local PDF knowledge", type=["pdf"])
    if upload and st.button("Ingest PDF"):
        response = httpx.post(f"{API_URL}/documents", files={"file": (upload.name, upload.getvalue(), "application/pdf")}, timeout=180)
        response.raise_for_status(); st.success(f"Stored {response.json()['chunks']} chunks")

    st.caption("No local PDF? Use a bundled sample instead.")
    try:
        samples = httpx.get(f"{API_URL}/documents/samples", timeout=10).json()
    except httpx.HTTPError:
        samples = []
    if samples:
        sample_choice = st.selectbox("Select sample PDF", samples)
        if st.button("Ingest sample"):
            response = httpx.post(f"{API_URL}/documents/samples/{sample_choice}", timeout=180)
            response.raise_for_status(); st.success(f"Stored {response.json()['chunks']} chunks")
    else:
        st.caption("No sample PDFs are bundled on the server.")
    clinical_image = st.file_uploader("Optional medical image", type=["jpg", "jpeg", "png", "webp"])
    if clinical_image:
        st.image(clinical_image, caption="Image sent only to the local vision model")

if "messages" not in st.session_state: st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message["content"].startswith("# Virtual Tumor Board Report"):
            _render_report(message["content"], board_plan=message.get("board_plan"))
        else:
            st.markdown(message["content"])

question = st.chat_input("Describe symptoms, medications, labs, or ask about local guidance")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"): st.markdown(question)
    with st.chat_message("assistant"):
        st.markdown("# Virtual Tumor Board Report")
        st.markdown("## Case")
        st.markdown(question)
        st.markdown("#### Specialty assessments · live")
        columns = st.columns(3)
        agent_slots = {
            name: columns[index % 3].empty()
            for index, name in enumerate(SPECIALISTS)
        }
        for name, slot in agent_slots.items():
            slot.markdown(_agent_card(name), unsafe_allow_html=True)

        status_placeholder = st.empty()
        status_placeholder.caption("Six specialists are reviewing the case with model-safe concurrent scheduling…")
        final_answer = ""
        final_state: dict = {}
        try:
            if clinical_image:
                stream_request = dict(
                    url=f"{API_URL}/ask-with-image/stream",
                    data={"question": question.strip(), "model": model, "vision_model": vision_model},
                    files={"image": (clinical_image.name, clinical_image.getvalue(), clinical_image.type)},
                )
            else:
                stream_request = dict(url=f"{API_URL}/ask/stream", json={"question": question.strip(), "model": model})
            initial_results: dict[str, dict] = {}
            deliberated: set[str] = set()
            with httpx.stream("POST", timeout=900, **stream_request) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[len("data: "):])
                    if event["type"] == "agent" and event["name"] in agent_slots:
                        name = event["name"]
                        phase = event.get("phase", "initial")
                        result = event.get("result") or {}
                        agent_slots[name].markdown(
                            _agent_card(name, result, phase=phase),
                            unsafe_allow_html=True,
                        )
                        if phase == "initial":
                            initial_results[name] = result
                            status_placeholder.caption(
                                f"Initial opinions: {len(initial_results)}/6 complete"
                            )
                            if len(initial_results) == len(SPECIALISTS):
                                for peer_name, peer_result in initial_results.items():
                                    agent_slots[peer_name].markdown(
                                        _agent_card(peer_name, peer_result, phase="deliberating"),
                                        unsafe_allow_html=True,
                                    )
                                status_placeholder.caption(
                                    "All initial opinions complete · cross-specialty deliberation in progress…"
                                )
                        else:
                            deliberated.add(name)
                            status_placeholder.caption(
                                f"Deliberated opinions: {len(deliberated)}/6 complete"
                            )
                    elif event["type"] == "error":
                        status_placeholder.error(event["message"])
                    elif event["type"] == "done":
                        final_answer = event["answer"]
                        final_state = event["state"]
        except httpx.HTTPStatusError as exc:
            st.error(f"API rejected the request ({exc.response.status_code}).")
            st.stop()
        except httpx.HTTPError as exc:
            st.error(f"Could not reach the API at {API_URL}: {exc}")
            st.stop()

        status_placeholder.success("Tumor-board synthesis complete")
        if final_answer:
            _render_report(
                final_answer,
                omit_case=True,
                board_plan=final_state.get("board_plan"),
            )
        else:
            st.markdown("No answer was produced.")
    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer,
        "board_plan": final_state.get("board_plan"),
    })
