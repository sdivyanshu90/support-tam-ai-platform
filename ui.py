"""Thin Streamlit UI for support engineers and TAMs.

    streamlit run ui.py

Two tabs matching the two workflows. Optimised for a non-technical user getting
an answer fast: every result shows what it was grounded in, and anything the
system is unsure about is called out rather than hidden.
"""

from __future__ import annotations

import streamlit as st

from app.config import get_settings
from app.errors import AppError
from app.models import TicketInput

st.set_page_config(page_title="Support & TAM AI Platform", page_icon="🎧", layout="wide")

SEVERITY_COLOUR = {"High": "🔴", "Medium": "🟠", "Low": "🟡"}
URGENCY_COLOUR = {"P1": "🔴", "P2": "🟠", "P3": "🟡", "P4": "🟢"}

EXAMPLE_TICKET = (
    "Production pipeline down since 06:00 — ERR_CONNECTION_TIMEOUT\n\n"
    "Our DataBridge Pro Connectors pipeline has been failing since 06:00 this morning.\n"
    "Error: ERR_CONNECTION_TIMEOUT after 30s\n\n"
    "All 47 engineers on the data team are blocked and our nightly reporting did not run.\n"
    "Environment: Production. Version: 3.1.2."
)


@st.cache_resource(show_spinner=False)
def _services():
    from app.data.repository import SupportRepository
    from app.services.account_health import AccountHealthService
    from app.services.triage import TriageService

    repo = SupportRepository()
    return TriageService(), AccountHealthService(repository=repo), repo


@st.cache_resource(show_spinner=False)
def _repository():
    from app.data.repository import SupportRepository

    return SupportRepository()


settings = get_settings()

st.title("🎧 Support & TAM AI Platform")
st.caption(
    "Ticket triage and account briefing, grounded in the local knowledge base. "
    "All data is synthetic."
)

with st.sidebar:
    st.subheader("Status")
    if settings.llm_available:
        st.success(f"Model configured · {settings.provider.name}")
    else:
        st.error(f"{settings.provider.api_key_env} not set")
        st.caption("Copy `.env.example` to `.env` and add a key, then restart.")
    st.write(f"**Model** `{settings.model_spec}`")
    st.write(f"**Structured output** `{settings.profile.structured}`")
    st.write(f"**Sampling** temp={settings.temperature}, seed={settings.seed}")
    st.write(f"**Window** {settings.account_window_days} days")
    st.caption(
        "Responses are cached by content hash, so repeating a request is instant "
        "and free."
    )

triage_tab, brief_tab = st.tabs(["🎫  Ticket Triage", "📊  Account Brief"])

# --------------------------------------------------------------------------- #
# Tab 1 — triage
# --------------------------------------------------------------------------- #

with triage_tab:
    st.subheader("Triage an incoming ticket")

    with st.form("triage_form"):
        subject = st.text_input("Subject", value=EXAMPLE_TICKET.split("\n")[0])
        body = st.text_area(
            "Ticket body", value=EXAMPLE_TICKET.split("\n\n", 1)[1], height=200
        )
        col_a, col_b = st.columns([1, 3])
        with col_a:
            stream_reply = st.checkbox("Stream the reply", value=True)
        submitted = st.form_submit_button("Triage ticket", type="primary")

    if submitted:
        if not settings.llm_available:
            st.error(
                    f"No API key configured — set `{settings.provider.api_key_env}` "
                    "in `.env` first."
                )
        elif not (subject.strip() or body.strip()):
            st.warning("Enter a subject or a body before triaging.")
        else:
            try:
                triage_service, _, _ = _services()
                ticket = TicketInput(subject=subject, body=body)
                with st.spinner("Retrieving knowledge base and classifying…"):
                    result = triage_service.triage(ticket)

                top = st.columns(4)
                top[0].metric(
                    "Urgency", f"{URGENCY_COLOUR.get(result.urgency.value, '')} {result.urgency.value}"
                )
                top[1].metric("Product", result.product)
                top[2].metric("Category", result.issue_category)
                top[3].metric("Confidence", f"{result.confidence:.0%}")

                if result.needs_human_review:
                    st.warning(
                        "**Human review required** — this ticket is P1, low-confidence, "
                        "unrecognised, or contained embedded instructions."
                    )

                st.markdown(f"**Route to:** {result.recommended_team}")
                st.caption(result.routing_rationale)

                st.markdown("#### Reasoning")
                st.write(result.reasoning)

                st.markdown("#### Known issue")
                if result.known_issue.matched:
                    st.success(
                        f"Matches **{result.known_issue.issue_name}** "
                        f"(confidence {result.known_issue.confidence:.0%})"
                    )
                    st.markdown(f"Source: `{result.known_issue.kb_document}`")
                    st.caption(result.known_issue.kb_heading or "")
                    st.info(f"“{result.known_issue.evidence}”")
                else:
                    st.info("No known issue matched.")
                    if result.known_issue.rejection_reason:
                        st.caption(f"Why: {result.known_issue.rejection_reason}")

                st.markdown("#### Draft first response")
                if stream_reply:
                    st.write_stream(
                        triage_service.stream_draft_response(ticket, result)
                    )
                else:
                    st.write(result.draft_response)

                with st.expander("Retrieved knowledge-base passages"):
                    for hit in result.retrieved:
                        st.markdown(
                            f"**{hit.normalised_score:.2f}** · `{hit.document}` — {hit.heading}"
                        )
                        st.caption(" ".join(hit.text.split())[:400] + "…")

                st.caption(
                    f"{result.model} · {result.prompt_version} · {result.latency_ms} ms · "
                    f"{result.request_id}"
                )
            except AppError as exc:
                st.error(f"{exc.message}")
                if exc.detail:
                    st.caption(exc.detail)

# --------------------------------------------------------------------------- #
# Tab 2 — account brief
# --------------------------------------------------------------------------- #

with brief_tab:
    st.subheader("Generate a TAM account brief")

    try:
        repo = _repository()
        options = [
            f"{a['account_id']} — {a['company']} ({a.get('health_status', '?')})"
            for a in repo.accounts
        ]
    except AppError as exc:
        options = []
        st.error(exc.message)

    if options:
        choice = st.selectbox("Account", options, index=0)
        account_id = choice.split(" — ")[0]
        if st.button("Generate brief", type="primary"):
            if not settings.llm_available:
                st.error(
                    f"No API key configured — set `{settings.provider.api_key_env}` "
                    "in `.env` first."
                )
            else:
                try:
                    _, account_service, _ = _services()
                    with st.spinner("Reading tickets, verifying evidence, synthesising…"):
                        brief = account_service.build_brief(account_id)

                    st.markdown(f"### {brief.company} ({brief.account_id})")
                    st.caption(
                        f"TAM {brief.tam} · as of {brief.as_of[:10]} · "
                        f"last {brief.window_days} days"
                    )
                    if brief.degraded:
                        st.warning(f"Degraded brief — {brief.degraded_reason}")

                    m = brief.metrics
                    cols = st.columns(5)
                    cols[0].metric("Health", m.health_status)
                    cols[1].metric("ARR", f"${m.arr_usd:,}")
                    cols[2].metric("Tickets (90d)", m.tickets_in_window)
                    cols[3].metric("Seat use", f"{m.seat_utilisation:.0%}")
                    cols[4].metric(
                        "Renewal",
                        f"{m.days_to_renewal}d" if m.days_to_renewal is not None else "—",
                    )

                    st.markdown("#### 1. Executive summary")
                    st.write(brief.executive_summary)

                    st.markdown(f"#### 2. Open risks & flagged issues ({len(brief.open_risks)})")
                    if not brief.open_risks:
                        st.success("No evidence-backed risks in this window.")
                    for risk in brief.open_risks:
                        icon = SEVERITY_COLOUR.get(risk.severity.value, "•")
                        with st.container(border=True):
                            st.markdown(
                                f"{icon} **{risk.risk_type}** · `{risk.ticket_id}` · _{risk.source}_"
                            )
                            st.write(risk.rationale)
                            st.info(f"“{risk.evidence_quote}”")

                    st.markdown("#### 3. Recommended talking points")
                    for i, point in enumerate(brief.recommended_talking_points, 1):
                        st.markdown(f"{i}. {point}")

                    with st.expander("Evidence and metrics detail"):
                        st.write(
                            f"Quotes verified: **{brief.quotes_verified}** · "
                            f"rejected as not verbatim: **{brief.quotes_rejected}**"
                        )
                        st.write(f"Tickets analysed: {', '.join(brief.tickets_considered)}")
                        st.json(m.model_dump())

                    st.caption(
                        f"{brief.model} · {brief.prompt_version} · {brief.latency_ms} ms · "
                        f"{brief.request_id}"
                    )
                except AppError as exc:
                    st.error(exc.message)
                    if exc.detail:
                        st.caption(exc.detail)
