import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_loader import (
    load_data,
    clean_tickets,
    clean_orders,
    clean_agents,
    clean_products
)

from src.metrics import (
    agent_metrics,
    team_metrics,
    channel_metrics,
    calculate_sla,
    calculate_replacement_cost,
    business_impact,
    bottom_ten_flags
)

from src.validation import (
    validate_data,
    data_quality_table
)

from src.ai_insights import (
    train_category_model,
    generate_ai_insights
)


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Vireo Audio Support Analytics",
    page_icon="🎧",
    layout="wide"
)


# ---------------------------------------------------------
# CUSTOM UI STYLING
# ---------------------------------------------------------

st.markdown("""
<style>

/* -------------------------------------------------------
   SELECTED SIDEBAR FILTER TAGS
   Yellow background + black text
   ------------------------------------------------------- */

div[data-baseweb="tag"] {
    background-color: #FFD700 !important;
    color: #000000 !important;
    border: 1px solid #000000 !important;
    border-radius: 6px !important;
}

/* Text inside selected tags */
div[data-baseweb="tag"] span {
    color: #000000 !important;
}

/* X / remove icon */
div[data-baseweb="tag"] svg {
    fill: #000000 !important;
    color: #000000 !important;
}

/* Hover effect */
div[data-baseweb="tag"]:hover {
    background-color: #FFC400 !important;
}


/* -------------------------------------------------------
   SIDEBAR MULTISELECT BOX
   ------------------------------------------------------- */

section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    border-color: #FFD700 !important;
}


/* -------------------------------------------------------
   SIDEBAR HEADERS
   ------------------------------------------------------- */

section[data-testid="stSidebar"] h2 {
    font-weight: 700;
}


/* -------------------------------------------------------
   GENERAL APP SPACING
   ------------------------------------------------------- */

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title(
    "🎧 Vireo Audio — Support Intelligence Dashboard"
)

st.caption(
    "AI-assisted support analytics for CSAT, resolution time, "
    "SLA performance and operational cost."
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

try:

    (
        tickets,
        agents,
        customers,
        orders,
        products
    ) = load_data()

    tickets = clean_tickets(tickets)
    agents = clean_agents(agents)
    orders = clean_orders(orders)
    products = clean_products(products)

except Exception as e:

    st.error(
        f"Could not load data: {e}"
    )

    st.stop()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("🔎 Filters")


channels = sorted(
    tickets["channel"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
)


selected_channels = st.sidebar.multiselect(
    "Channel",
    channels,
    default=channels
)


teams = sorted(
    tickets["assigned_team"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
)


selected_teams = st.sidebar.multiselect(
    "Team",
    teams,
    default=teams
)


# ---------------------------------------------------------
# FILTER DATA
# ---------------------------------------------------------

filtered = tickets[
    tickets["channel"].isin(
        selected_channels
    )
    &
    tickets["assigned_team"].isin(
        selected_teams
    )
].copy()


# ---------------------------------------------------------
# KPI CARDS
# ---------------------------------------------------------

total_tickets = len(filtered)


csat_responses = (
    filtered["csat_score"]
    .notna()
    .sum()
)


average_csat = (
    filtered["csat_score"]
    .mean()
)


replacement_count = (
    filtered["replacement_issued"]
    .eq("Y")
    .sum()
)


transfer_count = (
    filtered["transfers"]
    .sum()
)


sla_data = calculate_sla(
    filtered
)


sla_breaches = (
    sla_data["sla_breach"]
    .sum()
)


col1, col2, col3, col4, col5 = st.columns(5)


col1.metric(
    "🎫 Tickets",
    f"{total_tickets:,}"
)


col2.metric(
    "⭐ Avg CSAT",
    f"{average_csat:.2f}"
    if pd.notna(average_csat)
    else "N/A"
)


col3.metric(
    "💬 CSAT Responses",
    f"{csat_responses:,}"
)


col4.metric(
    "🔄 Replacements",
    f"{replacement_count:,}"
)


col5.metric(
    "⚠️ SLA Breaches",
    f"{sla_breaches:,}"
)


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------

(
    overview_tab,
    agents_tab,
    operations_tab,
    ai_tab,
    validation_tab
) = st.tabs([
    "📊 Overview",
    "👥 Agent Performance",
    "💰 Operations",
    "🤖 AI Insights",
    "🧪 Validation"
])


# =========================================================
# OVERVIEW
# =========================================================

with overview_tab:

    st.subheader(
        "CSAT by Channel"
    )


    channel_df = channel_metrics(
        filtered
    )


    fig = px.bar(
        channel_df,
        x="channel",
        y="avg_csat",
        text_auto=".2f",
        title="Average CSAT by Channel"
    )


    fig.update_yaxes(
        range=[1, 5]
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.subheader(
        "Ticket Volume by Category"
    )


    category_df = (
        filtered["category"]
        .value_counts()
        .reset_index()
    )


    category_df.columns = [
        "category",
        "tickets"
    ]


    fig2 = px.bar(
        category_df,
        x="category",
        y="tickets",
        title="Support Demand by Category"
    )


    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# =========================================================
# AGENTS
# =========================================================

with agents_tab:

    st.subheader(
        "👥 Agent Performance"
    )


    metrics = agent_metrics(
        filtered,
        agents
    )


    st.info(
        "Tier 2 agents are shown for visibility, "
        "but should not be compared with Tier 1 on "
        "ticket-volume metrics under the support policy."
    )


    display_columns = [
        "agent_id",
        "name",
        "team",
        "tier",
        "tickets_handled",
        "csat",
        "csat_responses",
        "avg_resolution_minutes",
        "transfers"
    ]


    st.dataframe(
        metrics[display_columns].round(2),
        use_container_width=True,
        hide_index=True
    )


    st.subheader(
        "🚩 Coaching Flags"
    )


    flags = bottom_ten_flags(
        filtered,
        agents
    )


    st.dataframe(
        flags[
            [
                "agent_id",
                "name",
                "team",
                "tier",
                "tickets_handled",
                "csat",
                "csat_responses",
                "avg_resolution_minutes",
                "coaching_flag"
            ]
        ].round(2),
        use_container_width=True,
        hide_index=True
    )


    st.caption(
        "Flags are screening signals, not proof of individual "
        "performance problems. Low sample sizes and queue mix "
        "should be reviewed before training decisions."
    )


# =========================================================
# OPERATIONS
# =========================================================

with operations_tab:

    st.subheader(
        "💰 Operational Cost"
    )


    impact = business_impact(
        filtered,
        products
    )


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "Replacement Cost",
        f"₹{impact['replacement_cost']:,.0f}"
    )


    c2.metric(
        "Transfer Cost",
        f"₹{impact['transfer_cost']:,.0f}"
    )


    c3.metric(
        "SLA Credit Cost",
        f"₹{impact['sla_credit_cost']:,.0f}"
    )


    c4.metric(
        "Total Operational Cost",
        f"₹{impact['total_operational_cost']:,.0f}"
    )


    st.subheader(
        "SLA Performance"
    )


    sla_summary = (
        sla_data
        .groupby("channel")
        .agg(
            tickets=("ticket_id", "count"),
            breaches=("sla_breach", "sum"),
            avg_response_minutes=(
                "first_response_minutes",
                "mean"
            )
        )
        .reset_index()
    )


    if len(sla_summary) > 0:

        sla_summary["breach_rate"] = (
            sla_summary["breaches"]
            / sla_summary["tickets"]
            * 100
        )

    else:

        sla_summary["breach_rate"] = []


    st.dataframe(
        sla_summary.round(2),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# AI
# =========================================================

with ai_tab:

    st.subheader(
        "🤖 AI-Assisted Support Insights"
    )


    st.write(
        "The prototype uses a lightweight local text "
        "classification model to identify ticket categories "
        "without sending every ticket to a paid external LLM."
    )


    model = train_category_model(
        filtered
    )


    if model:

        st.success(
            "Local category classifier trained successfully."
        )

    else:

        st.warning(
            "Not enough labelled data to train the classifier."
        )


    st.subheader(
        "Automated Findings"
    )


    insights = generate_ai_insights(
        filtered
    )


    if insights:

        for insight in insights:

            st.write(
                f"• {insight}"
            )

    else:

        st.info(
            "No automated findings available for the current filters."
        )


    st.subheader(
        "Try a Ticket"
    )


    message = st.text_area(
        "Paste a customer message",
        placeholder=(
            "Example: My left earbud is not charging..."
        )
    )


    if st.button(
        "Classify Ticket"
    ):

        if model and message.strip():

            prediction = model.predict(
                [message]
            )[0]


            st.success(
                f"Predicted category: **{prediction}**"
            )

        else:

            st.warning(
                "Enter a message first."
            )


# =========================================================
# VALIDATION
# =========================================================

with validation_tab:

    st.subheader(
        "🧪 Data & Model Validation"
    )


    validation = validate_data(
        filtered,
        agents,
        products
    )


    validation_df = pd.DataFrame(
        list(validation.items()),
        columns=[
            "Metric",
            "Value"
        ]
    )


    st.dataframe(
        validation_df,
        use_container_width=True,
        hide_index=True
    )


    st.subheader(
        "Missing Values"
    )


    quality = data_quality_table(
        filtered
    )


    st.dataframe(
        quality,
        use_container_width=True,
        hide_index=True
    )


    st.subheader(
        "IVR Transcript Check"
    )


    ivr_count = validation[
        "ivr_tickets"
    ]


    st.metric(
        "Tickets containing IVR transcript marker",
        ivr_count
    )


    st.caption(
        "IVR transcripts are separately identified because "
        "the assignment notes that approximately 40 voice "
        "tickets contain junk IVR content."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()


st.caption(
    "Vireo Audio Support Intelligence • "
    "AI-assisted prototype • Built for decision support"
)