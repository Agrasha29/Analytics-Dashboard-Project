import pandas as pd
import numpy as np


# ---------------------------------------------------------
# HANDLE TIME
# ---------------------------------------------------------
def calculate_handle_time(tickets):

    df = tickets.copy()

    df["created_at"] = pd.to_datetime(
        df["created_at"],
        errors="coerce"
    )

    df["resolved_at"] = pd.to_datetime(
        df["resolved_at"],
        errors="coerce"
    )

    df["resolution_minutes"] = np.nan

    valid_created = df["created_at"].notna()
    valid_resolved = df["resolved_at"].notna()

    valid_order = (
        df["resolved_at"] >= df["created_at"]
    )

    completed = df["status"].isin(
        ["resolved", "closed"]
    )

    valid_mask = (
        valid_created
        & valid_resolved
        & valid_order
        & completed
    )

    df.loc[valid_mask, "resolution_minutes"] = (
        (
            df.loc[valid_mask, "resolved_at"]
            - df.loc[valid_mask, "created_at"]
        ).dt.total_seconds()
        / 60
    )

    return df


# ---------------------------------------------------------
# CSAT
# ---------------------------------------------------------
def calculate_csat(tickets):

    df = tickets.copy()

    # Blank CSAT = no response, not zero
    df["csat_score"] = pd.to_numeric(
        df["csat_score"],
        errors="coerce"
    )

    valid = df["csat_score"].notna()

    result = (
        df[valid]
        .groupby("agent_id")
        .agg(
            csat=("csat_score", "mean"),
            csat_responses=("csat_score", "count")
        )
        .reset_index()
    )

    return result


# ---------------------------------------------------------
# AGENT METRICS
# ---------------------------------------------------------
def agent_metrics(tickets, agents):

    df = calculate_handle_time(tickets)

    completed = df[
        df["status"].isin(["resolved", "closed"])
    ]

    metrics = (
        completed
        .groupby("agent_id")
        .agg(
            tickets_handled=("ticket_id", "count"),
            avg_resolution_minutes=(
                "resolution_minutes",
                "mean"
            ),
            median_resolution_minutes=(
                "resolution_minutes",
                "median"
            ),
            transfers=("transfers", "sum")
        )
        .reset_index()
    )

    # Add CSAT
    csat = calculate_csat(df)

    metrics = metrics.merge(
        csat,
        on="agent_id",
        how="left"
    )

    # Add agent information
    metrics = metrics.merge(
        agents[
            [
                "agent_id",
                "name",
                "site",
                "team",
                "shift",
                "tier"
            ]
        ],
        on="agent_id",
        how="left"
    )

    return metrics


# ---------------------------------------------------------
# TEAM METRICS
# ---------------------------------------------------------
def team_metrics(tickets):

    df = tickets.copy()

    df["csat_score"] = pd.to_numeric(
        df["csat_score"],
        errors="coerce"
    )

    result = (
        df.groupby("assigned_team")
        .agg(
            tickets=("ticket_id", "count"),
            avg_csat=("csat_score", "mean"),
            transfers=("transfers", "sum"),
            replacements=(
                "replacement_issued",
                lambda x: (x == "Y").sum()
            )
        )
        .reset_index()
    )

    return result


# ---------------------------------------------------------
# CHANNEL METRICS
# ---------------------------------------------------------
def channel_metrics(tickets):

    df = tickets.copy()

    df["csat_score"] = pd.to_numeric(
        df["csat_score"],
        errors="coerce"
    )

    result = (
        df.groupby("channel")
        .agg(
            tickets=("ticket_id", "count"),
            avg_csat=("csat_score", "mean")
        )
        .reset_index()
    )

    return result


# ---------------------------------------------------------
# SLA
# ---------------------------------------------------------
SLA_MINUTES = {
    "chat": 15,
    "voice": 120,
    "social": 240,
    "email": 480
}


def calculate_sla(tickets):

    df = tickets.copy()

    df["created_at"] = pd.to_datetime(
        df["created_at"],
        errors="coerce"
    )

    df["first_response_at"] = pd.to_datetime(
        df["first_response_at"],
        errors="coerce"
    )

    df["first_response_minutes"] = (
        df["first_response_at"]
        - df["created_at"]
    ).dt.total_seconds() / 60

    df["channel_clean"] = (
        df["channel"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    df["sla_target_minutes"] = (
        df["channel_clean"].map(SLA_MINUTES)
    )

    df["sla_breach"] = (
        df["first_response_minutes"].notna()
        & df["sla_target_minutes"].notna()
        & (
            df["first_response_minutes"]
            > df["sla_target_minutes"]
        )
    )

    return df


# ---------------------------------------------------------
# REPLACEMENT COST
# ---------------------------------------------------------
def calculate_replacement_cost(tickets, products):

    df = tickets.copy()

    df = df.merge(
        products[
            [
                "sku",
                "unit_cost_inr"
            ]
        ],
        left_on="product_sku",
        right_on="sku",
        how="left"
    )

    # Policy:
    # Replacement cost = unit cost + ₹340 logistics
    df["replacement_cost_inr"] = np.where(
        df["replacement_issued"] == "Y",
        df["unit_cost_inr"] + 340,
        0
    )

    return df


# ---------------------------------------------------------
# TRANSFER COST
# ---------------------------------------------------------
def calculate_transfer_cost(tickets):

    df = tickets.copy()

    df["transfers"] = pd.to_numeric(
        df["transfers"],
        errors="coerce"
    ).fillna(0)

    # Policy: ₹305 per internal transfer
    df["transfer_cost_inr"] = (
        df["transfers"] * 305
    )

    return df


# ---------------------------------------------------------
# SLA CREDIT COST
# ---------------------------------------------------------
def calculate_sla_cost(tickets):

    df = calculate_sla(tickets)

    # Policy: ₹350 for every SLA breach
    df["sla_credit_cost_inr"] = np.where(
        df["sla_breach"],
        350,
        0
    )

    return df


# ---------------------------------------------------------
# BUSINESS IMPACT
# ---------------------------------------------------------
def business_impact(tickets, products):

    df = calculate_replacement_cost(
        tickets,
        products
    )

    df = calculate_transfer_cost(df)

    df = calculate_sla_cost(df)

    replacement_cost = df[
        "replacement_cost_inr"
    ].sum()

    transfer_cost = df[
        "transfer_cost_inr"
    ].sum()

    sla_credit_cost = df[
        "sla_credit_cost_inr"
    ].sum()

    total_cost = (
        replacement_cost
        + transfer_cost
        + sla_credit_cost
    )

    return {
        "replacement_cost": replacement_cost,
        "transfer_cost": transfer_cost,
        "sla_credit_cost": sla_credit_cost,
        "total_operational_cost": total_cost
    }


# ---------------------------------------------------------
# BOTTOM TEN FLAGS
# ---------------------------------------------------------
def bottom_ten_flags(tickets, agents):

    metrics = agent_metrics(
        tickets,
        agents
    )

    # Tier 2 should not be compared with Tier 1
    tier1 = metrics[
        metrics["tier"].fillna("Tier1") != "Tier2"
    ].copy()

    # Require at least 3 CSAT responses
    tier1["csat_flag"] = (
        tier1["csat"].notna()
        & (tier1["csat_responses"] >= 3)
    )

    # Lowest CSAT candidates
    csat_candidates = tier1[
        tier1["csat_flag"]
    ].sort_values(
        "csat",
        ascending=True
    )

    # Highest resolution-time candidates
    time_candidates = tier1[
        tier1["avg_resolution_minutes"].notna()
    ].sort_values(
        "avg_resolution_minutes",
        ascending=False
    )

    tier1["coaching_flag"] = ""

    # Flag low CSAT
    if len(csat_candidates) > 0:

        worst_csat = set(
            csat_candidates
            .head(10)["agent_id"]
        )

        tier1.loc[
            tier1["agent_id"].isin(worst_csat),
            "coaching_flag"
        ] += "Low CSAT sample; "

    # Flag high resolution time
    if len(time_candidates) > 0:

        slow_agents = set(
            time_candidates
            .head(10)["agent_id"]
        )

        tier1.loc[
            tier1["agent_id"].isin(slow_agents),
            "coaching_flag"
        ] += "High resolution time; "

    # Put flagged agents first
    tier1["has_flag"] = (
        tier1["coaching_flag"].str.len() > 0
    )

    return tier1.sort_values(
        ["has_flag", "csat", "avg_resolution_minutes"],
        ascending=[False, True, False],
        na_position="last"
    ).drop(
        columns=["has_flag"],
        errors="ignore"
    )