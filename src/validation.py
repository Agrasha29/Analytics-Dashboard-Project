import pandas as pd
import numpy as np


def validate_data(tickets, agents, products):

    results = {}

    results["ticket_rows"] = len(tickets)

    results["unique_ticket_ids"] = (
        tickets["ticket_id"].nunique()
    )

    results["duplicate_ticket_ids"] = (
        len(tickets)
        - tickets["ticket_id"].nunique()
    )

    results["missing_agent_ids"] = (
        tickets["agent_id"].isna().sum()
    )

    results["unknown_agent_ids"] = (
        ~tickets["agent_id"].isin(
            agents["agent_id"]
        )
    ).sum()

    results["unknown_product_skus"] = (
        ~tickets["product_sku"].isin(
            products["sku"]
        )
    ).sum()

    results["missing_csat"] = (
        tickets["csat_score"].isna().sum()
    )

    results["csat_response_rate"] = (
        tickets["csat_score"].notna().mean()
        * 100
    )

    results["missing_created_at"] = (
        tickets["created_at"].isna().sum()
    )

    results["missing_first_response"] = (
        tickets["first_response_at"].isna().sum()
    )

    results["missing_resolved_at"] = (
        tickets["resolved_at"].isna().sum()
    )

    results["ivr_tickets"] = (
        tickets["customer_message"]
        .fillna("")
        .str.contains(
            r"\[IVR transcript\]",
            case=False,
            regex=True
        )
        .sum()
    )

    return results


def data_quality_table(tickets):

    rows = []

    for column in tickets.columns:

        missing = tickets[column].isna().sum()

        rows.append({
            "column": column,
            "missing_values": missing,
            "missing_percentage":
                round(
                    missing / len(tickets) * 100,
                    2
                )
        })

    return pd.DataFrame(rows)