import pandas as pd
import numpy as np


def load_data(data_path="data"):
    tickets = pd.read_csv(
        f"{data_path}/tickets.csv",
        low_memory=False
    )

    agents = pd.read_csv(
        f"{data_path}/agents.csv",
        low_memory=False
    )

    customers = pd.read_csv(
        f"{data_path}/customers.csv",
        low_memory=False
    )

    orders = pd.read_csv(
        f"{data_path}/orders.csv",
        low_memory=False
    )

    products = pd.read_csv(
        f"{data_path}/products.csv",
        low_memory=False
    )

    return tickets, agents, customers, orders, products


def clean_tickets(tickets):

    tickets = tickets.copy()

    # Remove accidental whitespace from column names
    tickets.columns = tickets.columns.str.strip()

    # Convert timestamps
    date_columns = [
        "created_at",
        "first_response_at",
        "resolved_at"
    ]

    for col in date_columns:
        tickets[col] = pd.to_datetime(
            tickets[col],
            dayfirst=True,
            errors="coerce"
        )

    # Normalize text fields
    text_columns = [
        "status",
        "channel",
        "category",
        "priority",
        "assigned_team",
        "agent_id",
        "source_system",
        "replacement_issued"
    ]

    for col in text_columns:
        if col in tickets.columns:
            tickets[col] = (
                tickets[col]
                .astype("string")
                .str.strip()
            )

    # Numeric fields
    tickets["csat_score"] = pd.to_numeric(
        tickets["csat_score"],
        errors="coerce"
    )

    tickets["refund_amount_inr"] = pd.to_numeric(
        tickets["refund_amount_inr"],
        errors="coerce"
    ).fillna(0)

    tickets["transfers"] = pd.to_numeric(
        tickets["transfers"],
        errors="coerce"
    ).fillna(0)

    return tickets


def clean_orders(orders):

    orders = orders.copy()

    orders.columns = orders.columns.str.strip()

    # Remove exact duplicate order records
    orders = orders.drop_duplicates()

    # One order should map to one record
    if "order_id" in orders.columns:
        orders = orders.drop_duplicates(
            subset=["order_id"],
            keep="first"
        )

    orders["order_date"] = pd.to_datetime(
        orders["order_date"],
        dayfirst=True,
        errors="coerce"
    )

    orders["order_value_inr"] = pd.to_numeric(
        orders["order_value_inr"],
        errors="coerce"
    )

    return orders


def clean_agents(agents):

    agents = agents.copy()

    agents.columns = agents.columns.str.strip()

    agents["from_date"] = pd.to_datetime(
        agents["from_date"],
        dayfirst=True,
        errors="coerce"
    )

    agents["to_date"] = pd.to_datetime(
        agents["to_date"],
        dayfirst=True,
        errors="coerce"
    )

    return agents


def clean_products(products):

    products = products.copy()

    products.columns = products.columns.str.strip()

    products["unit_cost_inr"] = pd.to_numeric(
        products["unit_cost_inr"],
        errors="coerce"
    )

    products["retail_price_inr"] = pd.to_numeric(
        products["retail_price_inr"],
        errors="coerce"
    )

    return products