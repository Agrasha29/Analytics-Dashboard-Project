import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def train_category_model(tickets):

    df = tickets.copy()

    df["customer_message"] = (
        df["customer_message"]
        .fillna("")
        .astype(str)
    )

    # Need enough examples per category
    counts = df["category"].value_counts()

    valid_categories = counts[
        counts >= 2
    ].index

    df = df[
        df["category"].isin(valid_categories)
    ]

    if len(df) < 10:
        return None

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                max_features=5000
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ])

    model.fit(
        df["customer_message"],
        df["category"]
    )

    return model


def predict_categories(model, messages):

    if model is None:
        return []

    return model.predict(
        messages.fillna("").astype(str)
    )


def generate_ai_insights(tickets):

    insights = []

    # High-volume categories
    category_counts = (
        tickets["category"]
        .value_counts()
    )

    if len(category_counts) > 0:

        top_category = category_counts.index[0]

        insights.append(
            f"Highest ticket volume category: "
            f"{top_category} "
            f"({category_counts.iloc[0]} tickets)."
        )

    # Replacement concentration
    replacements = tickets[
        tickets["replacement_issued"] == "Y"
    ]

    if len(replacements) > 0:

        product = (
            replacements["product_sku"]
            .value_counts()
            .index[0]
        )

        insights.append(
            f"Most replacement-heavy SKU in "
            f"the current dataset: {product}."
        )

    # Transfer concentration
    transfer_tickets = tickets[
        tickets["transfers"] > 0
    ]

    if len(transfer_tickets) > 0:

        team = (
            transfer_tickets["assigned_team"]
            .value_counts()
            .index[0]
        )

        insights.append(
            f"Transfers are most frequently "
            f"associated with the {team} queue."
        )

    return insights