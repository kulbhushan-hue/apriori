"""
Grocery Market Basket Dashboard
--------------------------------
Streamlit app that loads `grocery_apriori_model.pkl` (built by
train_apriori.py from groceries.csv) and lets you explore:

  * Top-selling items and item combos (by purchase frequency)
  * Cross-sell recommendations for a basket you build in the sidebar
  * The full frequent-itemset / association-rule tables

NOTE ON "SALES": groceries.csv has no price or quantity column — only
whether an item was bought in a transaction. So every "sales" figure
in this app is purchase FREQUENCY (support), not revenue. If you have
per-item price data, multiply support by price to get a revenue-
weighted ranking instead.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push this folder (app.py, grocery_apriori_model.pkl, requirements.txt) to GitHub.
    2. Go to share.streamlit.io -> New app -> point to app.py.
"""

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Grocery Market Basket Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_artifact(path: str = "grocery_apriori_model.pkl"):
    return joblib.load(path)


artifact = load_artifact()
item_support: pd.Series = artifact["item_support"]
itemsets_df: pd.DataFrame = artifact["frequent_itemsets"]
rules_df: pd.DataFrame = artifact["association_rules"]
all_items = item_support.index.tolist()

st.title("🛒 Grocery Market Basket Dashboard")
st.caption(
    "Frequent itemsets and cross-sell rules mined with Apriori from "
    f"{artifact['n_transactions']:,} real transactions across {artifact['n_items']} items."
)

with st.expander("ℹ️ How \"maximum sales\" is defined here (read this once)"):
    st.write(artifact["sales_proxy_note"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{artifact['n_transactions']:,}")
    c2.metric("Items tracked", artifact["n_items"])
    c3.metric("Frequent itemsets", len(itemsets_df))
    c4.metric("Association rules", len(rules_df))
    st.caption(
        f"Mined with min_support={artifact['min_support']}, "
        f"min_confidence={artifact['min_confidence']}, "
        f"max itemset size={artifact['max_itemset_size']}."
    )

st.divider()

# ----------------------------------------------------------------------
# Sidebar: build a basket
# ----------------------------------------------------------------------
st.sidebar.header("🧺 Build a Basket")
basket = st.sidebar.multiselect(
    "Items already in the basket",
    options=all_items,
    help="Pick 1+ items — the Recommendations tab will suggest what to add next to maximize the sale.",
)
top_n = st.sidebar.slider("How many recommendations?", 3, 15, 5)
min_lift_filter = st.sidebar.slider("Minimum lift for recommendations", 1.0, 5.0, 1.2, 0.1)

tab_recommend, tab_top, tab_rules, tab_about = st.tabs(
    ["🎯 Recommendations", "📈 Top Sellers", "🔗 Rule Explorer", "🧠 Model Info"]
)

# ----------------------------------------------------------------------
# Tab 1: Recommendations for the current basket
# ----------------------------------------------------------------------
with tab_recommend:
    if not basket:
        st.info("👈 Add one or more items to the basket in the sidebar to see cross-sell recommendations.")
    else:
        basket_set = set(basket)
        matches = rules_df[rules_df["antecedents"].apply(lambda a: set(a).issubset(basket_set))]
        matches = matches[matches["lift"] >= min_lift_filter]
        matches = matches.sort_values(["lift", "confidence"], ascending=False)

        # Aggregate: a consequent item may appear in multiple matching rules — keep its best rule
        matches = matches.assign(consequent_str=matches["consequents"].apply(lambda c: ", ".join(c)))
        best_per_consequent = matches.sort_values("lift", ascending=False).drop_duplicates("consequent_str")
        top_recs = best_per_consequent.head(top_n)

        if top_recs.empty:
            st.warning(
                "No rules matched this basket at the current minimum lift. "
                "Try lowering the lift slider or choosing more common items (e.g. whole milk, other vegetables)."
            )
        else:
            st.subheader(f"Recommended add-ons for: {', '.join(basket)}")
            fig = px.bar(
                top_recs.sort_values("lift"),
                x="lift", y="consequent_str", orientation="h",
                color="confidence", color_continuous_scale="Greens",
                labels={"lift": "Lift", "consequent_str": "Recommended item", "confidence": "Confidence"},
                title="Recommended items to maximize this sale (higher lift = stronger cross-sell signal)",
            )
            fig.update_layout(height=90 + 40 * len(top_recs))
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                top_recs[["antecedents", "consequents", "support", "confidence", "lift"]]
                .rename(columns={
                    "antecedents": "If basket has",
                    "consequents": "Then also recommend",
                    "support": "Support (joint freq.)",
                    "confidence": "Confidence",
                    "lift": "Lift",
                }),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "**Lift > 1** means these items sell together more often than chance would predict — "
                "the stronger signal for a deliberate cross-sell placement or bundle discount."
            )

# ----------------------------------------------------------------------
# Tab 2: Top sellers (items and combos)
# ----------------------------------------------------------------------
with tab_top:
    st.subheader("Top-selling individual items")
    top_items_n = st.slider("Show top N items", 5, 30, 15, key="top_items_n")
    top_items_df = item_support.head(top_items_n).reset_index()
    top_items_df.columns = ["item", "support"]
    fig1 = px.bar(
        top_items_df.sort_values("support"), x="support", y="item", orientation="h",
        labels={"support": "Fraction of transactions", "item": ""},
        title=f"Top {top_items_n} items by purchase frequency (sales proxy)",
    )
    fig1.update_layout(height=90 + 28 * top_items_n, xaxis_tickformat=".0%")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Top-selling item COMBINATIONS")
    combos = itemsets_df[itemsets_df["length"] >= 2].head(15).copy()
    combos["items_str"] = combos["items"].apply(lambda t: " + ".join(t))
    fig2 = px.bar(
        combos.sort_values("support"), x="support", y="items_str", orientation="h",
        labels={"support": "Fraction of transactions", "items_str": ""},
        title="Top 15 item combinations by joint purchase frequency",
    )
    fig2.update_layout(height=90 + 28 * len(combos), xaxis_tickformat=".1%")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        f"Maximum single-item seller: **{item_support.index[0]}** "
        f"({item_support.iloc[0]*100:.1f}% of transactions). "
        f"Maximum combo seller: **{' + '.join(artifact['top_itemset'])}**."
    )

# ----------------------------------------------------------------------
# Tab 3: Full rule / itemset explorer
# ----------------------------------------------------------------------
with tab_rules:
    st.subheader("Explore all mined association rules")
    colf1, colf2, colf3 = st.columns(3)
    with colf1:
        min_conf = st.slider("Min confidence", 0.0, 1.0, 0.3, 0.05)
    with colf2:
        min_lift = st.slider("Min lift", 1.0, 5.0, 1.0, 0.1)
    with colf3:
        contains_item = st.selectbox("Only rules involving item (optional)", ["(any)"] + all_items)

    view = rules_df[(rules_df["confidence"] >= min_conf) & (rules_df["lift"] >= min_lift)]
    if contains_item != "(any)":
        view = view[
            view["antecedents"].apply(lambda a: contains_item in a)
            | view["consequents"].apply(lambda c: contains_item in c)
        ]

    st.caption(f"{len(view)} rule(s) match these filters.")
    st.dataframe(
        view.sort_values(["lift", "confidence"], ascending=False)[
            ["antecedents", "consequents", "support", "confidence", "lift"]
        ],
        use_container_width=True, hide_index=True,
    )

    st.subheader("All frequent itemsets")
    len_filter = st.multiselect("Itemset size", sorted(itemsets_df["length"].unique()), default=[1, 2, 3])
    st.dataframe(
        itemsets_df[itemsets_df["length"].isin(len_filter)].sort_values("support", ascending=False),
        use_container_width=True, hide_index=True,
    )

# ----------------------------------------------------------------------
# Tab 4: Model info / card
# ----------------------------------------------------------------------
with tab_about:
    st.subheader("Model card")
    st.json({
        "algorithm": artifact["algorithm"],
        "n_transactions": artifact["n_transactions"],
        "n_items": artifact["n_items"],
        "min_support": artifact["min_support"],
        "min_confidence": artifact["min_confidence"],
        "max_itemset_size": artifact["max_itemset_size"],
        "n_frequent_itemsets": len(itemsets_df),
        "n_association_rules": len(rules_df),
        "top_selling_item": item_support.index[0],
        "top_selling_combo": list(artifact["top_itemset"]),
    })
    st.info(artifact["sales_proxy_note"])

st.divider()
st.caption("Built with Streamlit • Apriori market-basket model trained on grocery transaction data.")
