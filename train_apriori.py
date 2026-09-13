"""
Market Basket Analysis: Apriori on the Groceries dataset
----------------------------------------------------------
Trains a from-scratch Apriori implementation (no mlxtend dependency —
scikit-learn has no built-in Apriori, and mlxtend isn't guaranteed to be
available everywhere) against `groceries.csv`, a one-hot encoded
transaction table (9,835 transactions x 169 items, True/False per item).

IMPORTANT ASSUMPTION
---------------------
The dataset has no price or quantity column, only whether an item was
purchased in a transaction. There is no ground-truth "sales" figure to
predict. So "maximum sales" is approximated by PURCHASE FREQUENCY
(support) — the fraction of transactions containing an item or itemset.
This is standard practice in market-basket analysis when revenue data
isn't available. If real price/quantity data becomes available, replace
the support-based ranking with a revenue-weighted one and keep the rest
of the pipeline unchanged.

Outputs
-------
* grocery_apriori_model.pkl — item supports, frequent itemsets,
  association rules, and top "maximum sales" rankings, all as plain
  data (no functions), so it loads identically in any environment.
"""

import itertools
import time

import joblib
import numpy as np
import pandas as pd

RANDOM_STATE = 42
MIN_SUPPORT = 0.01     # itemset must appear in >=1% of transactions
MIN_CONFIDENCE = 0.20  # rule must hold >=20% of the time antecedent occurs
                        # (0.30 was tried first, but it excludes every rule for
                        # "whole milk" alone -- its strongest single pairing,
                        # other vegetables, sits at 29.3% confidence. 0.20 keeps
                        # the most-purchased item usable for recommendations.)
MAX_ITEMSET_SIZE = 3   # mine up to triples (pairs + triples cover actionable cross-sell rules)

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
df = pd.read_csv("groceries.csv")
items = df.columns.tolist()
X = df.to_numpy(dtype=bool)
n_transactions, n_items = X.shape
print(f"Loaded {n_transactions} transactions x {n_items} items")

# ----------------------------------------------------------------------
# 2. Level 1: single-item support (vectorized)
# ----------------------------------------------------------------------
item_support = X.mean(axis=0)
support_series = pd.Series(item_support, index=items).sort_values(ascending=False)

L1_mask = item_support >= MIN_SUPPORT
L1_idx = np.where(L1_mask)[0]
print(f"L1: {len(L1_idx)} / {n_items} items pass min_support={MIN_SUPPORT}")

frequent_itemsets = []  # list of dicts: {"items": frozenset, "support": float}
for i in L1_idx:
    frequent_itemsets.append({"items": frozenset([items[i]]), "support": float(item_support[i])})

# ----------------------------------------------------------------------
# 3. Level 2: pairs, via a single co-occurrence matrix multiply
# ----------------------------------------------------------------------
t0 = time.time()
X_L1 = X[:, L1_idx]                                   # (n_transactions, n_L1)
co_counts = X_L1.T.astype(np.int32) @ X_L1.astype(np.int32)   # (n_L1, n_L1) co-occurrence counts
co_support = co_counts / n_transactions

L2_pairs = []  # (i_local, j_local, support)
n_L1 = len(L1_idx)
for a in range(n_L1):
    for b in range(a + 1, n_L1):
        s = co_support[a, b]
        if s >= MIN_SUPPORT:
            L2_pairs.append((a, b, s))
            frequent_itemsets.append({
                "items": frozenset([items[L1_idx[a]], items[L1_idx[b]]]),
                "support": float(s),
            })
print(f"L2: {len(L2_pairs)} frequent pairs (in {time.time()-t0:.2f}s)")

# ----------------------------------------------------------------------
# 4. Level 3: triples, candidates generated only from frequent pairs
#    (classic Apriori join + prune step)
# ----------------------------------------------------------------------
t0 = time.time()
pair_set = {(a, b) for a, b, _ in L2_pairs}
# neighbours[a] = set of b such that (a,b) is a frequent pair (a<b)
neighbours = {}
for a, b, _ in L2_pairs:
    neighbours.setdefault(a, set()).add(b)

L3_triples = []
candidate_triples = set()
for a in sorted(neighbours):
    b_candidates = sorted(neighbours[a])
    for bi in range(len(b_candidates)):
        b = b_candidates[bi]
        for cj in range(bi + 1, len(b_candidates)):
            c = b_candidates[cj]
            # join step: (a,b) and (a,c) frequent -> candidate (a,b,c)
            # prune step: also require (b,c) frequent
            if (b, c) in pair_set:
                candidate_triples.add((a, b, c))

for (a, b, c) in candidate_triples:
    mask = X_L1[:, a] & X_L1[:, b] & X_L1[:, c]
    s = mask.mean()
    if s >= MIN_SUPPORT:
        L3_triples.append((a, b, c, s))
        frequent_itemsets.append({
            "items": frozenset([items[L1_idx[a]], items[L1_idx[b]], items[L1_idx[c]]]),
            "support": float(s),
        })
print(f"L3: {len(candidate_triples)} candidates -> {len(L3_triples)} frequent triples (in {time.time()-t0:.2f}s)")

itemsets_df = pd.DataFrame(frequent_itemsets).sort_values("support", ascending=False).reset_index(drop=True)
itemsets_df["length"] = itemsets_df["items"].apply(len)
print(f"\nTotal frequent itemsets (size 1-3): {len(itemsets_df)}")

# ----------------------------------------------------------------------
# 5. Association rules from itemsets of size >= 2
# ----------------------------------------------------------------------
support_lookup = {row["items"]: row["support"] for _, row in itemsets_df.iterrows()}

rules = []
for _, row in itemsets_df[itemsets_df["length"] >= 2].iterrows():
    itemset = row["items"]
    itemset_support = row["support"]
    items_list = list(itemset)
    k = len(items_list)
    for r in range(1, k):
        for antecedent_tuple in itertools.combinations(items_list, r):
            antecedent = frozenset(antecedent_tuple)
            consequent = itemset - antecedent
            ant_support = support_lookup.get(antecedent)
            if ant_support is None or ant_support == 0:
                continue
            confidence = itemset_support / ant_support
            if confidence < MIN_CONFIDENCE:
                continue
            cons_support = support_lookup.get(consequent)
            if cons_support is None:
                # consequent is a single item not separately tracked as its own row for size>1 itemsets;
                # single items are always in itemsets_df at length 1, so this should always resolve.
                continue
            lift = confidence / cons_support
            rules.append({
                "antecedents": antecedent,
                "consequents": consequent,
                "antecedent_support": ant_support,
                "consequent_support": cons_support,
                "support": itemset_support,
                "confidence": confidence,
                "lift": lift,
            })

rules_df = pd.DataFrame(rules).sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)
print(f"Generated {len(rules_df)} association rules (min_confidence={MIN_CONFIDENCE})")

# ----------------------------------------------------------------------
# 6. "Maximum sales" rankings (support = purchase-frequency proxy for sales)
# ----------------------------------------------------------------------
top_items = support_series.head(10)
top_itemsets_size2plus = itemsets_df[itemsets_df["length"] >= 2].head(10)
top_rules_by_lift = rules_df.head(10)

print("\nTop-selling single item:", top_items.index[0], f"({top_items.iloc[0]*100:.2f}% of transactions)")
print("Top-selling combo:", dict(top_itemsets_size2plus.iloc[0][["items", "support"]]))
print("\nTop cross-sell rule by lift:")
print(top_rules_by_lift.iloc[0][["antecedents", "consequents", "support", "confidence", "lift"]])

# ----------------------------------------------------------------------
# 7. Persist everything the Streamlit app needs into ONE pickle
# ----------------------------------------------------------------------
# Convert frozensets to sorted tuples for a cleaner, environment-agnostic pickle
def fs_to_tuple(fs):
    return tuple(sorted(fs))

itemsets_export = itemsets_df.copy()
itemsets_export["items"] = itemsets_export["items"].apply(fs_to_tuple)

rules_export = rules_df.copy()
rules_export["antecedents"] = rules_export["antecedents"].apply(fs_to_tuple)
rules_export["consequents"] = rules_export["consequents"].apply(fs_to_tuple)

artifact = {
    "algorithm": "Custom Apriori (pandas/NumPy, no mlxtend dependency)",
    "n_transactions": int(n_transactions),
    "n_items": int(n_items),
    "min_support": MIN_SUPPORT,
    "min_confidence": MIN_CONFIDENCE,
    "max_itemset_size": MAX_ITEMSET_SIZE,
    "item_support": support_series,                 # pd.Series, all 169 items, sorted desc
    "frequent_itemsets": itemsets_export,            # DataFrame: items (tuple), support, length
    "association_rules": rules_export,               # DataFrame: antecedents, consequents, support, confidence, lift
    "top_items": top_items.index.tolist(),
    "top_itemset": fs_to_tuple(itemsets_df[itemsets_df["length"] >= 2].iloc[0]["items"]),
    "sales_proxy_note": (
        "No price/quantity data was available in groceries.csv. 'Maximum sales' "
        "is approximated by purchase frequency (support) across transactions."
    ),
}

joblib.dump(artifact, "grocery_apriori_model.pkl")
print("\nSaved grocery_apriori_model.pkl")
