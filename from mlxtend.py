from mlxtend.frequent_patterns import apriori, association_rules

# =======================================
# Step 1: Create Binary Features
# =======================================
low_unemployment_threshold = 5  # Define threshold
tourism_data['Low_Unemployment'] = tourism_data['unemployment_rate'] < low_unemployment_threshold

# =======================================
# Step 2: Create Pivot Table
# =======================================
# Pivot table for High Tourism Impact and Low Unemployment
pivot_table = tourism_data.pivot_table(
    index='country',  # Countries as rows
    columns='year',  # Years as columns
    values=['High_Tourism_Impact', 'Low_Unemployment'],
    aggfunc='any'  # Aggregate using any() for binary features
).fillna(False)

# Flatten multi-level column names
pivot_table.columns = [f"{value}_{year}" for value, year in pivot_table.columns]

# =======================================
# Step 3: Prepare Transactions for Apriori
# =======================================
# Convert pivot table to binary (True/False) transactions
transactions = pivot_table > 0

# =======================================
# Step 4: Apply Apriori Algorithm
# =======================================
min_support = 0.2  # Minimum support threshold
freq_itemsets = apriori(transactions, min_support=min_support, use_colnames=True)
print("Frequent Itemsets:")
print(freq_itemsets)

# Generate association rules
min_confidence = 0.7  # Minimum confidence threshold
rules = association_rules(freq_itemsets, metric="confidence", min_threshold=min_confidence)

# Filter rules to find those where High Tourism Impact implies Low Unemployment
rules_high_tourism_to_low_unemployment = rules[
    (rules['antecedents'].apply(lambda x: any('High_Tourism_Impact' in str(item) for item in x))) &
    (rules['consequents'].apply(lambda x: any('Low_Unemployment' in str(item) for item in x)))
]

print("Rules where High Tourism Impact -> Low Unemployment:")
print(rules_high_tourism_to_low_unemployment)

# =======================================
# Step 5: Interpret Results
# =======================================
if not rules_high_tourism_to_low_unemployment.empty:
    best_rule = rules_high_tourism_to_low_unemployment.iloc[0]
    print(
        f"High Tourism Impact is associated with Low Unemployment. "
        f"Support: {best_rule['support']:.2f}, Confidence: {best_rule['confidence']:.2f}, Lift: {best_rule['lift']:.2f}"
    )
else:
    print("No strong association between High Tourism Impact and Low Unemployment found.")
