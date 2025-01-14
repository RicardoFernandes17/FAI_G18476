
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt

# Loads the dataset
file_path = 'world_tourism_economy_data.csv'  # Replace with the actual path
tourism_data = pd.read_csv(file_path)

tourism_data['Tourism_GDP_Percentage'] = (tourism_data['tourism_receipts'] / tourism_data['gdp']) * 100
tourism_threshold = 5  # Adjust threshold as needed


tourism_data['High_Tourism_Impact'] = tourism_data['Tourism_GDP_Percentage'] > tourism_threshold
# Clean the data
tourism_data = tourism_data.dropna(subset=['High_Tourism_Impact'])

# Group the data by 'country' and 'year', and pivot
ds_grouped = tourism_data.groupby(['country', 'year'], as_index=False).agg({'High_Tourism_Impact': 'any'}).reset_index()

ds_pivot = pd.pivot(data=ds_grouped, index='year', columns='country', values='High_Tourism_Impact').fillna(False)
print(ds_pivot)

# Get frequent itemsets using apriori
min_support = 0.01
freq_itemsets = apriori(ds_pivot, min_support=min_support, use_colnames=True)

# Number of itemsets
num_itemsets = len(freq_itemsets)
#print(f'Number of itemsets: {num_itemsets}')

# Get association rules
rules = association_rules(freq_itemsets, metric="confidence", min_threshold=0.7, num_itemsets=num_itemsets)
rules = rules.sort_values(by='confidence', ascending=False)

# Display the top 10 rules
#print("Top 10 rules by confidence:")
#print(rules.head(10))

# For additional interpretation, list unique countries with 'High_Tourism_Impact'
countries_with_high_impact = tourism_data[tourism_data['High_Tourism_Impact']]['country'].unique()
#print(f"Countries with High Tourism Impact: {countries_with_high_impact}")