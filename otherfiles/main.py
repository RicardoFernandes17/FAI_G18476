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

# =======================================
# Step 1: Data Preparation
# =======================================
# Create a new feature: Tourism GDP Percentage
# Assuming 'tourism_receipts' is equivalent to 'Tourism GDP'
tourism_data['Tourism_GDP_Percentage'] = (tourism_data['tourism_receipts'] / tourism_data['gdp']) * 100

# Create a binary target variable: High Tourism Impact
tourism_threshold = 5  # Adjust threshold as needed
low_unemployment_threshold = 5  # Define threshold

tourism_data['High_Tourism_Impact'] = tourism_data['Tourism_GDP_Percentage'] > tourism_threshold
tourism_data['Low_Unemployment'] = tourism_data['unemployment_rate'] < low_unemployment_threshold

# Handle missing values (drop rows with missing required columns)
tourism_data = tourism_data.dropna(subset=['tourism_receipts', 'gdp', 'tourism_arrivals'])

# Inspect the dataset
# print(tourism_data.head())

# =======================================
# Step 2: Decision Tree Classification
# =======================================
# Prepare features and target
X = tourism_data[['tourism_arrivals', 'tourism_receipts', 'gdp']]  # Relevant columns
y = tourism_data['High_Tourism_Impact']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=1234, max_depth=4)
clf_model = clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy:.2f}")

# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(clf_model, filled=True, rounded=True, feature_names=X.columns, class_names=['Low Impact', 'High Impact'])
plt.show()

# =======================================
# Step 3: K-Means Clustering
# =======================================
# Select features for clustering
features = tourism_data[['tourism_arrivals', 'tourism_receipts', 'gdp']]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine the optimal number of clusters using the elbow method
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

# Perform clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_features)

# Add cluster labels to the dataset
tourism_data['Cluster'] = clusters

# Visualize the clusters in 2D
plt.scatter(features['tourism_receipts'], features['tourism_arrivals'], c=clusters, cmap='viridis', marker='o')
plt.xlabel('Tourism Receipts')
plt.ylabel('Tourism Arrivals')
plt.title('2D Clustering of Tourism Data')
plt.show()

# =======================================
# Step 4: Association Rule Mining
# =======================================
# Prepare the data for association rule mining
#print(tourism_data)
transactions = tourism_data[['country', 'High_Tourism_Impact', 'year']]

# Encode transactions

def predict_high_tourism_impact(country, min_support=0.2, min_confidence=0.7):
    # Filter data for the given country and after 2010
    country_data = tourism_data[(tourism_data['country'] == country) & (tourism_data['year'] > 2010)]

    if country_data.empty:
        return "Country not found in the dataset or no data after 2010."

    # Refine the pivot table for better insights
    ds_pivot = country_data.pivot_table(
        index='country',
        columns='year',
        values='High_Tourism_Impact',
        aggfunc='max'
    ).fillna(False)  # Binary pivot table

    # Run Apriori algorithm
    freq_itemsets = apriori(ds_pivot, min_support=min_support, use_colnames=True)
    print("Frequent Itemsets:")
    print(freq_itemsets)

    rules = association_rules(freq_itemsets, metric="confidence", min_threshold=min_confidence)
    print("Association Rules:")
    print(rules)

    if rules.empty:
        return f"No strong rules found to predict tourism impact for {country}."

    # Filter for rules predicting High_Tourism_Impact
    high_impact_rules = rules[rules['consequents'].apply(lambda x: 'High_Tourism_Impact' in list(x))]

    if not high_impact_rules.empty:
        best_rule = high_impact_rules.iloc[0]
        return (
            f"{country} might have high tourism impact based on patterns in the data. "
            f"(Support: {best_rule['support']:.2f}, Confidence: {best_rule['confidence']:.2f})"
        )

    return f"No strong indication of high tourism impact for {country}."


# Example usage
country_to_predict = 'Portugal'  # Replace with any country in your dataset
prediction = predict_high_tourism_impact(country_to_predict)
print(prediction)