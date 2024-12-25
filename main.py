import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
file_path = '/mnt/data/world_tourism_economy_data.csv'  # Update with your dataset
tourism_data = pd.read_csv(file_path)

# =======================================
# Step 1: Data Preparation
# =======================================
# Create a new feature: Tourism GDP Percentage
tourism_data['Tourism_GDP_Percentage'] = (tourism_data['Tourism GDP'] / tourism_data['Total GDP']) * 100

# Create a binary target variable: High Tourism Impact
threshold = 5  # Adjust threshold as needed
tourism_data['High_Tourism_Impact'] = tourism_data['Tourism_GDP_Percentage'] > threshold

# Inspect the dataset
print(tourism_data.head())

# =======================================
# Step 2: Decision Tree Classification
# =======================================
# Prepare features and target
X = tourism_data[['Tourism Arrivals', 'Tourism GDP', 'Total GDP']]  # Update with relevant columns
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
features = tourism_data[['Tourism Arrivals', 'Tourism GDP', 'Total GDP']].dropna()

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
plt.scatter(features['Tourism GDP'], features['Tourism Arrivals'], c=clusters, cmap='viridis', marker='o')
plt.xlabel('Tourism GDP')
plt.ylabel('Tourism Arrivals')
plt.title('2D Clustering of Tourism Data')
plt.show()

# =======================================
# Step 4: Association Rule Mining
# =======================================
# Convert data for Apriori
region_group = tourism_data.groupby('Region')[['High_Tourism_Impact']].apply(lambda x: x.astype(str).values.tolist())

# Encode transactions
te = TransactionEncoder()
te_ary = te.fit(region_group).transform(region_group)
transaction_df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori
frequent_itemsets = apriori(transaction_df, min_support=0.3, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)

# Display some rules
print(rules[['antecedents', 'consequents', 'support', 'confidence']])

# Example rule prediction
def predict_tourism_impact(rules_df, region):
    for _, row in rules_df.iterrows():
        if region in row['antecedents']:
            print(f"Association found: {row['antecedents']} -> {row['consequents']}")
            return
    print(f"No specific association rule found for {region}.")

predict_tourism_impact(rules, 'Europe')

# =======================================
# Summary and Insights
# =======================================
print("Data analysis complete. Review outputs and visualizations for insights.")
