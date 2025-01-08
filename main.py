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
threshold = 5  # Adjust threshold as needed
tourism_data['High_Tourism_Impact'] = tourism_data['Tourism_GDP_Percentage'] > threshold

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
transactions = tourism_data.groupby('country')['High_Tourism_Impact'].apply(lambda x: [f'High_Impact' if val else 'Low_Impact' for val in x]).tolist()

# Encode transactions
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
transaction_df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori
frequent_itemsets = apriori(transaction_df, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)

def predict_high_tourism_impact(rules, country):
    country_data = tourism_data[tourism_data['country'] == country]
    if country_data.empty:
        return "Country not found in the dataset."
    
    # Check if the country has consistently high impact
    if all(country_data['High_Tourism_Impact']):
        return f"{country} consistently has high tourism impact."
    
    # Look for rules that might predict high impact
    for _, rule in rules.iterrows():
        if 'High_Impact' in rule['consequents']:
            return f"{country} might have high tourism impact (confidence: {rule['confidence']:.2f})."
    
    return f"No strong indication of high tourism impact for {country}."

# Example usage
country_to_predict = 'Portugal'  # Replace with any country in your dataset
prediction = predict_high_tourism_impact(rules, country_to_predict)
print(prediction)