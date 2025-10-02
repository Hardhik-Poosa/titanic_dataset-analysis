import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
try:
    df = pd.read_excel('d:/nptel_ds/Titanic_Dataset.xlsx')
except FileNotFoundError:
    print("Error: Titanic_Dataset.xlsx not found. Please make sure the file is in the correct directory.")
    exit()

# --- Section 1: EDA ---
print("--- Section 1: EDA ---")

# 1. What is the average age of the passengers?
avg_age = df['Age'].mean()
print(f"1. Average age: {avg_age:.2f}")

# 2. Which passenger class (Pclass) has the highest number of passengers?
pclass_counts = df['Pclass'].value_counts()
highest_pclass = pclass_counts.idxmax()
print(f"2. Pclass with highest number of passengers: Class {highest_pclass}")

# 3. How many passengers were traveling alone?
traveling_alone_count = df['TravelingAlone'].sum()
print(f"3. Passengers traveling alone: {traveling_alone_count}")

# 4. What percentage of passengers survived?
survival_percentage = (df['Survived'].sum() / len(df)) * 100
print(f"4. Percentage of passengers survived: {survival_percentage:.0f}%")

# 5. Which group had a better chance of survival?
survival_by_gender = df.groupby('Gender')['Survived'].mean()
better_survival_gender = survival_by_gender.idxmax()
print(f"5. Group with better chance of survival: {better_survival_gender}s")

# 6. How many passengers embarked from Southampton (S)?
embarked_s_count = df[df['Embarked'] == 'S'].shape[0]
print(f"6. Passengers embarked from Southampton (S): {embarked_s_count}")

# 7. Among passengers aged below 18, which passenger class had the highest survival rate?
under_18 = df[df['Age'] < 18]
survival_rate_under_18_by_pclass = under_18.groupby('Pclass')['Survived'].mean()
highest_survival_pclass_under_18 = survival_rate_under_18_by_pclass.idxmax()
print(f"7. Among passengers below 18, Pclass with highest survival rate: Class {highest_survival_pclass_under_18}")

# 8. Which combination had the best chance of survival?
f_c1_survival_rate = df[(df['Gender'] == 'Female') & (df['Pclass'] == 1)]['Survived'].mean()
f_c3_survival_rate = df[(df['Gender'] == 'Female') & (df['Pclass'] == 3)]['Survived'].mean()
m_c2_survival_rate = df[(df['Gender'] == 'Male') & (df['Pclass'] == 2)]['Survived'].mean()
m_c3_survival_rate = df[(df['Gender'] == 'Male') & (df['Pclass'] == 3)]['Survived'].mean()
combinations = {
    "Female, Class 1": f_c1_survival_rate,
    "Female, Class 3": f_c3_survival_rate,
    "Male, Class 2": m_c2_survival_rate,
    "Male, Class 3": m_c3_survival_rate
}
best_combination = max(combinations, key=combinations.get)
print(f"8. Combination with the best chance of survival: {best_combination}")


# --- Section 2: K-NN ---
print("\n--- Section 2: K-NN ---")

df_encoded = df.copy()
df_encoded['Gender'] = df_encoded['Gender'].map({'Male': 0, 'Female': 1})
df_encoded['Embarked'] = df_encoded['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

new_passenger = {'Age': 61, 'Gender': 0, 'Pclass': 2, 'Embarked': 2, 'TravelingAlone': 1}
features = ['Age', 'Gender', 'Pclass', 'Embarked', 'TravelingAlone']

distances = []
for i, row in df_encoded.iterrows():
    dist = np.sqrt(sum([(row[feat] - new_passenger[feat])**2 for feat in features]))
    distances.append((row['PassengerID'], dist))

distances.sort(key=lambda x: x[1])

# 1. 3 nearest neighbors
nearest_3 = [p[0] for p in distances[:3]]
print(f"1. 3 nearest neighbors (PassengerIDs): {nearest_3}")

# 2. Predict survival with K=5
k = 5
k_nearest_passengers = df_encoded[df_encoded['PassengerID'].isin([p[0] for p in distances[:k]])]
prediction = k_nearest_passengers['Survived'].mode()[0]
print(f"2. Prediction for new passenger with K=5: {prediction} (0=No, 1=Yes)")

# 3. Count survivors with K=9
k = 9
k_nearest_passengers_9 = df_encoded[df_encoded['PassengerID'].isin([p[0] for p in distances[:k]])]
survivors_in_k9 = k_nearest_passengers_9['Survived'].sum()
print(f"3. Number of survivors in 9 nearest neighbors: {survivors_in_k9}")


# --- Section 3: K-Means Clustering ---
print("\n--- Section 3: K-Means Clustering ---")

df_kmeans = df_encoded.copy()

min_age = df_kmeans['Age'].min()
max_age = df_kmeans['Age'].max()
df_kmeans['Age_norm'] = round((df_kmeans['Age'] - min_age) / (max_age - min_age), 1)

clustering_features = ['Age_norm', 'Gender', 'Pclass', 'Embarked', 'TravelingAlone']

c1_data = df_kmeans[df_kmeans['PassengerID'] == 4][clustering_features].iloc[0]
c2_data = df_kmeans[df_kmeans['PassengerID'] == 46][clustering_features].iloc[0]

# 1. Assign passenger 99
passenger_99 = df_kmeans[df_kmeans['PassengerID'] == 99][clustering_features].iloc[0]
dist_to_c1 = np.sqrt(sum([(passenger_99[feat] - c1_data[feat])**2 for feat in clustering_features]))
dist_to_c2 = np.sqrt(sum([(passenger_99[feat] - c2_data[feat])**2 for feat in clustering_features]))

if dist_to_c1 < dist_to_c2:
    print("1. Passenger 99 is assigned to Cluster 1 (PassengerID 4)")
else:
    print("1. Passenger 99 is assigned to Cluster 2 (PassengerID 46)")

# 2. Distance between passenger 9 and C2
passenger_9 = df_kmeans[df_kmeans['PassengerID'] == 9][clustering_features].iloc[0]
dist_p9_c2 = np.sqrt(sum([(passenger_9[feat] - c2_data[feat])**2 for feat in clustering_features]))
print(f"2. Distance between passenger 9 and C2: {dist_p9_c2:.1f}")

# 3. Which cluster contained more passengers?
assignments = []
for i, row in df_kmeans.iterrows():
    p_data = row[clustering_features]
    dist1 = np.sqrt(sum([(p_data[feat] - c1_data[feat])**2 for feat in clustering_features]))
    dist2 = np.sqrt(sum([(p_data[feat] - c2_data[feat])**2 for feat in clustering_features]))
    if dist1 < dist2:
        assignments.append(1)
    else:
        assignments.append(2)

df_kmeans['cluster'] = assignments
cluster_counts = df_kmeans['cluster'].value_counts()
if cluster_counts[1] > cluster_counts[2]:
    print("3. Cluster 1 contained more passengers.")
elif cluster_counts[2] > cluster_counts[1]:
    print("3. Cluster 2 contained more passengers.")
else:
    print("3. Both clusters have an equal number of passengers.")

# --- Section 2: K-NN Visualization ---
print("\n--- Generating K-NN Visualization ---")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_encoded, x='Age', y='Pclass', hue='Survived', style='Survived', s=100, palette='viridis')

# Highlight the new passenger
plt.scatter(new_passenger['Age'], new_passenger['Pclass'], color='red', marker='*', s=300, label='New Passenger')

# Highlight the K-nearest neighbors
k=5
k_nearest_df = df_encoded[df_encoded['PassengerID'].isin([p[0] for p in distances[:k]])]
plt.scatter(k_nearest_df['Age'], k_nearest_df['Pclass'], facecolors='none', edgecolors='red', s=200, label='5-Nearest Neighbors')

plt.title('K-NN Visualization (k=5)')
plt.xlabel('Age')
plt.ylabel('Passenger Class')
plt.legend()
plt.grid(True)
plt.savefig('d:/nptel_ds/knn_visualization.png')
print("K-NN visualization saved to d:/nptel_ds/knn_visualization.png")


# --- Section 3: K-Means Visualization ---
print("\n--- Generating K-Means Visualization ---")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_kmeans, x='Age', y='Pclass', hue='cluster', style='cluster', s=100, palette='plasma')

# Plot cluster centers
c1_plot_data = df[df['PassengerID'] == 4]
c2_plot_data = df[df['PassengerID'] == 46]
plt.scatter(c1_plot_data['Age'], c1_plot_data['Pclass'], color='blue', marker='X', s=300, label='Cluster 1 Center (P4)')
plt.scatter(c2_plot_data['Age'], c2_plot_data['Pclass'], color='orange', marker='X', s=300, label='Cluster 2 Center (P46)')


plt.title('K-Means Clustering Visualization')
plt.xlabel('Age')
plt.ylabel('Passenger Class')
plt.legend()
plt.grid(True)
plt.savefig('d:/nptel_ds/kmeans_visualization.png')
print("K-Means visualization saved to d:/nptel_ds/kmeans_visualization.png")