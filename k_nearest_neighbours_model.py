import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import math
import matplotlib.pyplot as plt
import seaborn as sns

wine_quality_color = 'white'

print(f'You have chosen the {wine_quality_color} wine dataset.\n')

# Load the dataset
df = pd.read_csv(f"winequality-{wine_quality_color}.csv", delimiter=';')

# Separate features (input) and target (quality)
def quality_label(quality):
    if quality < 6:
        return 0
    elif quality == 6:
        return 1
    else:
        return 2

df['quality_class'] = df['quality'].apply(quality_label)
input_features = df.drop(['quality', 'quality_class'], axis=1)
target_quality = df['quality_class']

class_names = ["Low", "Medium", "High"]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(input_features, target_quality, test_size=0.3, random_state=42)

# --- Apply Feature Scaling ---
# Initialize the StandardScaler (no specific min/max)
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and test data
# It's crucial to fit ONLY on X_train to prevent data leakage from the test set
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Calculate 'k' (number of neighbors) using the square root heuristic on the original training size
count_of_training_samples = len(X_train)
neighbours_float = math.sqrt(count_of_training_samples)
max_k = math.floor(neighbours_float)

#test each K value to find best K for dataset
k_values = range(1, max_k+1, 2)
print(f'Testing each K value from 1 to {max_k}')

chosen_k = 0
chosen_k_percent = 0.0

for k in k_values:
    knn_scaled = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn_scaled.fit(X_train_scaled, Y_train)  # Fit on scaled training data
    Y_pred_scaled = knn_scaled.predict(X_test_scaled)

    accuracy_scaled = accuracy_score(Y_test, Y_pred_scaled)
    accuracy_percentage = accuracy_scaled * 100
    print(f"K Value: {k} | Accuracy (Scaled Features): {accuracy_percentage:.2f}%")

    if accuracy_percentage >= chosen_k_percent:
        chosen_k = k
        chosen_k_percent = accuracy_percentage

print(f'\nAfter test runs, the chosen K value is {chosen_k}')

# --- Model Evaluation with Scaled Features ---
print("\n--- Model Evaluation on Test Set with Scaled Features ---")

knn_scaled = KNeighborsClassifier(n_neighbors=chosen_k, metric='euclidean')
knn_scaled.fit(X_train_scaled, Y_train)  # Fit on scaled training data
Y_pred_scaled = knn_scaled.predict(X_test_scaled)

# Classification Report
print(f"\nGeneral Classification Report (Scaled Features) ({wine_quality_color}):\n")
print(classification_report(Y_test, Y_pred_scaled, target_names=class_names, zero_division=0))

print(f"Feature\t\t\tMutual Information Score ({wine_quality_color}):")
mi = mutual_info_classif(X_train, Y_train, discrete_features=False)
feature_importance_mi = pd.Series(mi, index=input_features.columns).sort_values(ascending=False)
print(feature_importance_mi)

example_tuples = [
    [7.2, 0.65, 0.12, 2.3, 0.076, 13.0, 45.0, 0.9967, 3.30, 0.54, 9.6],
    [6.1, 0.30, 0.44, 1.8, 0.059, 29.0, 90.0, 0.9936, 3.47, 0.72, 11.2],
    [9.4, 0.35, 0.58, 2.1, 0.074, 40.0, 122.0, 0.9969, 3.26, 0.87, 11.8],
    [5.5, 0.96, 0.02, 2.5, 0.104, 10.0, 18.0, 0.9980, 3.08, 0.38, 8.5],
    [8.7, 0.41, 0.45, 3.0, 0.091, 25.0, 110.0, 0.9973, 3.18, 0.70, 10.3],
]

scaled_tuples = scaler.fit_transform(example_tuples)

print()

for i in range(len(scaled_tuples)):
    print(f"Input Tuple: {scaled_tuples[i]}")
    prediction = knn_scaled.predict([scaled_tuples[i]])

    if prediction < 1:
        print('Prediction: Low Quality')
    elif prediction == 1:
        print('Prediction: Medium Quality')
    else:
        print('Prediction: High Quality')

# Confusion Matrix
class_labels = ['Low', 'Medium', 'High']
num_labels = [0, 1, 2]
cm_scaled = confusion_matrix(Y_test, Y_pred_scaled, labels=num_labels)
print("\nConfusion Matrix (Scaled Features):")

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_scaled, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Quality')
plt.ylabel('True Quality')
plt.title(f'Confusion Matrix for {wine_quality_color} Wine Quality Prediction (Scaled Features)')
plt.show()

