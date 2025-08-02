import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import tree

#select wine quality color data set here ('white' or 'red')
wine_quality_color = 'red'

print(f'You have chosen the {wine_quality_color} wine dataset.\n')

df = pd.read_csv(f"winequality-{wine_quality_color}.csv" , delimiter = ";")
df_input = df.drop("quality", axis ='columns')
target = df["quality"]
X_train , X_test , Y_train , Y_test = train_test_split(df_input, target, test_size= 0.3)

model = tree.DecisionTreeClassifier()
model.fit(X_train , Y_train)

Y_pred = model.predict(X_test)

target_names = ['Low', 'Medium', 'High']

def quality_to_label(q):
    if q < 6:
        return "Low"
    elif q == 6:
        return "Medium"
    else:
        return "High"

y_true_mapped = [quality_to_label(val) for val in Y_test]
y_pred_mapped = [quality_to_label(val) for val in Y_pred]

print('General Classification Report:')
print(classification_report(
    y_true_mapped,
    y_pred_mapped,
    target_names=target_names
))

#predicting tuples

tuple_tests = [
    [7.2, 0.65, 0.12, 2.3, 0.076, 13.0, 45.0, 0.9967, 3.30, 0.54, 9.6],
    [6.1, 0.30, 0.44, 1.8, 0.059, 29.0, 90.0, 0.9936, 3.47, 0.72, 11.2],
    [9.4, 0.35, 0.58, 2.1, 0.074, 40.0, 122.0, 0.9969, 3.26, 0.87, 11.8],
    [5.5, 0.96, 0.02, 2.5, 0.104, 10.0, 18.0, 0.9980, 3.08, 0.38, 8.5],
    [8.7, 0.41, 0.45, 3.0, 0.091, 25.0, 110.0, 0.9973, 3.18, 0.70, 10.3],
]

print('Tuple Test Predictions:\n')
for i in range(len(tuple_tests)):

    prediction = model.predict([tuple_tests[i]])

    if prediction <= 5:
        prediction = "Low"
    elif prediction == 6:
        prediction = "Medium"
    else:
        prediction = "High"

    print(f"Tuple: {tuple_tests[i]}")
    print(f"Prediction: {prediction}\n")

#feature importances
importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 6))
plt.bar(importance['Feature'], importance['Importance'])
plt.title(f'Feature Importances ({wine_quality_color})')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=70)
plt.tight_layout()
plt.show()
