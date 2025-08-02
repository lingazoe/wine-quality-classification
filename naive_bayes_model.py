# by Saad Shahid 218769547
from sklearn.feature_selection import mutual_info_classif
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#select wine quality color data set here ('white' or 'red')
wine_quality_color = 'white'

print(f'You have chosen the {wine_quality_color} wine dataset.\n')

wine_quality = fetch_ucirepo(id=186)

df_features = pd.DataFrame(wine_quality.data.features, columns=wine_quality.data.feature_names)

if hasattr(wine_quality.data, "target_names") and wine_quality.data.target_names:
    if isinstance(wine_quality.data.target_names, (list, tuple)) and len(wine_quality.data.target_names) > 0:
        target_name = wine_quality.data.target_names[0]
    elif isinstance(wine_quality.data.target_names, str) and wine_quality.data.target_names:
        target_name = wine_quality.data.target_names
    else:
        target_name = "quality"
else:
    target_name = "quality"

df_features[target_name] = wine_quality.data.targets

df = df_features.copy()

num_white = 4898
num_red = 1599
total_samples = df.shape[0]
if total_samples != num_white + num_red:
    print(f"Expected {num_white + num_red} samples, but got {total_samples}. Adjusting the color split.")
    num_white = int(total_samples * (4898/(4898 + 1599)))
    num_red = total_samples - num_white

colors = ["white"] * num_white + ["red"] * num_red
df["color"] = colors

def quality_to_label(q):
    """Categorical score needs to be mapped to numeric quality"""
    if q <= 5:
        return "Low"
    elif q == 6:
        return "Medium"
    else:
        return "High"

df["quality_label"] = df[target_name].apply(quality_to_label)

label_encoder_color = LabelEncoder()
label_encoder_quality = LabelEncoder()

df["color_encoded"] = label_encoder_color.fit_transform(df["color"])

df["quality_label_encoded"] = label_encoder_quality.fit_transform(df["quality_label"])

df.to_csv("wine_quality_full.csv", index=False)

feature_cols = wine_quality.data.features.columns.tolist()
feature_cols.append("color_encoded")
X = df[feature_cols].values
y = df["quality_label_encoded"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#for transforming the data to fit gaussian
pt = PowerTransformer(method="yeo-johnson")
X_train = pt.fit_transform(X_train)
X_test = pt.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


low_code = label_encoder_quality.transform(["Low"])[0]
med_code = label_encoder_quality.transform(["Medium"])[0]
high_code = label_encoder_quality.transform(["High"])[0]
ordered_labels = [low_code, med_code, high_code]
ordered_names = ["Low", "Medium", "High"]

print("Test Set -> Classification")
print(classification_report(y_test, y_pred, labels=ordered_labels, target_names=ordered_names))

conf_mat = confusion_matrix(y_test, y_pred, labels=ordered_labels)

plt.figure(figsize=(6,5))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=ordered_names, yticklabels=ordered_names)
plt.title("Wine Quality Prediction - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.tight_layout()
plt.show()

example_features = [
    [7.2, 0.65, 0.12, 2.3, 0.076, 13.0, 45.0, 0.9967, 3.30, 0.54, 9.6],
    [6.1, 0.30, 0.44, 1.8, 0.059, 29.0, 90.0, 0.9936, 3.47, 0.72, 11.2],
    [9.4, 0.35, 0.58, 2.1, 0.074, 40.0, 122.0, 0.9969, 3.26, 0.87, 11.8],
    [5.5, 0.96, 0.02, 2.5, 0.104, 10.0, 18.0, 0.9980, 3.08, 0.38, 8.5],
    [8.7, 0.41, 0.45, 3.0, 0.091, 25.0, 110.0, 0.9973, 3.18, 0.70, 10.3],
]

print("Feature\t\t\tMutual Information Score")
mi = mutual_info_classif(X_train, y_train, discrete_features=False)
feature_importance_mi = pd.Series(mi, index=feature_cols).sort_values(ascending=False)
print(feature_importance_mi)

print("\n Tuple Test Predictions\n")

for i in range(len(example_features)):
    example_color_code = label_encoder_color.transform([wine_quality_color])[0]

    example_input = np.array(example_features[i] + [example_color_code]).reshape(1, -1)
    example_input = pt.transform(example_input)
    example_input = scaler.transform(example_input)

    predicted_class = model.predict(example_input)
    predicted_label = label_encoder_quality.inverse_transform(predicted_class)[0]
    predicted_color = label_encoder_color.inverse_transform([example_color_code])[0]

    # print(f"\nWine sample -> manual prediction:")
    print(f"Features --> Input (Scaled): {example_input}, Color: {wine_quality_color},")
    print(f"Prediction Of Model -> Quality: {predicted_label} (Color: {predicted_color})\n")