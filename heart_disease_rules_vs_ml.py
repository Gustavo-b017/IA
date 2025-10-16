import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# UCI repository CSV link (hosted on archive.ics.uci.edu)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Column names from UCI documentation
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "target"
]

df = pd.read_csv(url, names=columns)

# Replace missing values ("?") with NaN and drop rows with NaN
df = df.replace("?", pd.NA)
df = df.dropna()
df = df.astype(float)

# Convert target to binary: 0 = no disease, 1 = disease present
df["target"] = (df["target"] > 0).astype(int)

# Rule-based classifier
def rule_based_classifier(row):
    if row["age"] > 50 and row["trestbps"] > 140 and row["chol"] > 240:
        return 1
    else:
        return 0

df["rule_prediction"] = df.apply(rule_based_classifier, axis=1)
print("Rule-Based Accuracy:", accuracy_score(df["target"], df["rule_prediction"]))

# ML model (SVM)
X = df.drop(columns=["target", "rule_prediction"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm_model = SVC(kernel="linear")
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

print("\nMachine Learning (SVM) Results:")
print(classification_report(y_test, y_pred))
print("SVM Accuracy:", accuracy_score(y_test, y_pred))
