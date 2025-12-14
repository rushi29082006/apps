import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


df = pd.read_csv("dataset.csv")

df["Loan_Status"] = df["Loan_Status"].map({"N": 0, "Y": 1})
df["Dependents"] = df["Dependents"].replace("3+", 4)

df.replace({
    "Gender": {"Male": 1, "Female": 0},
    "Married": {"Yes": 1, "No": 0},
    "Education": {"Graduate": 1, "Not Graduate": 0},
    "Self_Employed": {"Yes": 1, "No": 0},
    "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2}
}, inplace=True)

X = df.drop(columns=["Loan_ID", "Loan_Status"])
y = df["Loan_Status"]

X["ApplicantIncome"] = np.log1p(X["ApplicantIncome"])
X["CoapplicantIncome"] = np.log1p(X["CoapplicantIncome"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf"))
])

param_grid = {
    "svm__C": [1, 10],
    "svm__gamma": [0.1]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

model = grid.best_estimator_

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nModel depends on these features:")
print(list(X.columns))

print("\nSVM hyperparameters:")
print(model.named_steps["svm"].get_params())

joblib.dump(model, "loan_model.pkl")
joblib.dump(model.named_steps["scaler"], "scaler.pkl")

print("\nModel saved successfully")
