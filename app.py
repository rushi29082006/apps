import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# ================= LOAD DATA =================
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

# Log transform skewed income features
X["ApplicantIncome"] = np.log1p(X["ApplicantIncome"])
X["CoapplicantIncome"] = np.log1p(X["CoapplicantIncome"])


# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)


# ================= PIPELINE =================
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf"))
])


# ================= HYPERPARAMETERS =================
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


# ================= EVALUATION =================
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


# ================= DEPENDENCIES OUTPUT =================
print("\nMODEL DEPENDS ON THESE INPUT FEATURES:")
for col in X.columns:
    print("-", col)

print("\nSVM HYPERPARAMETERS USED:")
print(model.named_steps["svm"].get_params())

print("\nSCALER PARAMETERS (mean, std):")
print("Mean:", model.named_steps["scaler"].mean_)
print("Scale:", model.named_steps["scaler"].scale_)

print("\nIMPUTER STRATEGY:")
print(model.named_steps["imputer"].strategy)


# ================= SAVE =================
joblib.dump(model, "loan_model.pkl")
joblib.dump(model.named_steps["scaler"], "scaler.pkl")
