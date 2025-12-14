import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


df = pd.read_csv("dataset.csv")

df["Loan_Status"] = df["Loan_Status"].map({"N": 1, "Y": 0})  # 1 = default risk
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
    ("lr", LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    ))
])

pipeline.fit(X_train, y_train)

pd_scores = pipeline.predict_proba(X_test)[:, 1]
print("ROC AUC:", roc_auc_score(y_test, pd_scores))

joblib.dump(pipeline, "loan_model.pkl")
