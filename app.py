import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Load dataset
data = pd.read_csv("dataset.csv")

# Remove missing values
data = data.dropna()

# Encode target column
data["Loan_Status"] = data["Loan_Status"].map({"N": 0, "Y": 1})

# Handle special category
data["Dependents"] = data["Dependents"].replace("3+", 4)

# Encode categorical features
data.replace({
    "Gender": {"Male": 1, "Female": 0},
    "Married": {"Yes": 1, "No": 0},
    "Education": {"Graduate": 1, "Not Graduate": 0},
    "Self_Employed": {"Yes": 1, "No": 0},
    "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2}
}, inplace=True)

# Split features and label
X = data.drop(columns=["Loan_ID", "Loan_Status"])
y = data["Loan_Status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=2
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "loan_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully")
