import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
data = pd.read_csv("expenses.csv")

# Features & Label
X = data.drop("Risk_Level", axis=1)
y = data["Risk_Level"]

# Train test split of data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
# joblib.dump(model, "finance_model.pkl")

#For Report
y_pred = model.predict(X_test)
print("Model Trained Successfully!") 
accuracy = accuracy_score(y_test, y_pred)
print("\n Model Accuracy:", accuracy * 100, "%\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

