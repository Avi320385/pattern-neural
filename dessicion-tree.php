import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
file_path = 'pattern2.csv'
df = pd.read_csv(file_path)

# Convert 'Deaths' column to binary classification
df['Deaths_binary'] = (df['Deaths'] > 0).astype(int)

# Select features and target
selected_features = ['population', 'Cases', 'Deaths']
X = df[selected_features]
y = df['Deaths_binary']  # Target column for binary classification

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (80:20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define and train the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predict on the test set
y_pred_dt = dt_model.predict(X_test)

# Print Classification Report for Decision Tree
print("Classification Report for Decision Tree:\n", classification_report(y_test, y_pred_dt))

# Plot Confusion Matrix for Test Data
plt.figure(figsize=(8, 6))
cm_dt = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Blues")
plt.title('Decision Tree Confusion Matrix (Test Data)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Precision, Recall, F1-Score Plot
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred_dt, average='binary')
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1_score)

# Feature Importance Plot
feature_importances = dt_model.feature_importances_
features = selected_features

plt.figure(figsize=(10, 6))
plt.barh(features, feature_importances, color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Decision Tree')
plt.show()
