import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load your dataset
data = pd.read_csv('pattern1.csv')

# Define your target column and features
target_column = 'Cases'  # Replace with your actual target column name ('Cases', 'Deaths', etc.)
if target_column in data.columns:
    X = data.drop(target_column, axis=1)
    y = data[target_column]
else:
    raise KeyError(f'{target_column} not found in the dataset. Please check your column names.')

# Drop non-numeric columns for simplicity (day, time)
X.drop(['day', 'time'], axis=1, inplace=True)

# Handle categorical columns using OneHotEncoder
categorical_cols = ['country', 'continent']  # Adjust based on your dataset
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# One-hot encode categorical columns
onehot_encoder = OneHotEncoder(drop='first', sparse=False)
X_encoded = onehot_encoder.fit_transform(X[categorical_cols])

# Create column names for the one-hot encoded columns
cat_cols_encoded = onehot_encoder.get_feature_names_out(categorical_cols)
X_encoded_df = pd.DataFrame(X_encoded, columns=cat_cols_encoded)

# Concatenate encoded categorical columns with numerical columns
X_processed = pd.concat([X_encoded_df, X[numeric_cols]], axis=1)

# Convert target to labels for multiclass classification
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)

# Create a Decision Tree classifier
dt_model = DecisionTreeClassifier(random_state=42)

# Train the Decision Tree classifier
dt_model.fit(X_train, y_train)

# Predictions on the test set
y_pred = dt_model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
conf_matrix = confusion_matrix(y_test, y_pred)

# Print metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1:.2f}')
print('Confusion Matrix:')
print(conf_matrix)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()
