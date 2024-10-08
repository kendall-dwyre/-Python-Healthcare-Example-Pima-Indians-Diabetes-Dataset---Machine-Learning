# -*- coding: utf-8 -*-
"""
Machine Learning on Pima Indians Diabetes Dataset
This notebook demonstrates machine learning skills using the 'diabetes.csv' dataset.
"""

# Importing Packages and Data File
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
diabetes = pd.read_csv('diabetes.csv')

# Quick look at the first few rows of the dataset
print(diabetes.head())

# Data overview
print(diabetes.describe())
print(diabetes.dtypes)

# Checking for null values
print("Missing values per column:\n", diabetes.isnull().sum())

# Pairplot for exploratory data analysis
sns.pairplot(diabetes, hue='Outcome')
plt.show()

# Feature scaling using StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(diabetes.drop(columns=['Outcome']))
scaled_diabetes = pd.DataFrame(scaled_features, columns=diabetes.columns[:-1])
scaled_diabetes['Outcome'] = diabetes['Outcome']

# Defining features (X) and target (y)
X = scaled_diabetes.drop(columns=['Outcome'])
y = scaled_diabetes['Outcome']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model dictionary for ease of use
models = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# Function to evaluate models with accuracy, precision, recall, and F1 score
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Display confusion matrix
    print(f"Confusion Matrix for {model.__class__.__name__}:\n", confusion_matrix(y_test, y_pred))
    
    return accuracy, precision, recall, f1

# Dictionary to store results
results = {}

# Evaluate each model
for model_name, model in models.items():
    accuracy, precision, recall, f1 = evaluate_model(model, X_train, y_train, X_test, y_test)
    results[model_name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}

# Display results in a DataFrame
results_df = pd.DataFrame(results).T
print("Model Evaluation Results:")
print(results_df)

# Visualizing model performance
results_df.plot(kind='bar', figsize=(12, 8), title="Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.show()

# Cross-validation for the chosen model (Decision Tree as an example)
dt_model = DecisionTreeClassifier()
cv_scores = cross_val_score(dt_model, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy for Decision Tree: {cv_scores.mean():.4f}")
