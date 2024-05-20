import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt

"""One of the first things to do when you get a file is to prep the data for analysis.  For instance, we are looking for bad data, nulls, etc."""

diabetes = pd.read_csv('diabetes.csv')
print(diabetes.head())

"""The code below is some exploratory analysis to see what we need to do to prep the data correctly."""

len(diabetes) # 768 (this is the length of our file)
diabetes.dtypes # This is important for us to see the different data types of the variables
diabetes.describe() # This gives us a summary of the data

"""Are there nulls?"""

diabetes.isnull().sum() # No nulls

"""In the event that there were nulls, which there are not, we could run some code like what is below.  It will be commented out for the sake of this project."""

#diabetes_clean = diabetes.dropna()
#len(diabetes_clean)
#diabetes_clean.isnull().sum # Rerunning to check for nulls

"""Now that we have looked at the data and took necessary measures to make sure it is clean, we can move forward.  The next step is splitting the data into a training and test step.  However, we need to identify the target / label attribute that we want to work with."""

# One-hot encoding for the 'category' feature
#encoder = OneHotEncoder(sparse = False)
#encoded_categories = encoder.fit_transform(diabetes[['category']])
#encoded_diabetes = pd.DataFrame(encoded_categories, columns = encoder.get_feature_names_out(['category']))

# Concatenate the encoded categories back to the original DataFrame
#df = pd.concat([diabetes.drop('category', axis=1), encoded_diabetes], axis=1)

"""Next, it's important for us to identify what it is that we want to "target", and if there are any variables that provide unessary / irrelevant information that can be dropped entirely.

"Outcome" seems to be a good variable to use as a target.  This variable is predicting whether people have diabetes based on the other factors.  The goal of our machine learning is to look at the connection between these variables and the target value.  
"""

# Dropping our target variable
X = diabetes.drop(columns=['Outcome'])
y = diabetes['Outcome']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# We can look at some of the values
#print(X_train)
print(X_train.describe()) # Descriptive statistics
#print(y_train)
#print(X_test)
#print(y_test)

"""Let's look at some visualizations."""

sns.pairplot(diabetes, hue = 'Outcome')
plt.show()

"""Logistic Regression."""

# Create a logistic regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

"""Decision Tree."""

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Prediction
y_pred_train = dt_model.predict(X_train)
y_pred_test = dt_model.predict(X_test)

# Accuracy
accuracy_test = accuracy_score(y_test, y_pred)
accuracy_train = accuracy_score(y_train, y_pred_train)
print("Test Accuracy:", accuracy_test)
print("Train Accuracy:", accuracy_train)

"""Random Forest."""

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Predictions
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# Accuracy
accuracy_test = accuracy_score(y_test, y_pred)
accuracy_train = accuracy_score(y_train, y_pred_train)
print("Test Accuracy:", accuracy_test)
print("Train Accuracy:", accuracy_train)

"""These are helpful parameters as we look at the different evaluators:

1. Accuracy

Good is greater than 90%, acceptable 70% - 90%, and poor is less than 70%.

2. Precision

What is actually correct.

Good is greater than 90%, acceptable 70% - 90%, and poor is less than 70%.

3. Recall (Sensitivity)

Proportion of actual positives that are correctly identified.

Good is greater than 90%, acceptable is 70% - 90%, and poor 70%.

4. F1 Score

Harmonic mean of precision and recall and provides a balance between the two.  

Good is greater than 90%, acceptable 70% - 90%, and poor is less than 70%.




"""

models = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# Function to evaluate models
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

# Evaluate each model and store results
results = {}
for model_name, model in models.items():
    accuracy, precision, recall, f1 = evaluate_model(model, X_train, y_train, X_test, y_test)
    results[model_name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}

# Display the results
results_df = pd.DataFrame(results).T
print(results_df)

"""The model that is chosen will be the Decision Tree.  It seems to have the best values across the different categories."""
