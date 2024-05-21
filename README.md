# Machine-Learning
The purpose of this repository is to demonstrate machine learning skills by going through a dataset from online. 

The dataset that I am working with is called "Pima Indians Diabetes Database" from Kaggle.  You can find a link to it here:
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?resource=download

Explanation about the dataset from Kaggle:

"This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage."

The objective of this repository is to demonstrate some machine learning skills; however, I will be going through a typical work flow that a data scientist would go through when working with new data.  The steps are as follows:

1.) Define the Problem

2.) Collect Data

3.) Data Cleaning

4.) Explortory Analysis

5.) Feature Engineering 

6.) Model Selection / Model Training

7.) Model Evaluation

8.) Model Deployment

9.) Monitor and Maintenance

10.) Communication and Reporting

Given that I gathered this data from Kaggle, steps 1.) and 2.) will be skipped.  If we were starting scratch, we would begin by asking a question and gathering appropriate data.  

**Data Cleaning**

We are looking for anything that may throw an issue - such as Null Values, NA's, or simply put bad data.  In our case, it seems to be that this dataset was prepared before loading it onto Kaggle, but it's still good to go through the process and make sure nothing is overlooked.  

Below is an example of some code that I wrote to ascertain if there are any null values (which there are none):

	diabetes.isnull().sum()

If there were null values that would require further thought about what to do with them.  In our case, we can move forward to the next step.  

**Exploratory Analysis**

Any data scientist will know that pictures can sometimes speak much louder than words.  It can be easy to get lost in all the data, so pictures are important tools to help us visually understand what is going on.  We can use different plots to see distributions (is the data "normalized"?).  We can also use plots to help us see relationships between different variables (positive / negative / neutral relationship).  Below is some code that I used to create visualizations: 

	sns.pairplot(diabetes, hue = 'Outcome')
	plt.show()

This code shows the relatationship between a pair of variables in the dataset (which I named 'diabetes'). 

**Feature Engineering**

Another "preparing" step.  The "cleaned" data is being prepared for proper machine learning analysis.  Below is some code that I used for this step:  

	scaler = StandardScaler()
	scaled_features = scaler.fit_transform(diabetes.drop(columns=['Outcome']))
	scaled_diabetes = pd.DataFrame(scaled_features, columns=diabetes.columns[:-1])
	scaled_diabetes['Outcome'] = diabetes['Outcome']

**Model Selection / Model Training**

In this step, the dataset is split into a training and test set and then it is tested against various model types.  Below is code that was written to split the data set into training and test sets.  Take note that it is using the "prepared" data from the step above.  

	# Dropping our target variable
	X = scaled_diabetes.drop(columns=['Outcome'])
	y = scaled_diabetes['Outcome']
	
	# Splitting the data into training and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
	
	# We can look at some of the values
	#print(X_train)
	print(X_train.describe()) # Descriptive statistics
	#print(y_train)
	#print(X_test)
	#print(y_test)

Now that the data is split into training and test sets we can look at various model types. 

Logistic Regression 

	# Create a logistic regression model
	model = LogisticRegression()
	
	# Train the model on the training data
	model.fit(X_train, y_train)
	
	# Make predictions on the test data
	y_pred = model.predict(X_test)
	
	# Evaluate the model
	accuracy = accuracy_score(y_test, y_pred)
	print("Accuracy:", accuracy)

Decision Tree

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

Random Forest

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

 This code below shows several different model types (including the example ones above) in a more consise cell of data:

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

Below is the output from the cell that is ran above: 

	                         Accuracy  Precision    Recall  F1 Score
	Logistic Regression     0.753247   0.649123  0.672727  0.660714
	Decision Tree           0.727273   0.600000  0.709091  0.650000
	Random Forest           0.720779   0.607143  0.618182  0.612613
	Gradient Boosting       0.740260   0.627119  0.672727  0.649123
	Support Vector Machine  0.727273   0.632653  0.563636  0.596154
	K-Nearest Neighbors     0.688312   0.574468  0.490909  0.529412
