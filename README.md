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

6.) Model Selection

7.) Model Training

8.) Model Evaluation

9.) Model Deployment

10.) Monitor and Maintenance

11.) Communication and Reporting

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

Another "preparing" step.  The "cleaned" data is being prepared for proper machine learning analysis. 
