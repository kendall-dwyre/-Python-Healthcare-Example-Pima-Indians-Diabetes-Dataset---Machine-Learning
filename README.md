# Pima Indians Diabetes Prediction - Machine Learning Project

## Overview

This project demonstrates machine learning techniques applied to the **Pima Indians Diabetes Dataset**. The goal is to predict whether a person has diabetes based on specific diagnostic measurements. Various machine learning models were used and compared to determine which performs best for this classification task. The dataset was obtained from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

## Dataset Description

The Pima Indians Diabetes Dataset consists of 768 observations of female patients aged 21 and older of Pima Indian heritage. Each observation includes several health-related measurements and a binary outcome indicating whether the patient was diagnosed with diabetes.

### Features:
- **Pregnancies:** Number of times pregnant.
- **Glucose:** Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
- **Blood Pressure:** Diastolic blood pressure (mm Hg).
- **Skin Thickness:** Triceps skinfold thickness (mm).
- **Insulin:** 2-Hour serum insulin (mu U/ml).
- **BMI:** Body mass index (weight in kg/(height in m)^2).
- **Diabetes Pedigree Function:** A function that scores the likelihood of diabetes based on family history.
- **Age:** Age of the patient.
- **Outcome (Target):** Binary outcome where 1 indicates the patient has diabetes and 0 means no diabetes.

## Approach

### 1. **Data Preprocessing**
- **Loading the Data:** The dataset was loaded using `pandas` for inspection and exploration.
- **Exploratory Data Analysis (EDA):** A quick look at the data using summary statistics and visualizations. A pairplot was created to visualize relationships between the features and the target variable (Outcome).
- **Feature Scaling:** Since some models (e.g., Logistic Regression, SVM) perform better with standardized data, `StandardScaler` was used to scale the features to ensure they are on the same scale.
- **Handling Missing Values:** The dataset does not contain any null values; however, some features may have incorrect values such as 0 for attributes like BMI and blood pressure, which were handled appropriately.

### 2. **Model Selection and Evaluation**
- Several machine learning models were trained and evaluated to predict diabetes. These include:
  - **Logistic Regression**
  - **Decision Tree**
  - **Random Forest**
  - **Gradient Boosting**
  - **Support Vector Machine**
  - **K-Nearest Neighbors**

- **Evaluation Metrics:**
  The models were evaluated using the following metrics:
  - **Accuracy:** Overall, how often the model makes correct predictions.
  - **Precision:** Proportion of positive identifications that were actually correct.
  - **Recall (Sensitivity):** Proportion of actual positives that were identified correctly.
  - **F1 Score:** Harmonic mean of Precision and Recall, providing a balanced metric.

### 3. **Cross-Validation:**
- Cross-validation was implemented on the chosen models to get a more robust evaluation of model performance across different data splits. This helps avoid overfitting and gives a clearer picture of model accuracy.

### 4. **Confusion Matrix:**
- A confusion matrix was used for each model to visualize the true positives, true negatives, false positives, and false negatives. This helps understand model behavior beyond simple accuracy.

## Insights

From this project, several insights were drawn:
- **Decision Tree Model:** This model showed good performance across different metrics and was relatively simple to interpret. However, tuning hyperparameters like tree depth could further improve its performance.
- **Random Forest and Gradient Boosting Models:** These ensemble models performed well and showed promise for real-world applications. They handle overfitting better than single decision trees and provide feature importance, which can help in identifying the most impactful factors for diabetes prediction.
- **Feature Importance:** Features like **Glucose** and **BMI** were found to be the most influential in predicting diabetes, aligning with medical knowledge about diabetes risk factors.
- **Cross-Validation:** Using cross-validation gave us a more reliable estimate of model performance. The Decision Tree model had an average cross-validation accuracy of ~75%, making it the best fit for this problem in this instance.

## Conclusion

This project demonstrates the application of several machine learning algorithms to a real-world healthcare dataset. By comparing models and analyzing evaluation metrics, we gained a better understanding of the factors that influence diabetes diagnosis. 

Further improvements to the project could include:
- **Hyperparameter tuning** to optimize model performance.
- **Feature engineering** to explore interactions between features.
- **Handling potential outliers or erroneous data** in features like insulin levels or skin thickness.

## How to Use

1. **Clone this repository**:
    ```bash
    git clone https://github.com/your-username/Pima-Indians-Diabetes-Prediction.git
    cd Pima-Indians-Diabetes-Prediction
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the notebook** in your local environment or use Google Colab:
    ```bash
    jupyter notebook machineLearning.ipynb
    ```

4. **Explore the results**: The models will be trained and evaluated automatically. You can visualize the performance of each model and inspect the confusion matrices to understand the predictions.

## Acknowledgments

The dataset used in this project is from the UCI Machine Learning Repository and made available through Kaggle: [Pima Indians Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database).

## Contact

For any questions or suggestions, feel free to contact me:
- **Name**: Kendall Dwyre
- **Email**: ksdwyre@gmail.com
- **LinkedIn**: [Kendall Dwyre LinkedIn](https://www.linkedin.com/in/kendall-dwyre/)
- **GitHub**: [Kendall Dwyre GitHub](https://github.com/kendall-dwyre/)
