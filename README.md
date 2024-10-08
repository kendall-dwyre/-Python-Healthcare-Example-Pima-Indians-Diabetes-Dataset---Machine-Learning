# Pima Indians Diabetes Prediction - Machine Learning Project

## Overview

This project demonstrates machine learning techniques applied to the **Pima Indians Diabetes Dataset**. The goal is to predict whether a person has diabetes based on specific diagnostic measurements. Various machine learning models were trained, evaluated, and compared to determine which performed best in predicting diabetes. The dataset was obtained from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

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

## Diabetes Insights Based on the Data and Models

From the analysis performed using the Pima Indians Diabetes Dataset, several key insights about diabetes and its risk factors were observed. The models gave us a better understanding of which features play a crucial role in predicting the likelihood of diabetes.

### 1. **Glucose Levels Are the Most Important Predictor**
- **Glucose** was consistently shown to be the most significant feature across all the models, including Decision Trees and Random Forests, which provide feature importance metrics.
- This aligns with medical knowledge that higher glucose levels are a critical indicator of diabetes. Patients with elevated glucose levels, particularly those above the normal range during an oral glucose tolerance test, are more likely to have diabetes.

### 2. **BMI and Age as Contributing Factors**
- **BMI (Body Mass Index)** was another key predictor identified by the models. Higher BMI values were associated with an increased likelihood of diabetes, reinforcing the well-documented relationship between obesity and diabetes risk.
- **Age** also plays a role in diabetes prediction. Older individuals, particularly those above 50, were found to be more at risk. This is consistent with the fact that diabetes, especially Type 2, tends to manifest later in life.

### 3. **The Role of Pregnancies and Insulin**
- The number of **pregnancies** also contributed to the prediction of diabetes, particularly among women who had multiple pregnancies. This insight could relate to conditions like gestational diabetes, where temporary diabetes during pregnancy could later lead to permanent Type 2 diabetes.
- The **insulin** feature was less impactful in the models, possibly due to missing or inconsistent values in the dataset. This might suggest that insulin levels alone are not sufficient to predict diabetes unless combined with other factors.

### 4. **Blood Pressure and Skin Thickness Have Limited Impact**
- Features like **Blood Pressure** and **Skin Thickness** showed a relatively lower contribution to diabetes prediction compared to glucose, BMI, and age. While high blood pressure is often associated with diabetes, it seems that glucose levels and BMI were far more important in identifying patients with diabetes in this particular dataset.

### 5. **Models' Performance in Predicting Diabetes**
- The **Decision Tree** and **Random Forest** models performed the best in terms of accuracy, precision, recall, and F1 score, with Decision Tree having the best cross-validation score (~75% accuracy).
- **Logistic Regression** provided an interpretable model but didn’t perform as well as tree-based models. Its simplicity may make it useful in some cases but lacks the ability to capture complex patterns in the data compared to ensemble models like Random Forest and Gradient Boosting.
- **Gradient Boosting** also showed promise, with strong precision and recall values, making it a robust model for diabetes prediction in this context.

### Key Takeaway:
The most impactful factors for predicting diabetes are **Glucose levels, BMI, and Age**. These features, when combined, provide a reliable indication of diabetes risk, helping models like Decision Trees and Random Forests achieve good accuracy. Insights from this analysis suggest that glucose monitoring and maintaining a healthy BMI are critical in diabetes prevention.

## Conclusion

This project demonstrates the application of several machine learning algorithms to a real-world healthcare dataset. By comparing models and analyzing evaluation metrics, we gained a better understanding of the factors that influence diabetes diagnosis. The insights provided by this project can help medical professionals focus on the most critical health metrics when assessing diabetes risk.

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
