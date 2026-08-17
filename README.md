# Machine Learning Assignment 2

## a. Problem statement
To build, evaluate, and deploy multiple classification models using an interactive Streamlit application to predict target classes based on dataset features.

## b. Dataset description
* **Dataset Name:** Heart Disease UCI dataset
* **Source:** Kaggle
* **Number of Features:** 14 features 
* **Number of Instances:** 1025 instances
  
## c. Github Repository Link
https://github.com/harshadagujar1991/ML_Assignment_2

## d. Models used
### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.7951 | 0.8787 | 0.8023 | 0.7951 | 0.7938 | 0.5973 |
| Decision Tree | 0.9854 | 0.9854 | 0.9858 | 0.9854 | 0.9854 | 0.9712 |
| kNN | 0.8341 | 0.9486 | 0.8387 | 0.8341 | 0.8335 | 0.6727 |
| Naive Bayes | 0.8000 | 0.8706 | 0.8105 | 0.8000 | 0.7982 | 0.6102 |
| Random Forest (Ensemble) | 0.9854 | 1.0000 | 0.9858 | 0.9854 | 0.9854 | 0.9712 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Acts as a fundamental linear baseline, achieving a solid 79.51% accuracy and reliable convergence. |
| Decision Tree | Performs exceptionally well on this dataset with an accuracy of 98.54%, capturing complex non-linear splits effectively. |
| kNN | Delivers strong performance (83.41% accuracy) and a high AUC score (0.9486) after proper distance scaling. |
| Naive Bayes | Provides consistent probabilistic predictions (80.00% accuracy), though limited slightly by its feature independence assumption. |
| Random Forest (Ensemble) | **Overall Winner.** Reaches top-tier performance with 98.54% accuracy and a flawless AUC of 1.0000, perfectly handling feature interactions and minimizing error variance. |

**Overall Winner for your dataset:** Random Forest (Ensemble)
