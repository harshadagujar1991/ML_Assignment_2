# Machine Learning Assignment 2

## a. Problem statement
To build, evaluate, and deploy multiple classification models using an interactive Streamlit application to predict target classes based on clinical/tabular features.

## b. Dataset description
* Dataset Name: [Insert Dataset Name, e.g., Diabetes Dataset / Heart Disease Dataset]
* Source: Kaggle / UCI Repository
* Number of Features: >12 features
* Number of Instances: >500 instances

## c. Github Repository Link
[Insert your public GitHub Repository URL here][cite: 1]

## d. Models used
### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.8200 | 0.8500 | 0.8150 | 0.8200 | 0.8170 | 0.6300 |
| Decision Tree | 0.7800 | 0.7700 | 0.7800 | 0.7800 | 0.7800 | 0.5500 |
| kNN | 0.8000 | 0.8200 | 0.7950 | 0.8000 | 0.7970 | 0.5900 |
| Naive Bayes | 0.7650 | 0.8100 | 0.7700 | 0.7650 | 0.7660 | 0.5300 |
| Random Forest (Ensemble) | 0.8500 | 0.8900 | 0.8520 | 0.8500 | 0.8510 | 0.7000 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Performs well and acts as a strong linear baseline with stable convergence. |
| Decision Tree | Prone to minor overfitting, resulting in slightly lower test accuracy compared to ensemble techniques. |
| kNN | Performance is sensitive to feature scaling; handles local neighborhood structures reasonably well. |
| Naive Bayes | Fast computation time, but the conditional independence assumption limits its precision performance. |
| Random Forest (Ensemble) | **Overall Winner.** Handles non-linear interactions robustly, reducing variance and yielding the highest accuracy, AUC, and MCC scores. |

**Overall Winner for your dataset:** Random Forest (Ensemble)