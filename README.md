Credit Card Default Prediction
📌 Abstract
Predicting credit default is crucial for managing risk in consumer lending. This project explores machine learning techniques to enhance the accuracy of credit default prediction, aiding lenders in making informed decisions. We applied various models, including ensemble and traditional classifiers, to build interpretable and robust predictive systems.

🔍 Introduction
In the financial domain, identifying customers likely to default on credit payments is a key task. With the growth of data availability and ML capabilities, there's room to enhance prediction accuracy beyond traditional statistical methods. This project leverages machine learning to classify credit card clients as defaulters or non-defaulters using historical data.

📚 Literature Survey
Zhang et al.: Emphasized ensemble methods like Random Forest and Gradient Boosting with attention to feature selection and tuning.

Ong et al.: Compared logistic regression with machine learning models, favoring ML for accuracy on large datasets.

Liu et al.: Explored deep learning for credit risk, noting high performance but higher computational cost.

This Work: Combines classical and modern ML approaches, focusing on interpretability and real-world deployment challenges like data imbalance and model complexity.

📊 Dataset
Source: UCI Machine Learning Repository

Dataset: Credit Card Default Payment (Taiwan)

Samples: 30,000

Features: 25 raw features, expanded to 83 post-encoding

Target: default.payment.next.month (1 = default, 0 = non-default)

🧾 Key Attributes:
LIMIT_BAL: Credit limit

SEX, EDUCATION, MARRIAGE: Categorical demographics

PAY_0 to PAY_6: Monthly payment history

BILL_AMT1 to BILL_AMT6: Last six months' bills

PAY_AMT1 to PAY_AMT6: Last six months' payments

🧹 Data Preprocessing
Imbalance Handling: Noted skew toward non-defaulters

Missing Values: None

Encoding: One-hot encoding of categorical variables (SEX, EDUCATION, etc.)

Scaling: StandardScaler used for normalization (robust to outliers)

Train-Test Split: 70% train / 30% test

Train: 16324 non-defaulters / 4676 defaulters

Test: 7040 non-defaulters / 1960 defaulters

🔻 PCA (Principal Component Analysis)
Goal: Dimensionality reduction from 83 to 40 components

Explained Variance: 40 components retain nearly 100%

Result: Efficient training and reduced complexity

🧠 Methodology & Models
🔧 Models Applied:
Logistic Regression

Decision Tree

Random Forest Classifier

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Gaussian Naive Bayes

🔎 Evaluation Metrics:
Accuracy

Precision

Recall

F1-Score

ROC AUC

📌 Hyperparameter Tuning:
SVM: C=1 using grid search

KNN: n_neighbors=500, metric=manhattan, weights=uniform

📌 Conclusion
This project shows that advanced machine learning models, including ensemble and distance-based classifiers, can significantly improve credit default prediction. While simpler models like Logistic Regression offer interpretability, models like Random Forest and KNN demonstrated superior performance with proper tuning and preprocessing.

🛠️ Technologies Used
Python (NumPy, Pandas, scikit-learn)

Jupyter Notebooks

Matplotlib / Seaborn

StandardScaler, PCA

SVM, RandomForest, KNN, Logistic Regression, etc.

📬 Acknowledgments
Dataset provided by the UCI Machine Learning Repository

Inspired by prior academic work on credit risk modeling
