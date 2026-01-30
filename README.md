# Titanic Survival Prediction: A Comparative Study of Machine Learning Architectures

This project implements a comprehensive machine learning pipeline to predict passenger survival for the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic/overview). The study spans from traditional linear models to modern deep learning architectures, focusing on hyperparameter optimization and ensemble methods.

## Tech Stack & Environment
- **Language:** Python 3.12.9
- **Primary Libraries:** 
  - [Scikit-Learn](https://scikit-learn.org) for preprocessing (StandardScaler, OneHotEncoder) and traditional ML (LogisticRegression, XGBClassifier, SVC) 
  - [TensorFlow / Keras](https://www.tensorflow.org) for neural network implementation
  - [Pandas](https://pandas.pydata.org) & [NumPy](https://numpy.org) for data manipulation

## Model Development Lifecycle

### 1. Data Preprocessing & Engineering
A robust `ColumnTransformer` was implemented to handle heterogeneous data:
*   **Numerical Features:** Standardized using `StandardScaler` to ensure convergence for distance-based models.
*   **Categorical Features:** Encoded via `OneHotEncoder` with handling for unknown categories.
*   **Pipeline Architecture:** All preprocessing steps were integrated into pipelines to prevent data leakage during cross-validation.

### 2. Traditional Machine Learning (Tournament Phase)
I conducted an exhaustive search for optimal hyperparameters using `GridSearchCV` across several algorithms:
*   **Logistic Regression:** Optimized using `ElasticNet` penalties and the `SAGA` solver.
*   **Support Vector Machines (SVM):** Explored non-linear boundaries using the `RBF` kernel.
*   **Random Forest:** Tuned for depth and leaf stability to mitigate overfitting.
*   **AdaBoost & XGBoost:** Utilized gradient boosting to focus on difficult-to-classify demographic samples.

### 3. Ensemble Strategy
To improve model generalization, a **Soft Voting Classifier** was developed. This ensemble combined the probabilistic predictions of the top-performing Logistic Regression, SVM, and XGBoost models, leveraging the "wisdom of the crowds" to stabilize the decision boundary.

### 4. Deep Learning Implementation
The project concluded with a **Keras Sequential Neural Network**:
*   **Architecture:** A multi-layer perceptron (MLP) using the `Swish` activation function.
*   **Regularization:** Integrated `Dropout` layers and `EarlyStopping` callbacks to prevent overfitting on the relatively small dataset.

### 5. Deliverables & Submissions
*   `titanic_disaster_analysis.ipynb`: The primary notebook containing all EDA, Feature Engineering, and Model Training.
*   `submission.csv`: Final output for the **Traditional ML Ensemble** (Voting Classifier), which achieved a leaderboard score of **0.78708**.
*   `tf_submission.csv`: Final output for the **Keras Sequential Neural Network**, which achieved the project-high score of **0.79186**.

## Conclusion
The experimental results demonstrate that while ensemble methods like Voting Classifiers offer stability, Neural Network architectures can capture unique non-linearities in passenger demographics. The systematic tuning of regularization parameters (C, Gamma, and Dropout) proved essential in navigating the bias-variance tradeoff inherent in this tabular classification task.
