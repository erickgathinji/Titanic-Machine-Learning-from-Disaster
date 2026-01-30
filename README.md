# Titanic Survival Prediction: A Comparative Study of Machine Learning Architectures

This project implements a comprehensive machine learning pipeline to predict passenger survival for the [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/overview). The study spans from linear models to ensemble methods and deep learning models, focusing on hyperparameter optimization to improve performance.

## Tech Stack & Environment
- **Language:** Python 3.12.9
- **Primary Libraries:** 
  - [Scikit-Learn](https://scikit-learn.org) for preprocessing (`StandardScaler`, `OneHotEncoder`) and traditional ML (`LogisticRegression`, `AdaBoostClassifier`, `XGBClassifier`, `SVC`, `KNeighborsClassifier`, `RandomForestClassifier`) 
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

### 5.  Final Kaggle Leaderboard Standings
After exhaustive hyperparameter tuning using `GridSearchCV`, the models were ranked by their actual performance on the Kaggle hidden test set:

| Rank | Model | Kaggle Score |
| :--- | :--- | :--- |
| **1st** | **Logistic Regression** | **0.79425** |
| **2nd** | **Keras Neural Network**| **0.79186** |
| **3rd** | **SVC (RBF Kernel)** | **0.77751** |
| **3rd** | **Voting Ensemble** | **0.77751** |
| **5th** | **AdaBoost** | **0.77511** |
| **6th** | **XGBoost** | **0.77272** |
| **7th** | **Random Forest** | **0.76794** |
| **8th** | **KNN** | **0.75358** |

### 6. Deliverables & Submissions
*   `titanic_disaster_analysis.ipynb`: The primary notebook containing all EDA, Feature Engineering, and Model Training.
*   `vc_submission.csv`: Final predictions for the **Voting Ensemble** (`VotingClassifier`).
*   `tf_submission.csv`: Final predictions for the **Keras Sequential Neural Network**.
*   `individual_trad_model_submissions` folder stores all the predictions for each and every other model.

## Conclusion
**Logistic Regression** outperformed more complex models. On a small, tabular dataset like the Titanic, high-complexity models tended to overfit to noise, while the linear baseline captured the patterns most effectively.

