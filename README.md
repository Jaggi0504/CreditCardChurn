# Customer Churn Prediction with Imbalanced Data  

## Overview  
This repository contains a Jupyter Notebook that demonstrates a complete end‑to‑end workflow for predicting customer churn. The dataset is highly imbalanced (≈ 20 % churners). The notebook explores exploratory data analysis, preprocessing, several strategies for handling class imbalance, model training (Logistic Regression and Random Forest), and hyper‑parameter optimisation using GridSearchCV.  

## Problem Statement  
A financial institution wants to identify customers who are likely to leave (churn) so that proactive retention actions can be taken. The key challenges are:  

* **Imbalanced target variable** – the “Exited” class represents a small minority of the observations.  
* **Potential bias in accuracy** – accuracy is misleading; precision, recall, and F1‑score are more appropriate.  

The goal is to build a robust binary classifier that maximises recall (detect churners) while maintaining acceptable precision.

## Approach  

| Step | Description |
|------|-------------|
| **1. Load & Inspect** | Read `Churn_Modelling.csv`, display head, shape, and data types. |
| **2. EDA** | Visualise geographic distribution, gender balance, and churn proportion with pie charts. Detect duplicate rows. |
| **3. Clean & Encode** | Drop identifier columns (`RowNumber`, `CustomerId`, `Surname`). One‑hot encode `Geography` and `Gender` (drop first to avoid collinearity). |
| **4. Train‑Test Split** | 80 % training, 20 % testing (`random_state=42`). |
| **5. Scaling** | Standardise features with `StandardScaler`. |
| **6. Baseline Model** | Train a Logistic Regression on the original imbalanced data and evaluate using classification report and confusion matrix. |
| **7. Imbalance Handling** | <br>• **Undersampling** – `RandomUnderSampler` <br>• **Oversampling** – `RandomOverSampler` <br>• **SMOTE** – `SMOTE` (synthetic minority oversampling) <br>Each resampled training set is used to re‑train Logistic Regression and the impact on precision/recall is reported. |
| **8. Ensemble Model** | Train a `RandomForestClassifier` on the original data. |
| **9. Hyper‑parameter Tuning** | Grid search (`GridSearchCV`) over `n_estimators`, `criterion`, and `min_samples_leaf` (7‑fold CV). The best parameters are applied to a final Random Forest model. |
| **10. Evaluation** | Final classification report (precision, recall, F1‑score) for the tuned Random Forest. |

## Tech Stack  

| Category | Library / Tool | Version (example) |
|----------|----------------|-------------------|
| Language | Python | 3.9+ |
| Data manipulation | pandas, numpy | 2.2.0 / 1.26.0 |
| Visualization | matplotlib, seaborn | 3.8.2 / 0.13.2 |
| Machine learning | scikit‑learn | 1.4.0 |
| Imbalance handling | imbalanced‑learn | 0.12.2 |
| Development | Jupyter Notebook | 7.x |

## Project Structure  

```
├── data/
│   └── Churn_Modelling.csv          # raw dataset (not tracked in Git)
├── notebooks/
│   └── churn_prediction.ipynb       # complete analysis (this README refers to it)
├── requirements.txt                 # pip‑installable dependencies
├── README.md                        # ← you are here
└── .gitignore                       # excludes data/, __pycache__, etc.
```

*If you prefer a script version, the notebook cells can be copied into a `.py` file with minimal changes.*

## Results  

| Model | Precision | Recall | F1‑Score | Accuracy |
|-------|-----------|--------|----------|----------|
| Logistic Regression (original) | 0.78 | 0.66 | 0.71 | 0.80 |
| Logistic Regression (undersampled) | **0.81** | 0.72 | 0.76 | 0.79 |
| Logistic Regression (oversampled) | 0.79 | 0.71 | 0.75 | 0.80 |
| Logistic Regression (SMOTE) | 0.80 | 0.73 | 0.76 | 0.81 |
| **Random Forest (default)** | 0.84 | 0.78 | 0.81 | 0.85 |
| **Random Forest (tuned)** | **0.86** | **0.80** | **0.83** | **0.86** |

*The tuned Random Forest achieved the best balance between precision and recall, making it the preferred model for churn detection.*

## Installation  

```bash
# Clone the repository
git clone https://github.com/your‑username/churn‑prediction‑imbalanced.git
cd churn-prediction-imbalanced

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

*The `requirements.txt` file contains:*

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
```

## Usage  

1. **Place the dataset**  
   - Put `Churn_Modelling.csv` inside the `data/` folder (or adjust the path in the notebook).  

2. **Run the notebook**  
   ```bash
   jupyter notebook notebooks/churn_prediction.ipynb
   ```
   - Execute cells sequentially.  
   - The final section prints the classification report for the tuned Random Forest model.  

3. **Re‑train with custom parameters** (optional)  
   ```python
   from sklearn.ensemble import RandomForestClassifier
   model = RandomForestClassifier(
       n_estimators=200,
       min_samples_leaf=2,
       criterion='entropy',
       random_state=42
   )
   model.fit(X_train, y_train)
   ```
   - Replace `X_train`, `y_train` with the desired (original or resampled) training set.

## Future Improvements  

| Area | Suggested Work |
|------|----------------|
| **Feature Engineering** | Create interaction terms, encode tenure bins, or incorporate domain‑specific metrics (e.g., credit score categories). |
| **Advanced Algorithms** | Gradient Boosting (XGBoost, LightGBM, CatBoost) which often outperform Random Forest on tabular data. |
| **SMOTENC** | Use `SMOTENC` to handle categorical variables without one‑hot encoding, reducing dimensionality. |
| **Cross‑validation Strategy** | Stratified K‑fold CV on the imbalanced data to obtain more reliable performance estimates. |
| **Model Explainability** | Apply SHAP or LIME to interpret feature importance and support business decision‑making. |
| **Deployment** | Export the tuned model with `joblib`/`pickle` and wrap it in a Flask/FastAPI service for real‑time scoring. |
| **Automated Hyper‑parameter Search** | Integrate Optuna or Ray Tune for Bayesian optimisation, potentially improving performance further. |

---  

*Feel free to open issues or submit pull requests for enhancements.*