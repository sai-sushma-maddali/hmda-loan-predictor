# HMDA Loan Decision Predictor

**Live App:** [https://hmda-loan-predictor.streamlit.app/](https://hmda-loan-predictor.streamlit.app/)

Mortgage underwriting relies on the evaluation of numerous applicant, loan, and socioeconomic variables, making the decision process inherently complex and susceptible to inconsistencies or bias. Mississippi, characterized by pronounced demographic and economic variation, presents a meaningful case study for examining the determinants of mortgage loan approval. The Home Mortgage Disclosure Act (HMDA) dataset provides a comprehensive foundation for such analysis, yet its scale and heterogeneity pose challenges for traditional manual review. This project addresses these issues by applying statistical methods and machine learning models to identify significant predictors of loan outcomes and to evaluate potential disparities. The resulting insights contribute to improved decision-making, enhanced transparency, and stronger regulatory oversight in mortgage lending.

---

## Features

* **Loan Approval Prediction:** Random Forest model achieving 85% validation accuracy and 0.9155 ROC-AUC.
* **Interactive Web App:** Users input applicant, property, and loan details to generate approval predictions.
* **Data Analysis:** Exploratory analysis, t-tests, chi-square tests, and feature engineering to identify drivers of lending outcomes.
* **Dashboard Insights:** Tableau dashboard visualizing approval rates by income, race, property type, county, and more.

---

## How It Works

1. HMDA data filtered to conventional home-purchase loans with approved/denied outcomes.
2. Data cleaned, encoded, scaled, and transformed using PCA (23 components explaining ~90% variance).
3. Several ML models evaluated; Random Forest selected as best performer.
4. Streamlit app loads the trained model and preprocessing objects for real-time predictions.

---

## Run Locally

```bash
git clone https://github.com/<your-username>/hmda-loan-predictor.git
cd hmda-loan-predictor/app
pip install -r requirements.txt
streamlit run main.py
```

---

## Repository Structure

```
app/
  main.py                    # Streamlit interface
  targetPrediction.py        # Prediction logic
  dataPrep.py                # Data preparation utilities
  dataProcessing.py          # Encoding, scaling, PCA
  best_rf_model.pkl          # Trained Random Forest model
  pca_obj.pkl                # PCA transformer
  one_hot_encoder.pkl        # One-hot encoder
  standard_scalar_obj.pkl    # Standard scaler
  target_label_encoder.pkl   # Label encoder
  county_census_data.json    # County → census tract mapping
  info.json
  requirements.txt
```

---

## Machine Learning Model Evaluation Results

Multiple supervised learning models were trained and evaluated on the processed HMDA dataset. The **Random Forest Classifier (GridSearch #2)** emerged as the best-performing model based on validation accuracy and ROC-AUC.

### **Model Performance Summary**

| Model                             | Train Accuracy | Validation Accuracy | ROC-AUC    | Notes                       |
| --------------------------------- | -------------- | ------------------- | ---------- | --------------------------- |
| **Random Forest (GridSearch #2)** | **0.95**       | **0.85**            | **0.9155** | Best overall performance    |
| Random Forest (GridSearch #1)     | 0.95           | 0.84                | 0.914      | Slight overfitting          |
| XGBoost                           | 1.00           | 0.84                | 0.915      | Strong overfitting          |
| SVM (RBF Kernel)                  | 0.83           | 0.83                | 0.9121     | Best generalization, stable |
| Decision Tree (1)                 | 0.81           | 0.81                | 0.8625     | Baseline model              |
| Decision Tree (2)                 | 0.88           | 0.82                | 0.8402     | Lower generalization        |

### **Final Model Selected:**

**Random Forest Classifier (GridSearch #2)**

* `max_depth = 20`
* `min_samples_leaf = 2`
* `min_samples_split = 10`
* `n_estimators = 250`

---

## Confusion Matrix (Training & Validation Sets)

### **Training Set Performance**

| Class    | Precision | Recall | F1-Score | Support |
| -------- | --------- | ------ | -------- | ------- |
| Denied   | 0.85      | 0.99   | 0.92     | 4,337   |
| Approved | 1.00      | 0.93   | 0.96     | 10,345  |

The model achieves very high recall for the *Denied* class and near-perfect precision for the *Approved* class, though the training accuracy indicates slight overfitting.

---

### **Validation Set Performance**

| Class    | Precision | Recall | F1-Score | Support |
| -------- | --------- | ------ | -------- | ------- |
| Denied   | 0.72      | 0.78   | 0.75     | 933     |
| Approved | 0.90      | 0.87   | 0.89     | 2,213   |

The model generalizes well, maintaining strong discriminative performance with:

* **High precision for Approvals (0.90)**
* **Balanced recall for both classes**
* **Overall ROC-AUC of 0.9155**


---
