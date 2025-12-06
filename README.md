# Vaccine Hesitancy ML Project

## 1. Overview

This repository implements a machine learning pipeline to analyse **COVID-19 vaccine hesitancy** in the United States. Using survey data (two waves) from Harvard Dataverse, we model the probability that a respondent is **vaccine hesitant** (`vaccine_hesitant`) based on:

- socio-demographic characteristics (age, gender, race, income, college education, etc.)
- political attitudes and trust in government and media
- trust in science and perceived risk of COVID-19
- health conditions and previous infection history :contentReference[oaicite:0]{index=0}

The final goal is to:
- build predictive models (e.g. Lasso, Ridge, Logistic Regression, Tree-based models), and  
- interpret which features most strongly predict vaccine hesitancy.

---

## 2. Data

### 2.1 Source

- **Original dataset**: Harvard Dataverse (COVID vaccine hesitancy survey)  
- **Wave 1**: no vaccine-hesitancy label → used later for *prediction only*  
- **Wave 2**: includes `vaccine_hesitant` → used as *training dataset* :contentReference[oaicite:1]{index=1}  

(https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FGJVWYF&fbclid=IwY2xjawORHtpleHRuA2FlbQIxMQBzcnRjBmFwcF9pZAEwAAEeFA8hkI5DVFFs3e4UinyZszS9x4FfHL5gNE-EBJfltBWlE_9Jjxoyyeh3FMs_aem_L2iGJBJikODnIj3Ct4ysTQ)

### 2.2 Files in this repo

- `data/vh_data14.csv` – raw combined dataset with both waves (from Dataverse). :contentReference[oaicite:2]{index=2}  
- `variable_descriptions.docx` – codebook describing each variable, type, and scale. :contentReference[oaicite:3]{index=3}  

Variables include:

- **Binary (0/1) dummies**: `male`, `college`, `evangelical`, `infected_personal`, `infected_network`, `president_approval`, `trump_approval_retrospective`, health conditions (e.g. `condition_diabetes`, `condition_obesity`), and `vaccine_hesitant`.  
- **Categorical**: `race` (1–4), `party_id` (1–5).  
- **Scales (0–10 or 0–12)**: trust in science, trust in media, trust in government, perceived risk, income, etc.  
- **Numeric**: age, county density, indices of political / science literacy, pandemic impact, and `vaccine_trust`. :contentReference[oaicite:4]{index=4}  

> See `variable_descriptions.docx` for full details on each feature.

### 2.3 How to get the data into `data/`

1. Download the dataset from Harvard Dataverse (Wave 1 + Wave 2 combined file).
2. Save it in this repository as:

   ```text
   data/vh_data14.csv


## 3 FULL PROJECT SETUP & EXECUTION

### 3.1 Clone repository and enter it
    git clone (https://github.com/Levi030416/vax-hesitancy-ml)
    cd vax-hesitancy-ml

### 3.2 Create virtual environment
    python -m venv .venv

### 3.3 Activate the environment
    Windows:
     .venv\Scripts\activate
    macOS / Linux:
     source .venv/bin/activate

### 3.4 Install all required packages
    pip install -r requirements.txt

### 3.5 EXPECTED FOLDER STRUCTURE
    project_root/
    ├── main.py
    ├── requirements.txt
    ├── data/
    │   └── dataset.csv        <-- DATA IS ALREADY INCLUDED HERE
    ├── code/
    │   ├── data.py
    │   ├── features.py
    │   ├── models.py
    │   └── evaluate.py
    └── outputs/               <-- GENERATED AUTOMATICALLY

### 3.6 RUN THE FULL MACHINE LEARNING PIPELINE
    --data_path is REQUIRED by the script even though the data is already in /data
    --seed is OPTIONAL (default = 101)
    python main.py --data_path data/dataset.csv --seed 101

### 3.7 PIPELINE BEHAVIOR
- Loads dataset from data/
- Uses Wave 2 for training
- 80/20 stratified train–test split
- Feature engineering (dummies, type conversion, imputation, scaling)

Model training:
 - OLS, Ridge, Lasso, Decision Tree, Random Forest (Optuna tuned)
 - Model evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
 - Wave 1 out-of-sample prediction
 - All results saved automatically into /outputs/

### 3.8 GENERATED OUTPUT FILES
    outputs/
    ├── metrics.json
    ├── metrics_table.csv
    ├── predictions.json
    ├── wave1_predictions.csv
    ├── wave1_prediction_descriptives.csv
    └── figures/
      ├── random_forest_feature_importance_top15.png
      ├── roc_curves.png
      ├── precision_recall_curves.png
      ├── wave1_distribution_ols.png
      ├── wave1_distribution_ridge.png
      ├── wave1_distribution_lasso.png
      ├── wave1_distribution_decision_tree.png
      └── wave1_distribution_random_forest.png

### 3.9 REPRODUCIBILITY
- Default seed = 101
- All tuning and cross-validation are deterministic
- Rerunning with the same seed reproduces identical results




