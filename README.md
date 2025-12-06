# Vaccine Hesitancy ML Project

## 1. Overview

This repository implements a machine learning pipeline to analyse **COVID-19 vaccine hesitancy** in the United States. Using survey data (two waves) from Harvard Dataverse, we classify respondents as **vaccine hesitant** (`vaccine_hesitant`) or not, based on:

- socio-demographic characteristics (age, gender, race, income, college education, etc.)
- political attitudes and trust in government and media
- trust in science and perceived risk of COVID-19
- health conditions and previous infection history

This could help decisionmakers in case of a new epidemic, to estimate the amount of people that can be vaccinated, which is a crucial feature of handling a situation like that.

The final goal is to:
- build predictive models (e.g. Lasso, Ridge, Logistic Regression, Tree-based models), and  
- interpret which features most strongly predict vaccine hesitancy.

---

## 2. Data

### 2.1 Source

- **Original dataset**: Harvard Dataverse (COVID vaccine hesitancy survey)  
- **Wave 1**: no vaccine-hesitancy label → used later for *prediction only*  
- **Wave 2**: includes `vaccine_hesitant` → used as *training dataset*

(https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FGJVWYF&fbclid=IwY2xjawORHtpleHRuA2FlbQIxMQBzcnRjBmFwcF9pZAEwAAEeFA8hkI5DVFFs3e4UinyZszS9x4FfHL5gNE-EBJfltBWlE_9Jjxoyyeh3FMs_aem_L2iGJBJikODnIj3Ct4ysTQ)

### 2.2 Files in this repo

- `data/vh_data14.csv` – raw combined dataset with both waves (from Dataverse).
- `variable_descriptions.docx` – codebook describing each variable, type, and scale.

Variables include:

- **Binary (0/1) dummies**: `male`, `college`, `evangelical`, `infected_personal`, `infected_network`, `president_approval`, `trump_approval_retrospective`, health conditions (e.g. `condition_diabetes`, `condition_obesity`), and `vaccine_hesitant`.  
- **Categorical**: `race` (1–4), `party_id` (1–5).  
- **Scales (0–10 or 0–12)**: trust in science, trust in media, trust in government, perceived risk, income, etc.  
- **Numeric**: age, county density, indices of political / science literacy, pandemic impact, and `vaccine_trust`.

> See `variable_descriptions.docx` for full details on each feature.

## 3 Setup and Exectuion

### 3.1 Download repository, unzip it, and enter the folder vax-hesitancy-ml-main in a terminal window
    cd Downloads/vax-hesitancy-ml-main

### 3.2 Create virtual environment
    python -m venv .venv

### 3.3 Activate the environment
    Windows:
     .venv\Scripts\activate
    macOS / Linux:
     source .venv/bin/activate

### 3.4 Install all required packages
    pip install -r requirements.txt

### 3.6 Run pipeline (arguments --data_path dataset/vh_data14.csv and --seed 101 are optional)
    python main.py

### 3.7 Expected runtime can vary between 5 to 10 minutes based on hardware specifications.

### 3.8 Generated output files
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










