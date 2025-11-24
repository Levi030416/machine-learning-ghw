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

> **TODO:** Insert the exact Dataverse URL here once you finalize it:
> - `https://doi.org/...` (Harvard Dataverse link)

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
