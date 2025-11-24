# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.model_selection import train_test_split

## Loading the dataset
vh = pd.read_csv("vh_data14.csv")
vh.info()
vh.describe()

# Wave 1: no vaccine hesitancy data, it will be the unlabeled prediction dataset
# Wave 2: vaccine hesitancy data is available, training data

## WAVE 2:
# Filtering for the second wave
vh_wave2 = vh[vh['round'] == 2]
vh_wave2.info()

# Deleting redundant columns
vh_wave2 = vh_wave2.drop(columns=['inv_p', 'response_round_2', 'obs', 'round', 'county_covid_cap_cases', 'county_covid_cap_cases2wk'])

# Setting the respondent_id as the index column
vh_wave2.set_index("respondent_id", inplace=True)



## WAVE 1:
vh_wave1 = vh.sort_values(by=["respondent_id", "round"])

# Dropping the rows which appear in both waves so that the wave1 and wave2 dataframes will not overlap
counts = vh_wave1['respondent_id'].value_counts()   # how many times an id occurs
unique_ids = counts[counts == 1].index              # getting the ids that only appear once
vh_wave1 = vh_wave1[vh_wave1['respondent_id'].isin(unique_ids)] # keeping the ids that only appear once

# Deleting redundant columns
vh_wave1 = vh_wave1.drop(columns=['inv_p', 'response_round_2','round', 'obs', 'perceived_personal_riskq297_4',
                      'perceived_network_risk', 'doctor_comfort', 'fear_needles',
                      'trump_approval_retrospective', 'vaccine_trust', 'vaccine_hesitant', 'county_covid_cap_cases', 'county_covid_cap_cases2wk'])

# Setting the respondent_id as the index column
vh_wave1.set_index("respondent_id", inplace=True)


## Splitting Wave 2 into test and train
y = vh_wave2['vaccine_hesitant']
X = vh_wave2.drop(columns=['vaccine_hesitant'])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,        
    random_state=100,
    stratify=y) 

