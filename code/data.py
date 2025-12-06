# code/data.py

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    vh = pd.read_csv(path)
    return vh


def wave2(vh: pd.DataFrame) -> pd.DataFrame:

    # Filtering for the second wave
    vh_wave2 = vh[vh['round'] == 2]
    
    # Deleting redundant columns
    vh_wave2 = vh_wave2.drop(columns=['inv_p', 'response_round_2', 'obs', 'round', 'county_covid_cap_cases', 'county_covid_cap_cases2wk'])
    
    # Setting the respondent_id as the index column
    vh_wave2.set_index("respondent_id", inplace=True)

    return vh_wave2


def wave1(vh: pd.DataFrame) -> pd.DataFrame:
    
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

    return vh_wave1


def split(
    vh_wave2: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int | None = None,
):
    
    y = vh_wave2["vaccine_hesitant"]
    X = vh_wave2.drop(columns=["vaccine_hesitant"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test
