# code/features.py
# Only features(X_train, X_test, y_train, y_test) to be called from main.py

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


numeric_cols = [
    'individual_responsibility', 'trust_science_community', 'trust_science_polmotives',
    'trust_science_politicians', 'trust_science_media', 'trust_media',
    'trust_gov_nat', 'trust_gov_state', 'trust_gov_local',
    'perceived_personal_riskq297_4', 'perceived_network_risk',
    'doctor_comfort', 'fear_needles', 'income', 'county_density',
    'age','psindex', 'nsindex', 'pandemic_impact_personal', 'pandemic_impact_network',
    'vaccine_trust']


def make_approval_dummies(X_train: pd.DataFrame,
                          X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    X_train['president_approval'] = X_train['president_approval'].map({'Yes': 1, 'No': 0})
    X_train['trump_approval_retrospective'] = X_train['trump_approval_retrospective'].map({'Yes': 1, 'No': 0})
    
    X_test['president_approval'] = X_test['president_approval'].map({'Yes': 1, 'No': 0})
    X_test['trump_approval_retrospective'] = X_test['trump_approval_retrospective'].map({'Yes': 1, 'No': 0})

    return X_train, X_test


def convert_categorical_columns(X_train: pd.DataFrame,
                                X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    categorical_cols = [
        'male', 'college', 'evangelical', 'infected_personal', 'infected_network',
        'president_approval', 'trump_approval_retrospective',
        'condition_pregnant', 'condition_asthma', 'condition_lung', 'condition_diabetes',
        'condition_immune', 'condition_obesity', 'condition_heart', 'condition_organ', 'race', 'party_id']
    
    for col in categorical_cols:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    return X_train, X_test


def convert_numeric_columns(X_train: pd.DataFrame,
                            X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    for col in numeric_cols:
        X_train[col] = X_train[col].astype('float')
        X_test[col] = X_test[col].astype('float')

    return X_train, X_test


def make_other_categorical_dummies(X_train: pd.DataFrame,
                                   X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    X_train = pd.get_dummies(X_train, columns=['race', 'party_id'], drop_first=True)
    X_test = pd.get_dummies(X_test, columns=['race', 'party_id'], drop_first=True)
    
    return X_train, X_test


def impute_missing_values(X_train: pd.DataFrame,
                          X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    median_imputer = SimpleImputer(strategy='median')
    X_train[numeric_cols] = median_imputer.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = median_imputer.transform(X_test[numeric_cols])

    categorical_cols = [
        'male', 'college', 'evangelical', 'infected_personal', 'infected_network',
        'president_approval', 'trump_approval_retrospective',
        'condition_pregnant', 'condition_asthma', 'condition_lung', 'condition_diabetes',
        'condition_immune', 'condition_obesity', 'condition_heart', 'condition_organ', 'race_2', 'race_3', 'race_4', 
        'party_id_Independent', 'party_id_Libertarian', 'party_id_Other party', 'party_id_Republican']

    mode_imputer = SimpleImputer(strategy='most_frequent')
    X_train[categorical_cols] = mode_imputer.fit_transform(X_train[categorical_cols])
    X_test[categorical_cols] = mode_imputer.transform(X_test[categorical_cols])

    for col in categorical_cols:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    return X_train, X_test


def standardize(X_train: pd.DataFrame,
                X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    X_train_standardized = X_train.copy()
    X_test_standardized = X_test.copy()
    scaler = StandardScaler()
    X_train_standardized[numeric_cols] = scaler.fit_transform(X_train_standardized[numeric_cols])
    X_test_standardized[numeric_cols] = scaler.transform(X_test_standardized[numeric_cols])
    
    return X_train_standardized, X_test_standardized
   
def features(X_train, X_test, y_train, y_test):

    # 1) special dummies
    X_train, X_test = make_approval_dummies(X_train, X_test)

    # 2) dtypes
    X_train, X_test = convert_categorical_columns(X_train, X_test)
    X_train, X_test = convert_numeric_columns(X_train, X_test)

    # 3) other categoricals → dummies
    X_train, X_test = make_other_categorical_dummies(X_train, X_test)

    # 4) impute missing values
    X_train, X_test = impute_missing_values(X_train, X_test)

    # 5) standardized copies for Lasso/Ridge
    X_train_std, X_test_std = standardize(X_train, X_test)
    
    print("Feature engineering done.")

    return X_train, X_test, y_train, y_test, X_train_std, X_test_std
