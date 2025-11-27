# main.py

import time
import argparse
import numpy as np
import random
import math

from code.data import load_data, wave2, wave1, split
from code.features import features
from code.models import models
from code.evaluate import evaluate

# SETTINGS

TRIALS_NUMBER = 50
K_FOLD_K = 10

# 0) Parse cmd line args

start_time = time.perf_counter()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=101,
    )
    return parser.parse_args()


args = parse_args()

data_path = args.data_path
seed = args.seed

random.seed(seed)
np.random.seed(seed)

# 1) load data
vh = load_data(data_path)
vh_wave2 = wave2(vh)

# 2) split
X_train, X_test, y_train, y_test = split(vh_wave2, random_state=seed)

# 3) features
X_train, X_test, y_train, y_test, X_train_std, X_test_std = features(
    X_train, X_test, y_train, y_test
)

# 4) models
ols_model, ridge_model, lasso_model, dt_model, rf_model = models(
    seed, TRIALS_NUMBER, K_FOLD_K, X_train, y_train, X_train_std
)

# 5) evaluation
evaluate(
    ols_model,
    ridge_model,
    lasso_model,
    dt_model,
    rf_model,
    X_test,
    X_test_std,
    y_test,
)

# 6) Wave 1 estimation and analysis



# 7) Graphs for outputs (other metrics too if necessary)
 


# 8) Timer output

end_time = time.perf_counter()
total_time = end_time - start_time
minutes = math.floor(total_time / 60)
seconds = total_time % 60
print(f"Total runtime: {minutes} minutes, {seconds:.2f} seconds for {TRIALS_NUMBER} trials per model and {K_FOLD_K}-fold.\n")