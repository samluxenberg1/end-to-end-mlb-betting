import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, log_loss 
from mlxtend.evaluate.time_series import GroupTimeSeriesSplit
import numpy as np



from utils_models import (
    load_model_data, 
    data_split,
    categorical_encoding
)

def train_model(
        X: pd.DataFrame, y: pd.Series, 
        hyperparams_dict: dict, 
        groups: pd.Series, cv_dict: dict, 
        cat_encoding_type: str, cat_cols: list[str]
        ):

    # Create folds
    gts = GroupTimeSeriesSplit(**cv_dict)

    # Instantiate Model
    model = RandomForestClassifier(**hyperparams_dict)

    print("\n--- Executing Rolling Window Cross-Validation ---")
    fold_num = 0
    individual_fold_accuracies = []

    for train_index, test_index in gts.split(X, y, groups=groups):
        fold_num += 1

        # Extract training and testing sets for this fold
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Get the unique dates for current train and test sets to verify split
        train_dates = groups.iloc[train_index].unique()
        test_dates = groups.iloc[test_index].unique()

        print(f"\n--- Fold {fold_num} ---")
        print(f"    Train samples: {len(X_train)} | Test samples: {len(X_test)}")
        print(f"    Train unique days: {len(train_dates)} ({train_dates.min()} to {train_dates.max()})")
        print(f"    Test unique days: {len(test_dates)} ({test_dates.min()} to {test_dates.max()})")

        # Verification: Ensure no date overlap and test dates are after train dates
        if len(train_dates) > 0 and len(test_dates) > 0:
            if train_dates.max() >= test_dates.min():
                print(f"    ERROR: Date overlap! Train max date ({train_dates.max().date()}) is not strictly before Test min date ({test_dates.min().date()}).")

        X_train, X_test = categorical_encoding(
            X_train=X_train, 
            X_test=X_test, 
            categorical_cols=cat_cols, 
            encoding_type="one-hot"
        )

        # Train model
        model.fit(X_train, y_train)

        # Evaluate model
        score = model.score(X_test, y_test)
        individual_fold_accuracies.append(score)
        print(f"    Accuracy for this fold: {score: .4f}")
    
    print("\n--- Cross-Validation Summary ---")
    print(f"Individual fold accuracies: {individual_fold_accuracies}")
    print(f"Mean accuracy across folds: {np.mean(individual_fold_accuracies): .4f}")
    print(f"Standard deviation of accuracy: {np.std(individual_fold_accuracies): .4f}")

if __name__=='__main__':
    input_model_data = "data/processed/model_data.csv"
    df_model = load_model_data(input_model_data)

    target = "home_win"

    """
    Data Splitting Strategy

    - Split between training and testing sets where the test set will be reserved
    as the holdout set for this project. The holdout set will contain the most 
    recent data that could be close (in time) to the real-time scoring I 
    will need beginning July 24th, 2025. 

    - Within the training set, I will use GroupTimeSeriesSplit with an expanding
    window from mlxtend, and a gap size of 3 days between each training and test
    set of each split. I chose 3 days as the gap because I currently have some
    lag 3 features. While this isn't perfect, it does help toward preventing
    some unnecessary data leakage. 
    """

    # Initial Split by Time
    df_train_init, df_holdout, groups = data_split(
        df=df_model, 
        holdout_start_date='2025-05-01', 
        group_col='game_date'
        )

    # Separate out target
    y_train_init = df_train_init[target]
    y_holdout = df_holdout[target] 

    # Define groups for GroupTSCV
    groups = df_train_init["game_date"]

    # Remove columns
    cols_to_remove = [
        "game_id", 
        "game_date",
        "game_date_time",
        "home_team_id",
        "away_team_id",
        "home_score",
        "away_score",
        "state",
        target
    ]
    df_train_init.drop(cols_to_remove, axis=1, inplace=True)
    df_holdout.drop(cols_to_remove, axis=1, inplace=True)

    # Convert categorical features
    cat_cols = ['home_team','away_team','venue','game_type']
    # df_train_init, df_holdout = categorical_encoding(
    #     X_train=df_train_init, 
    #     X_test=df_holdout, 
    #     categorical_cols=cat_cols, 
    #     encoding_type="one-hot"
    #     )

    # Set time series split group arguments
    cv_args = {
        "test_size": 30, # test about 30 days worth of games
        "train_size": 360, # 2021 to 2022 seasons are beginning of each training fold
        "gap_size": 3, # skip 3 days between training and test sets to avoid data leakage
        "window_type": "rolling" # fix beginning of training period
        }

    # Train Model
    rf_hyperparams = {
        'min_samples_leaf': 5,
        'class_weight': 'balanced',
        'random_state': 888,
        'n_jobs': -1
    }

    train_model(
        X=df_train_init, 
        y=y_train_init, 
        hyperparams_dict=rf_hyperparams, 
        groups=groups, 
        cv_dict=cv_args, 
        cat_encoding_type="one-hot", 
        cat_cols=cat_cols
        )