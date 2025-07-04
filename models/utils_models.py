import pandas as pd
from mlxtend.evaluate.time_series import GroupTimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder
from typing import Literal

def load_model_data(input_path):
    return pd.read_csv(input_path, parse_dates=['game_date','game_date_time'])
    
def data_split(
        df: pd.DataFrame, 
        holdout_start_date: str,  
        group_col: str
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    
    df = df.copy()

    # Sort data by date
    df.sort_values('game_date_time', inplace=True)
    
    # Initial split
    df_train = df[df["game_date"]<holdout_start_date]
    df_holdout = df[df["game_date"]>=holdout_start_date]
    
    # Define groups for time series cv
    groups = df_train[group_col]

    return df_train, df_holdout, groups

def categorical_encoding(
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame,
        categorical_cols: list[str], 
        encoding_type = Literal["one-hot", "category"]
        ) -> tuple[pd.DataFrame, pd.DataFrame]:

    # Make copies to avoid modifying original DataFrame
    X_train = X_train.copy()
    X_test = X_test.copy()

    # 2 Options - 1. One-Hot-Endcoding, 2. "category" type
    if encoding_type == "one-hot":
        
        # Apply one-hot encoding to train and test
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_train_cat_encoded = encoder.fit_transform(X_train[categorical_cols])
        X_test_cat_encoded = encoder.transform(X_test[categorical_cols])

        feature_names = encoder.get_feature_names_out(categorical_cols)
        X_train_cat = pd.DataFrame(X_train_cat_encoded, columns=feature_names, index=X_train.index)
        X_test_cat = pd.DataFrame(X_test_cat_encoded, columns=feature_names, index=X_test.index)

        # Drop these columns from original data
        X_train.drop(categorical_cols, axis=1, inplace=True)
        X_test.drop(categorical_cols, axis=1, inplace=True)

        return pd.concat([X_train, X_train_cat], axis=1), pd.concat([X_test, X_test_cat], axis=1)

    else: # encoding_type == "category"
        # Handle unseen categories
        for col in categorical_cols:
            # Get categories from training data
            train_categories = X_train[col].unique()

            # Convert to category with training categories
            X_train[col] = pd.Categorical(X_train[col], categories=train_categories)
            X_test[col] = pd.Categorical(X_test[col], categories=train_categories)

        return X_train, X_test

