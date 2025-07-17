import pandas as pd
from mlxtend.evaluate.time_series import GroupTimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder
from typing import Literal, List, Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_data(input_path):
    return pd.read_csv(input_path, parse_dates=['game_date','game_date_time'])
    
def data_split(
        df: pd.DataFrame, 
        holdout_start_date: str,  
        group_col: str
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    
    df = df.copy()

    # Sort data by date
    df.sort_values('game_date_time', inplace=True)
    
    # Initial split
    df_train = df[df["game_date"]<holdout_start_date]
    df_holdout = df[df["game_date"]>=holdout_start_date]
    
    # Define groups for time series cv
    groups = df_train[group_col]

    return df_train, df_holdout, groups

def transform_categorical_features(
        df: pd.DataFrame, 
        categorical_cols: List[str],
        encoding_type: Literal["one-hot", "category"],
        one_hot_encoder: OneHotEncoder = None, # Optional: pass a fitted OneHotEncoder
        category_maps: Dict[str, List[Any]] = None, # Optional: pass a dict of {col: [categories]}
        numerical_cols: List[str] = None, # Optional: list of numerical cols for ordering
        all_final_cols: List[str] = None # Optional: final ordered list of all columns
) -> pd.DataFrame:
    """
    Transforms categorical columns using a pre-fitted encoder or pre-defined cateogry maps,
    and combines with numerical columns, ensuring consistent column order. 

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to transform.
    categorical_cols : List[str]
        List of column names to encode. 
    encoding_type : Literal["one-hot", "category"]
        Type of encoding to apply.
    one_hot_encoder : OneHotEncoder, optional
        A pre-fitted sklearn.preprocessing.OneHotEncoder instance for "one-hot" encoding.
        Required if encoding_type is "one-hot".
    category_maps : Dict[str, List[Any]]
        A dictionary where keys are column names and values are lists of ordered categories.
        Required if encoding_type is "category".
    numerical_cols : List[str], optional
        List of column names that are numerical. Use for consistent column ordering.
        Highly recommended if encoding_type is "one-hot".
    all_final_cols : List[str], optional
        The complete ordered list of column names that the output DataFrame should have.
        Required if encoding_type is "one-hot" to ensure consistent feature order. 

    Returns
    -------
    pd.DataFrame
        The DataFrame with transformed categorical features.
    """
    df_transformed = df.copy()

    if encoding_type == "one-hot":
        if one_hot_encoder is None:
            raise ValueError("For 'one-hot' encoding, a fitted 'one_hot_encoder' must be provided.")
        if numerical_cols is None or all_final_cols is None:
            raise ValueError("For 'one-hot' encoding, 'numerical_cols' and 'all_final_cols' must be provided.")
        
        # Transform categorical columns using the provided encoder
        X_cat_encoded = one_hot_encoder.transform(df_transformed[categorical_cols])
        encoded_feature_names = one_hot_encoder.get_feature_names_out(categorical_cols)

        X_cat_df = pd.DataFrame(X_cat_encoded, columns=encoded_feature_names, index=df_transformed.index)

        # Handle case where df might be empty
        if df_transformed.empty:
            return pd.DataFrame(columns=all_final_cols)
        
        # Separate original numerical columns
        df_numerical_part = df_transformed[numerical_cols]

        # Concatenate numerical and encoded categorical parts
        df_final = pd.concat([df_numerical_part, X_cat_df], axis=1)

        # Reorder columns to match the expected final order from training
        df_final = df_final[all_final_cols]

        return df_final
    
    elif encoding_type == "category":
        if category_maps is None:
            raise ValueError("For 'category' encoding, 'category_maps' must be provided with predefined categories.")
        
        # Apply pre-defined categories to ensure consistency
        for col in categorical_cols:
            if col in df_transformed.columns:
                if col in category_maps:
                    # Convert to category dtype with observed categories
                    df_transformed[col] = pd.Categorical(df_transformed[col], categories=category_maps[col])
                else:
                    # Fallback if a map for a categorical column is missing
                    df_transformed[col] = df_transformed[col].astype('category')

            else:
                # Handle missing categorical columns in the DataFrame being transformed
                # If the column is missing, and it's supposed to be categorical add it as a categorical column with no data
                df_transformed[col] = pd.Categorical([], categories=category_maps.get(col, []))

        return df_transformed
    
    else:
        raise NotImplementedError(f"Encoding type '{encoding_type}' not supported.")