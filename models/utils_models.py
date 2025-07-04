import pandas as pd

def load_model_data(input_path):
    return pd.read_csv(input_path, parse_date=['game_date','game_date_time'])
    