import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.betting.betting_utils import (
    convert_odds_to_probability,
    compute_expected_profit, 
    kelly_criterion
)

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def keep_preferred_bookmaker(group, preferred_bookmakers):

    # Filter to same_time and preferred bookmaker
    same_time_bookmaker_preferred = group[(group['time_match']==1) & (group['bookmaker'].isin(preferred_bookmakers))]

    # Filter to preferred bookmaker
    bookmaker_preferred = group[group['bookmaker'].isin(preferred_bookmakers)]

    if len(same_time_bookmaker_preferred) > 0:
        # Keep first preferred bookmaker where time matches
        return same_time_bookmaker_preferred.iloc[[0]]
    
    elif len(same_time_bookmaker_preferred) == 0 and len(bookmaker_preferred) > 0: 
        # Keep first preferred bookmaker
        return bookmaker_preferred.iloc[[0]]
    
    else:
        # If no preferred bookmaker, keep all rows for this game_id
        return group


def join_odds_to_scored_data(df_scored: pd.DataFrame, df_odds: pd.DataFrame) -> pd.DataFrame:

    logging.info(f"Earliest Date in Scored Data: {df_scored['game_date'].min()}, Latest Date in Scored Data: {df_scored['game_date'].max()}")
    logging.info(f"Earliest Date in Odds Data: {df_odds['game_date'].min()}, Latest Date in Odds Data: {df_odds['game_date'].max()}")

    # Convert to datetime
    df_scored['game_date_time'] = pd.to_datetime(df_scored['game_date_time'], utc=True)
    df_scored['game_date'] = pd.to_datetime(df_scored['game_date'], utc=True)
    df_odds['game_date'] = pd.to_datetime(df_odds['game_date'], utc=True)
    df_odds['commence_time'] = pd.to_datetime(df_odds['commence_time'], utc=True)
    df_odds['timestamp'] = pd.to_datetime(df_odds['timestamp'], utc=True)

    df_scored['season'] = df_scored['game_date'].dt.year

    # Join odds data to scored data
    odds_cols = ['timestamp','game_date','commence_time','home_team','away_team','bookmaker','home_odds','away_odds']
    logging.info(f"df_score Shape BEFORE Join: {df_scored.shape}")
    logging.info(f"df_scored # Unique Games BEFORE Join: {df_scored['game_id'].nunique()}")
    df_scored = df_scored.merge(
        df_odds[odds_cols],
        how='left',
        on=['game_date','home_team','away_team'], # will produce duplicate rows for double-headers (ok for now...)
        suffixes=('','_odds'),
        indicator=True
        )
    logging.info(f"df_score Shape AFTER Join: {df_scored.shape}")
    logging.info(f"df_scored # Unique Games AFTER Join: {df_scored['game_id'].nunique()}")

    logging.info(f"Join Match Rate: {df_scored['_merge'].value_counts(dropna=False, normalize=True).round(3)}")
    df_scored.drop('_merge', axis=1, inplace=True)

    # Dedup based on approximate times and selecting preferred bookmaker
    df_scored['time_diff'] = (df_scored['game_date_time']-df_scored['commence_time']).dt.total_seconds()/60 # difference in minutes
    df_scored['time_match'] = (df_scored['time_diff'].abs() < 10).astype(int) # times within 10 minutes of each other are the same
    
    PREFERRED_BOOKMAKERS = ["betmgm", "draftkings", "fanduel", "williamhill_us", "espnbet"]
    df_scored['preferred_bookmaker'] = df_scored['bookmaker'].isin(PREFERRED_BOOKMAKERS).astype(int)

    df_scored = (
        df_scored
        .groupby('game_id')
        .apply(keep_preferred_bookmaker, PREFERRED_BOOKMAKERS, include_groups=False)
        .reset_index()
        .drop('level_1', axis=1)
    )

    df_scored['game_date_only'] = df_scored['game_date'].dt.date

    logging.info(f"df_scored Shape AFTER Deduplication: {df_scored.shape}")
    logging.info(f"df_scored # Unique Games AFTER Join: {df_scored['game_id'].nunique()}")

    return df_scored


def calculate_betting_metrics(df_scored: pd.DataFrame) -> pd.DataFrame:

    # Compute Implied Probabilities
    df_scored['home_implied_probability'] = df_scored['home_odds'].apply(convert_odds_to_probability)
    df_scored['away_implied_probability'] = df_scored['away_odds'].apply(convert_odds_to_probability)

    # Edge
    df_scored['home_edge'] = df_scored['scoring_probabilities'] - df_scored['home_implied_probability']
    df_scored['away_edge'] = (1-df_scored['scoring_probabilities']) - df_scored['away_implied_probability']

    # Compute Expected Profit
    df_scored['home_expected_profit'] = df_scored.apply(lambda x: compute_expected_profit(x['scoring_probabilities'],x['home_odds']), axis=1)
    df_scored['away_expected_profit'] = df_scored.apply(lambda x: compute_expected_profit(1-x['scoring_probabilities'],x['away_odds']), axis=1)

    # Kelly Bet Size
    df_scored['home_full_kelly'] = df_scored.apply(lambda x: kelly_criterion(x['home_odds'], x['scoring_probabilities'], frac=1), axis=1)
    df_scored['away_full_kelly'] = df_scored.apply(lambda x: kelly_criterion(x['away_odds'], 1-x['scoring_probabilities'], frac=1), axis=1)
    df_scored['home_half_kelly'] = df_scored.apply(lambda x: kelly_criterion(x['home_odds'], x['scoring_probabilities'], frac=0.5), axis=1)
    df_scored['away_half_kelly'] = df_scored.apply(lambda x: kelly_criterion(x['away_odds'], 1-x['scoring_probabilities'], frac=0.5), axis=1)

    # Value
    df_scored['home_is_valuable'] = (df_scored['home_expected_profit'] > 0).astype('Int64')
    df_scored['away_is_valuable'] = (df_scored['away_expected_profit'] > 0).astype('Int64')

    # Value vs. Result
    df_scored['home_value_bet_agree'] = (df_scored['home_is_valuable'] == df_scored['home_win']).astype('Int64')
    df_scored['home_value_bet_win'] = ((df_scored['home_is_valuable']==1) & (df_scored['home_win']==1)).astype('Int64')
    df_scored['away_value_bet_agree'] = (df_scored['away_is_valuable'] != df_scored['home_win']).astype('Int64')
    df_scored['away_value_bet_win'] = ((df_scored['away_is_valuable']==1) & (df_scored['home_win']==0)).astype('Int64')

    # Full amount to win and lose
    df_scored['home_win_full_amount'] = np.where(df_scored['home_odds'] < 0, 100, df_scored['home_odds'])
    df_scored['home_loss_full_amount'] = np.where(df_scored['home_odds'] < 0, df_scored['home_odds'], -100)
    df_scored['away_win_full_amount'] = np.where(df_scored['away_odds'] < 0, 100, df_scored['away_odds'])
    df_scored['away_loss_full_amount'] = np.where(df_scored['away_odds'] < 0, df_scored['away_odds'], -100)

    # Net profit per bet
    home_profit_conditions = [
        (df_scored['home_is_valuable']==1) & (df_scored['home_win']==1),
        (df_scored['home_is_valuable']==1) & (df_scored['home_win']==0),
        df_scored['home_is_valuable']==0
        ]
    home_choices = [
        df_scored['home_win_full_amount'],
        df_scored['home_loss_full_amount'],
        np.nan
        ]
    df_scored['home_net_profit'] = np.select(home_profit_conditions, home_choices, default=np.nan)

    away_profit_conditions = [
        (df_scored['away_is_valuable']==1) & (df_scored['home_win']==0),
        (df_scored['away_is_valuable']==1) & (df_scored['home_win']==1),
        df_scored['away_is_valuable']==0
        ]
    away_choices = [
        df_scored['away_win_full_amount'],
        df_scored['away_loss_full_amount'],
        np.nan
        ]
    df_scored['away_net_profit'] = np.select(away_profit_conditions, away_choices, default=np.nan)
    

    return df_scored

def plot_net_profit(df_scored: pd.DataFrame):
    df_scored_grp_home = df_scored.groupby('game_date')['home_net_profit'].sum().reset_index()
    df_scored_grp_home.columns=['game_date','home_net_profit']
    df_scored_grp_away = df_scored.groupby('game_date')['away_net_profit'].sum().reset_index()
    df_scored_grp_away.columns=['game_date','away_net_profit']

    df_scored_grp = df_scored_grp_home.merge(df_scored_grp_away, how='left', on=['game_date'])
    df_scored_grp['home_net_profit'].fillna(0, inplace=True)
    df_scored_grp['away_net_profit'].fillna(0, inplace=True)
    df_scored_grp['total_net_profit'] = df_scored_grp['home_net_profit']+df_scored_grp['away_net_profit']

    plt.plot(df_scored_grp_home['game_date'], df_scored_grp_home['home_net_profit'], alpha=.1, label='Home Net Profit')
    plt.plot(df_scored_grp_away['game_date'], df_scored_grp_away['away_net_profit'], alpha=.1, label='Away Net Profit')
    plt.plot(df_scored_grp['game_date'], df_scored_grp['total_net_profit'], label='Total Net Profit')
    plt.axhline(0, color='black', linestyle='-')
    plt.legend(loc='best')
    plt.title('Net Profits Over Time | Backtest')
    plt.xticks(rotation=90)
    plt.show()

def summarize_evaluation(df_scored: pd.DataFrame, cutoff_date_str: str = None):

    if cutoff_date_str:
        cutoff_date = pd.to_datetime(cutoff_date_str, utc=True)
        df_scored = df_scored[df_scored['game_date']>cutoff_date]
    else:
        cutoff_date=pd.to_datetime('2021-04-01',utc=True)
    # Number of valuable bets
    num_home_valuable = df_scored['home_is_valuable'].sum()
    num_away_valuable = df_scored['away_is_valuable'].sum()
    num_valuable = num_home_valuable + num_away_valuable

    logging.info(f"Number of Valuable Bets Since {cutoff_date.date().strftime('%Y-%m-%d')}: {num_valuable} out of {len(df_scored)} games")
    logging.info(f"Proportion of Valuable Home Bets Since {cutoff_date.date().strftime('%Y-%m-%d')}: {num_home_valuable/df_scored['home_is_valuable'].count(): .3f}")
    logging.info(f"Proportion of Valuable Away Bets Since {cutoff_date.date().strftime('%Y-%m-%d')}: {num_away_valuable/df_scored['away_is_valuable'].count(): .3f}")
    num_home_value_bet_agree = df_scored['home_value_bet_agree'].sum()
    num_away_value_bet_agree = df_scored['away_value_bet_agree'].sum()
    num_home_value_bet_win = df_scored['home_value_bet_win'].sum()
    num_away_value_bet_win = df_scored['away_value_bet_win'].sum()

    logging.info(f"Number of Home Value Decision Agreements: {num_home_value_bet_agree}")
    logging.info(f"Number of Away Value Decision Agreements: {num_away_value_bet_agree}")
    logging.info(f"Number of Home Value Bet Wins: {num_home_value_bet_win} out of {num_home_valuable} = {num_home_value_bet_win/num_home_valuable : .3f}")
    logging.info(f"Number of Away Value Bet Wins: {num_away_value_bet_win} out of {num_away_valuable} = {num_away_value_bet_win/num_away_valuable : .3f}")

    total_home_net_profit = df_scored['home_net_profit'].sum()
    total_away_net_profit = df_scored['away_net_profit'].sum()

    logging.info(f"Total Home Net Profits Since {cutoff_date.date().strftime('%Y-%m-%d')}: ${total_home_net_profit}")
    logging.info(f"Total Away Net Profits Since {cutoff_date.date().strftime('%Y-%m-%d')}: ${total_away_net_profit}")
    logging.info(f"Total Net Profits Since {cutoff_date.date().strftime('%Y-%m-%d')}: ${total_home_net_profit+total_away_net_profit}")

    plot_net_profit(df_scored=df_scored)


    return 
    

def run_evaluation(df_scored: pd.DataFrame, df_odds: pd.DataFrame):

    # Step 1 - Join
    df_scored = join_odds_to_scored_data(df_scored=df_scored, df_odds=df_odds)

    # Step 2 - Betting Metrics
    df_scored = calculate_betting_metrics(df_scored=df_scored)

    # Step 3 - Output
    summarize_evaluation(df_scored=df_scored)

    summarize_evaluation(df_scored=df_scored, cutoff_date_str='2025-05-15')

    return df_scored


if __name__=='__main__':

    odds_path = "data/processed_game_odds.csv"
    scored_data_path = "src/scoring/output_logistic.csv"

    df_odds = pd.read_csv(odds_path)
    df_scored = pd.read_csv(scored_data_path)

    df_out = run_evaluation(df_scored=df_scored, df_odds=df_odds)

    print(df_out.columns[df_out.columns.str.contains('game_date')])

    save_path = "src/scoring/evaluation_logistic.csv"
    df_out.to_csv(save_path, index=False)

    # Need to finish bankro











        
    

