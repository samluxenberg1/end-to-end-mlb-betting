def convert_odds_to_probability(odds: int) -> float:
    """Convert moneyline odds to implied probability"""
    if odds < 0:
        probability = abs(odds) / (100 + abs(odds))
    else:
        probability = 100 / (100 + odds)
    
    return probability

def compute_expected_profit(pred_prob: float, odds: int) -> float:
    """Compute expected profit given predicted probabilities and odds"""
    if odds < 0:
        exp_profit = 100*pred_prob + odds*(1-pred_prob)
    else:
        exp_profit = odds*(pred_prob) - 100*(1-pred_prob)

    return exp_profit

def kelly_criterion(odds: int, pred_prob: float, frac: float = 1.0) -> float:
    """Compute fractional Kelly for optimal bet sizing"""
    if odds < 0:
        W = 100 / abs(odds)
    else:
        W = odds / 100

    return frac*(pred_prob-(1-pred_prob)/W)
    

def analyze_bet(pred_prob: float, odds: int) -> dict:
    """
    Comprehensive bet analysis including expected profit, implied probability,
    edge, and value determination.

    Parameters
    ----------
    pred_prob: float
        Predicted probability from model scoring
    odds: int
        American moneyline odds for the home team

    Returns
    -------
    dict
        Contains expected profit, implied probabability, edge, and value indicator
    
    """
    if not (0.0 <= pred_prob <= 1.0):
        raise ValueError("Predicted probability must be between 0.0 and 1.0")

    if odds == 0:
        raise ValueError("Odds cannot be zero")
    
    # Calculate expected profit
    expected_profit = compute_expected_profit(pred_prob, odds)

    # Calculate implied probability
    implied_prob = convert_odds_to_probability(odds)

    # Calculate edge
    edge = pred_prob - implied_prob# if odds < 0 else 1-pred_prob - implied_prob

    # Calculate bet sizes
    full_kelly = kelly_criterion(odds, pred_prob)
    half_kelly = .5*full_kelly

    return {
        "expected_profit": expected_profit,
        "implied_prob": implied_prob,
        "predicted_win_prob": pred_prob,
        "edge": edge,
        "is_valuable": expected_profit > 0,
        "full_kelly_bet_size": full_kelly,
        "half_kelly_bet_size": half_kelly
    }


if __name__=='__main__':
    # Examples
    # Case 1
    print("Case 1")
    away_odds = 120
    home_odds = -200
    away_win_prob = .35
    home_win_prob = 1-away_win_prob

    print(f"Away (+120) vs. Home (-200)")
    print(f"Bet on Away...")
    print(analyze_bet(away_win_prob, away_odds))

    print("Bet on Home...")
    print(analyze_bet(home_win_prob, home_odds))

    print("\nCase 2")
    away_odds = 120
    home_odds = -200
    away_win_prob = .65
    home_win_prob = 1-away_win_prob
    print(f"Away (+120) vs. Home (-200)")
    print(f"Bet on Away...")
    print(analyze_bet(away_win_prob, away_odds))

    print("Bet on Home...")
    print(analyze_bet(home_win_prob, home_odds))

    print("\nCase 3")
    away_odds = -200
    home_odds = 120
    away_win_prob = .35
    home_win_prob = 1-away_win_prob
    print(f"Away (-200) vs. Home (+120)")
    print(f"Bet on Away...")
    print(analyze_bet(away_win_prob, away_odds))

    print("Bet on Home...")
    print(analyze_bet(home_win_prob, home_odds))

    print("\nCase 4")
    away_odds = -200
    home_odds = 120
    away_win_prob = .65
    home_win_prob = 1-away_win_prob
    print(f"Away (-200) vs. Home (+120)")
    print(f"Bet on Away...")
    print(analyze_bet(away_win_prob, away_odds))

    print("Bet on Home...")
    print(analyze_bet(home_win_prob, home_odds))