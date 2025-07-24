import argparse

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
        "expected_profit": round(expected_profit,2),
        "implied_prob": round(implied_prob,4),
        "predicted_win_prob": round(pred_prob,4),
        "edge": round(edge,4),
        "is_valuable": expected_profit > 0,
        "full_kelly_bet_size": round(full_kelly,4),
        "half_kelly_bet_size": round(half_kelly,4)
    }


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--home-odds", type=int)
    parser.add_argument("--away-odds", type=int)
    parser.add_argument("--home-win-prob", type=float)
    args = parser.parse_args()

    away_odds = args.away_odds
    home_odds = args.home_odds
    home_win_prob = args.home_win_prob
    away_win_prob = 1-args.home_win_prob
    

    print(f"Away ({away_odds}) vs. Home ({home_odds})\n")
    print(f"Bet on Away...")
    print(analyze_bet(away_win_prob, away_odds))

    print("\nBet on Home...")
    print(analyze_bet(home_win_prob, home_odds))

    