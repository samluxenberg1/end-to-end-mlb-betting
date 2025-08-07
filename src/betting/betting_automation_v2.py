import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import argparse
from datetime import date

class BettingAutomation:
    def __init__(self, bankroll: float = 600, kelly_fraction: float = 0.25):
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
    
    def calculate_kelly(self, odds: int, pred_prob: float) -> float:
        """Calculate Kelly criterion for given odds and predicted probability"""
        if odds < 0:
            W = 100 / abs(odds)
        else:
            W = odds / 100
        
        # Kelly formula: f = (bp - q) / b where b = W, p = pred_prob, q = 1-pred_prob
        kelly_f = (W * pred_prob - (1 - pred_prob)) / W
        return self.kelly_fraction * kelly_f
    
    def calculate_implied_probability(self, odds: int) -> float:
        """Convert American odds to implied probability"""
        if odds < 0:
            return abs(odds) / (abs(odds) + 100)
        else:
            return 100 / (odds + 100)
    
    def calculate_expected_value(self, odds: int, pred_prob: float, bet_amount: float) -> float:
        """Calculate expected value of a bet"""
        if odds < 0:
            win_amount = bet_amount * (100 / abs(odds))
        else:
            win_amount = bet_amount * (odds / 100)
        
        expected_return = (pred_prob * win_amount) - ((1 - pred_prob) * bet_amount)
        return expected_return
    
    def process_game(self, home_team: str, away_team: str, 
                    home_odds: int, away_odds: int, 
                    home_win_prob: float) -> Dict:
        """Process a single game and determine betting recommendation"""
        
        away_win_prob = 1 - home_win_prob
        
        # Calculate implied probabilities
        home_implied_prob = self.calculate_implied_probability(home_odds)
        away_implied_prob = self.calculate_implied_probability(away_odds)
        
        # Calculate edges
        home_edge = home_win_prob - home_implied_prob
        away_edge = away_win_prob - away_implied_prob
        
        # Calculate Kelly bet sizes
        home_kelly = self.calculate_kelly(home_odds, home_win_prob)
        away_kelly = self.calculate_kelly(away_odds, away_win_prob)
        
        # Calculate bet amounts
        home_bet_amount = max(0, home_kelly * self.bankroll)
        away_bet_amount = max(0, away_kelly * self.bankroll)
        
        # Determine recommendation
        recommendation = {
            'matchup': f"{away_team} @ {home_team}",
            'home_team': home_team,
            'away_team': away_team,
            'home_odds': home_odds,
            'away_odds': away_odds,
            'home_win_prob': home_win_prob,
            'away_win_prob': away_win_prob,
            'home_edge': home_edge,
            'away_edge': away_edge,
            'home_bet_amount': home_bet_amount,
            'away_bet_amount': away_bet_amount,
            'recommendation': 'NO BET'
        }
        
        # Determine best bet
        if home_edge > 0 and away_edge > 0:
            if home_edge > away_edge:
                recommendation['recommendation'] = f"BET ${home_bet_amount:.2f} on {home_team} ({home_odds:+d})"
                recommendation['bet_team'] = home_team
                recommendation['bet_amount'] = home_bet_amount
                recommendation['bet_odds'] = home_odds
                recommendation['edge'] = home_edge
            else:
                recommendation['recommendation'] = f"BET ${away_bet_amount:.2f} on {away_team} ({away_odds:+d})"
                recommendation['bet_team'] = away_team
                recommendation['bet_amount'] = away_bet_amount
                recommendation['bet_odds'] = away_odds
                recommendation['edge'] = away_edge
        elif home_edge > 0:
            recommendation['recommendation'] = f"BET ${home_bet_amount:.2f} on {home_team} ({home_odds:+d})"
            recommendation['bet_team'] = home_team
            recommendation['bet_amount'] = home_bet_amount
            recommendation['bet_odds'] = home_odds
            recommendation['edge'] = home_edge
        elif away_edge > 0:
            recommendation['recommendation'] = f"BET ${away_bet_amount:.2f} on {away_team} ({away_odds:+d})"
            recommendation['bet_team'] = away_team
            recommendation['bet_amount'] = away_bet_amount
            recommendation['bet_odds'] = away_odds
            recommendation['edge'] = away_edge
        
        return recommendation
    
    def generate_betting_summary(self, games_data: List[Dict]) -> pd.DataFrame:
        """Generate a clean betting summary from games data"""
        
        recommendations = []
        for game in games_data:
            rec = self.process_game(
                game['home_team'], game['away_team'],
                game['home_odds'], game['away_odds'],
                game['home_win_prob']
            )
            recommendations.append(rec)
        
        # Convert to DataFrame
        df = pd.DataFrame(recommendations)
        
        # Filter to only recommended bets
        bets_to_make = df[df['recommendation'] != 'NO BET'].copy()
        
        if len(bets_to_make) > 0:
            # Sort by edge (highest first)
            bets_to_make = bets_to_make.sort_values('edge', ascending=False)
            
            # Clean summary columns
            summary_cols = ['matchup', 'bet_team', 'bet_odds', 'bet_amount', 'edge', 'recommendation']
            return bets_to_make[summary_cols].reset_index(drop=True)
        else:
            return pd.DataFrame()
    
    def print_betting_summary(self, games_data: List[Dict]):
        """Print a clean, actionable betting summary"""
        
        print(f"\n{'='*60}")
        print(f"BETTING SUMMARY - {date.today()}")
        print(f"Bankroll: ${self.bankroll} | Kelly Fraction: {self.kelly_fraction}")
        print(f"{'='*60}")
        
        df = self.generate_betting_summary(games_data)
        
        if len(df) == 0:
            print("\n❌ NO BETS RECOMMENDED TODAY")
            print("No positive expected value opportunities found.")
            return
        
        print(f"\n✅ {len(df)} RECOMMENDED BETS:")
        print("-" * 60)
        
        total_action = 0
        for idx, row in df.iterrows():
            print(f"{idx+1}. {row['recommendation']}")
            print(f"   Edge: {row['edge']:.1%} | Expected Value: ${row['bet_amount'] * row['edge']:.2f}")
            print(f"   Matchup: {row['matchup']}")
            total_action += row['bet_amount']
            print()
        
        print(f"Total Action: ${total_action:.2f} ({total_action/self.bankroll:.1%} of bankroll)")
        
        # Safety checklist
        print("\n" + "="*60)
        print("SAFETY CHECKLIST:")
        print("□ Double-check team names match DraftKings exactly")
        print("□ Verify odds haven't moved significantly")
        print("□ Confirm bet amounts before placing")
        print("□ Check for duplicate bets in DraftKings app")
        print("="*60)

    def load_predictions_and_odds(self, predictions_csv: str, odds_csv: str = None) -> List[Dict]:
        """Load your prediction data and combine with odds"""
        
        # Load predictions (your scoring output)
        df_predictions = pd.read_csv(predictions_csv)
        
        if odds_csv:
            # If you have odds data, merge it
            df_odds = pd.read_csv(odds_csv)
            # Add your merge logic here based on your existing join_odds_to_scored_data function
            
        # For now, create manual template for DraftKings odds entry
        games_data = []
        
        print("MANUAL ODDS ENTRY REQUIRED:")
        print("="*50)
        
        for idx, row in df_predictions.iterrows():
            home_team = row.get('home_team', 'Unknown')
            away_team = row.get('away_team', 'Unknown') 
            home_win_prob = row.get('scoring_probabilities', 0.5)
            
            print(f"\nGame {idx+1}: {away_team} @ {home_team}")
            print(f"Predicted Home Win Prob: {home_win_prob:.3f}")
            
            # Manual odds entry (you'd replace this with your Excel data)
            try:
                home_odds_input = input(f"Enter HOME odds for {home_team} (e.g., +170 or -150): ")
                away_odds_input = input(f"Enter AWAY odds for {away_team} (e.g., +170 or -150): ")
                
                home_odds = int(home_odds_input.replace('+', ''))
                away_odds = int(away_odds_input.replace('+', ''))
                
                games_data.append({
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_odds': home_odds,
                    'away_odds': away_odds,
                    'home_win_prob': home_win_prob
                })
                
            except (ValueError, KeyboardInterrupt):
                print(f"Skipping game: {away_team} @ {home_team}")
                continue
        
        return games_data


# Example usage
if __name__ == "__main__":
    
    # Initialize betting automation
    betting_bot = BettingAutomation(bankroll=600, kelly_fraction=0.25)
    
    # Option 1: Use your actual prediction files
    try:
        predictions_file = "src/scoring/output_logistic_all_data.csv"  # Your scoring output
        games_data = betting_bot.load_predictions_and_odds(predictions_file)
        betting_bot.print_betting_summary(games_data)
    except FileNotFoundError:
        print("Prediction file not found, using sample data...")
        
        # Option 2: Manual sample data (for testing)
        sample_games = [
            {
                'home_team': 'San Francisco Giants',
                'away_team': 'Pittsburgh Pirates', 
                'home_odds': 170,
                'away_odds': -210,
                'home_win_prob': 0.567
            },
            {
                'home_team': 'Philadelphia Phillies',
                'away_team': 'Baltimore Orioles',
                'home_odds': -150,
                'away_odds': 130,
                'home_win_prob': 0.45
            }
        ]
        
        betting_bot.print_betting_summary(sample_games)