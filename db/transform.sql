-- Remove any clearly bad or null data
-- Convert date to date type
-- Add winner column
-- Create new table called games_clean

DROP TABLE IF EXISTS games_clean;

CREATE TABLE games_clean AS
SELECT 
    game_id,
    CAST(game_date AS DATE) AS game_date,
    home_team,
    away_team,
    home_score,
    away_score, 
    CASE
        WHEN home_score > away_score THEN home_team 
        WHEN home_score < away_score THEN away_team
        ELSE 'TIE'
    END AS winner,
    home_score - away_score AS score_differential,
    CASE
        WHEN home_score > away_score THEN 1
        ELSE 0
    END as home_win
FROM games
WHERE 
    home_team IS NOT NULL
    AND away_team IS NOT NULL
    AND home_score IS NOT NULL
    AND away_score IS NOT NULL;