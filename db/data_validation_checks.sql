-- Run inside PostgreSQL with \i db/data_validation_checks.sql
-- Total regular season games per year
SELECT EXTRACT(YEAR FROM game_date) as season_year, COUNT(*) FROM games
WHERE game_type = 'R'
GROUP BY season_year
ORDER BY season_year;

-- Check for missing team names
SELECT COUNT(*) FROM games
WHERE home_team IS NULL OR away_team IS NULL;

-- Check for missing scores in completed games
SELECT COUNT(*) FROM games
WHERE (home_score IS NULL OR away_score IS NULL) 
AND (state = 'Final' OR state = 'Completed Early');

-- Duplicate game_id check
SELECT game_id, COUNT(*) FROM games
GROUP BY game_id 
HAVING COUNT(*) > 1;