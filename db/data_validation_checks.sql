-- Run inside PostgreSQL with \i db/data_validation_checks.sql

-- Checks on games
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

-- Checks on team stats
-- Check for games in games table with missing team stats
SELECT game_id, game_date FROM 
    (
        SELECT game_id, game_date, home_team, away_team, home_score, away_score, avg, pitchesthrown FROM games
        LEFT JOIN team_stats
        ON games.game_id = team_stats.game_pk
    )                                                                                                                                                                                                       on games.game_id = team_stats.game_pk)                                                                                                                                                                                     
WHERE avg IS NULL OR pitchesthrown IS NULL;

-- Check for one entry per team per game
SELECT game_pk, COUNT(*) FROM team_stats
GROUP BY game_pk
HAVING COUNT(*) != 2;

-- Check for missing critical stats
SELECT * FROM team_stats 
WHERE runs_batting IS NULL OR errors IS NULL OR hits_batting IS NULL;

-- Check for duplicate rows
SELECT game_pk, team_side, COUNT(*) FROM team_stats
GROUP BY game_pk, team_side
HAVING COUNT(*) > 1;

-- Check for matching team ids with games table --> need to revisit this. There are no team ids in the games table currently!!!!

-- Checks on player stats
-- Check for games in games table with missing player stats
SELECT game_id, game_date FROM
    (
        SELECT game_id, game_date, home_team, away_team, home_score, away_score, player_name FROM games
        LEFT JOIN player_stats
        ON games.game_id = player_stats.game_pk
    )
WHERE player_name IS NULL;

-- Join test: all game_pk values exist in games table
SELECT DISTINCT game_pk FROM player_stats
LEFT JOIN games 
ON games.game_id = player_stats.game_pk
WHERE games.game_id IS NULL;

-- Check that each row has valid player id and game_pk
SELECT * FROM player_stats 
WHERE player_id IS NULL OR game_pk IS NULL;

-- No duplicate rows per player per game (unless split by position)
SELECT game_pk, player_id, COUNT(*) FROM player_stats
GROUP BY game_pk, player_id
HAVING COUNT(*) > 1;

-- Stat range sanity checks
SELECT * FROM player_stats
WHERE at_bats < 0 OR hits < 0 OR home_runs < 0;

-- Check team totals = sum of player stats -- 22 data points where player total hits != team total hits
SELECT ps.game_pk, ps.team_id, SUM(ps.hits) as player_hits, ts.hits_batting as team_hits from player_stats ps
LEFT JOIN team_stats ts
ON ps.game_pk = ts.game_pk AND ps.team_id = ts.team_id
GROUP BY ps.game_pk, ps.team_id, ts.hits_batting
HAVING SUM(ps.hits) != ts.hits_batting;

-- Checks games where sum of player hits != team hits
-- Out of 22 games, 18 are exhibition games, which I plan to exclude from modeling anyway.
-- What are the other 4 regular season games? game ids: 633206, 777574, 777543, 777547
SELECT * FROM games
WHERE game_id in (
    SELECT ps.game_pk fro player_stats ps
    LEFT JOIN team_stats ts
    ON ps.game_pk = ts.game_pk AND ps.team_id = ts.game_id
    GROUP BY ps.game_pk, ps.team_id, ts.hits_batting
    HAVING SUM(ps.hits) != ts.hits_batting
);

-- Investigate those 4 games
SELECT * FROM games
WHERE game_id in (
    SELECT ps.game_pk FROM player_stats ps
    LEFT JOIN team_stats ts
    ON ps.game_pk = ts.game_pk AND ps.team_id = ts.team_id
    GROUP BY ps.game_pk, ps.team_id, ts.hits_batting
    HAVING SUM(ps.hits) != ts.hits_batting
) AND games.game_type != "E";