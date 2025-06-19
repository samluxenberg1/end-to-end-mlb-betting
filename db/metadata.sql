CREATE TABLE IF NOT EXISTS known_data_issues (
    game_id INTEGER PRIMARY KEY,
    team_id INTEGER,
    game_type TEXT,
    issue_type TEXT,
    notes TEXT
);

INSERT INTO known_data_issues (game_id, team_id, game_type, issue_type, notes)
SELECT
    ps.game_pk,
    ps.team_id,
    games.game_type,
    'player_team_stat_mismatch' AS issue_type,
    'Player hits do not sum to team hits' AS notes
FROM player_stats ps
LEFT JOIN team_stats ts
    ON ps.game_pk = ts.game_pk AND ps.team_id = ts.team_id
LEFT JOIN games 
    ON ps.game_pk = games.game_id
GROUP BY ps.game_pk, ps.team_id, ts.hits_batting, games.game_type
HAVING SUM(ps.hits) != ts.hits_batting
ON CONFLICT DO NOTHING;

-- Filter out games with issues 
/*
SELECT * FROM games 
WHERE game_type = 'R' AND
    game_id NOT IN (SELECT game_pk FROM known_data_issues);
*/
 