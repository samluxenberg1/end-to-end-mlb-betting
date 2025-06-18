-- db/schema.sql  (skeleton—edit as you like)
-- cat db/schema.sql | psql -h localhost -p 5432 -U mlb_user -d mlb_db
-- psql -h localhost -U mlb_user -d mlb_db -f db/schema.sql
CREATE TABLE IF NOT EXISTS games (
    game_id      INTEGER PRIMARY KEY,
    game_date    DATE,
    home_team_id INTEGER,
    away_team_id INTEGER,
    home_team    TEXT,
    away_team    TEXT,
    home_score   INTEGER,
    away_score   INTEGER,
    state        TEXT,
    venue        TEXT,
    game_type    TEXT
);

CREATE TABLE IF NOT EXISTS team_stats (
    game_pk INTEGER,
    team_side TEXT,
    team_id INTEGER,
    runs_batting INTEGER,
    hits_batting INTEGER,
    strikeOuts_batting INTEGER,
    baseOnBalls_batting INTEGER,
    avg NUMERIC,
    obp NUMERIC,
    slg NUMERIC,
    pitchesThrown INTEGER,
    balls_pitching INTEGER,
    strikes_pitching INTEGER,
    strikeOuts_pitching INTEGER,
    baseOnBalls_pitching INTEGER,
    hits_pitching INTEGER,
    earnedRuns INTEGER,
    homeRuns_pitching INTEGER,
    runs_pitching INTEGER,
    era NUMERIC,
    whip NUMERIC,
    groundOuts_pitching INTEGER,
    airOuts_pitching INTEGER,
    total INTEGER,
    putOuts INTEGER,
    assists INTEGER,
    errors INTEGER,
    doublePlays INTEGER,
    triplePlays INTEGER,
    rangeFactor NUMERIC,
    caughtStealing INTEGER,
    passedBall INTEGER,
    innings NUMERIC,
    PRIMARY KEY (game_pk, team_side)
);

CREATE TABLE IF NOT EXISTS player_stats (
    game_pk INTEGER,
    team_id INTEGER,
    team_side TEXT,          -- 'home' or 'away'
    player_id INTEGER,
    player_name TEXT,

    -- Batting
    at_bats INTEGER,
    runs_scored INTEGER,
    hits INTEGER,
    home_runs INTEGER,
    rbis INTEGER,
    walks_batting INTEGER,
    strikeouts_batting INTEGER,
    left_on_base INTEGER,
    stolen_bases INTEGER,

    -- Pitching
    innings_pitched TEXT,             -- Format: 'X.Y' (e.g., '2.1' innings) – store as TEXT 
    hits_allowed INTEGER,
    runs_allowed INTEGER,
    earned_runs INTEGER,
    strikeouts_pitching INTEGER,
    walks_pitching INTEGER,
    pitches_thrown INTEGER,

    -- Fielding
    putouts INTEGER,
    assists INTEGER,
    errors INTEGER,

    PRIMARY KEY (game_pk, player_id)
);
