-- db/schema.sql  (skeleton—edit as you like)
-- cat db/schema.sql | psql -h localhost -p 5432 -U mlb_user -d mlb_db
-- psql -h localhost -U mlb_user -d mlb_db -f db/schema.sql
CREATE TABLE IF NOT EXISTS games (
    game_id      INTEGER PRIMARY KEY,
    game_date    DATE,
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
    homeRuns_pitcing INTEGER,
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