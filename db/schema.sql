-- db/schema.sql  (skeleton—edit as you like)
-- cat db/schema.sql | psql -h localhost -p 5432 -U mlb_user -d mlb_db
-- psql -h localhost -U mlb_user -d mlb_db -f db/schema.sql
CREATE TABLE IF NOT EXISTS games (
    game_id      TEXT PRIMARY KEY,
    game_date    DATE,
    home_team    TEXT,
    away_team    TEXT,
    home_score   INT,
    away_score   INT
);