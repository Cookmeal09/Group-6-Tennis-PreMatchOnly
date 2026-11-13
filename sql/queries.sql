-- Step 3: sql/queries.sql
CREATE OR REPLACE TABLE tn AS SELECT * FROM 'data/processed/tennis.parquet';
-- Q1: how often the better-ranked player wins, by year, with a three-year moving average (window function)
SELECT year, n, better_ranked_wins_pct, AVG(better_ranked_wins_pct) OVER (ORDER BY year ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_avg_3y FROM (
  SELECT year, COUNT(*) AS n, 100.0 * AVG(CASE WHEN (rank_diff < 0 AND y = 1) OR (rank_diff > 0 AND y = 0) THEN 1 ELSE 0 END) AS better_ranked_wins_pct FROM tn GROUP BY 1) ORDER BY 1;
-- Q2: the same share by surface and by best-of-five matches
SELECT surface, best_of_5, COUNT(*) AS n, 100.0 * AVG(CASE WHEN (rank_diff < 0 AND y = 1) OR (rank_diff > 0 AND y = 0) THEN 1 ELSE 0 END) AS better_ranked_wins_pct FROM tn GROUP BY 1, 2 ORDER BY 1, 2;
-- Q3: win share of player A by decile of the rating gap (NTILE)
SELECT d AS elo_gap_decile, COUNT(*) AS n, AVG(elo_diff) AS mean_elo_gap, 100.0 * AVG(y) AS player_a_wins_pct FROM (SELECT *, NTILE(10) OVER (ORDER BY elo_diff) AS d FROM tn) GROUP BY 1 ORDER BY 1;
-- Q4: model table
CREATE OR REPLACE TABLE feat AS SELECT * FROM tn;
SELECT year, COUNT(*) AS n FROM feat WHERE year >= 2020 GROUP BY 1 ORDER BY 1;
