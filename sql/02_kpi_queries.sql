-- =============================================================================
-- 02_kpi_queries.sql
-- NZ Shipping Delay Analysis — Operational KPI Queries
-- =============================================================================


-- -----------------------------------------------------------------------------
-- KPI 1: Overall On-Time Performance
-- -----------------------------------------------------------------------------
SELECT
    COUNT(*)                                                AS total_shipments,
    SUM(CASE WHEN is_delayed = FALSE THEN 1 ELSE 0 END)    AS on_time,
    SUM(CASE WHEN is_delayed = TRUE  THEN 1 ELSE 0 END)    AS delayed,
    ROUND(
        SUM(CASE WHEN is_delayed = FALSE THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1
    )                                                       AS on_time_rate_pct,
    ROUND(SUM(delay_cost_nzd), 0)                          AS total_delay_cost_nzd
FROM shipments;


-- -----------------------------------------------------------------------------
-- KPI 2: Delay Rate & Avg Cost by Origin Port
-- -----------------------------------------------------------------------------
SELECT
    origin_port,
    COUNT(*)                                                AS total_shipments,
    SUM(CASE WHEN is_delayed THEN 1 ELSE 0 END)            AS delayed_count,
    ROUND(
        SUM(CASE WHEN is_delayed THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1
    )                                                       AS delay_rate_pct,
    ROUND(AVG(CASE WHEN is_delayed THEN delay_hours END), 1) AS avg_delay_hours,
    ROUND(AVG(CASE WHEN is_delayed THEN delay_cost_nzd END), 0) AS avg_delay_cost_nzd,
    ROUND(SUM(delay_cost_nzd), 0)                          AS total_delay_cost_nzd
FROM shipments
GROUP BY origin_port
ORDER BY delay_rate_pct DESC;


-- -----------------------------------------------------------------------------
-- KPI 3: Delay Rate & Cost by Cargo Type
-- -----------------------------------------------------------------------------
SELECT
    cargo_type,
    COUNT(*)                                                AS total_shipments,
    ROUND(
        SUM(CASE WHEN is_delayed THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1
    )                                                       AS delay_rate_pct,
    ROUND(AVG(CASE WHEN is_delayed THEN delay_cost_nzd END), 0) AS avg_delay_cost_nzd,
    ROUND(SUM(delay_cost_nzd), 0)                          AS total_delay_cost_nzd
FROM shipments
GROUP BY cargo_type
ORDER BY avg_delay_cost_nzd DESC;


-- -----------------------------------------------------------------------------
-- KPI 4: Seasonal Delay Patterns
-- -----------------------------------------------------------------------------
SELECT
    season,
    COUNT(*)                                                AS total_shipments,
    ROUND(
        SUM(CASE WHEN is_delayed THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1
    )                                                       AS delay_rate_pct,
    ROUND(AVG(CASE WHEN is_delayed THEN delay_hours END), 1) AS avg_delay_hours
FROM shipments
GROUP BY season
ORDER BY delay_rate_pct DESC;


-- -----------------------------------------------------------------------------
-- KPI 5: Carrier Reliability Impact
-- -----------------------------------------------------------------------------
SELECT
    carrier_reliability,
    COUNT(*)                                                AS total_shipments,
    ROUND(
        SUM(CASE WHEN is_delayed THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1
    )                                                       AS delay_rate_pct,
    ROUND(SUM(delay_cost_nzd), 0)                          AS total_delay_cost_nzd
FROM shipments
GROUP BY carrier_reliability
ORDER BY delay_rate_pct DESC;


-- -----------------------------------------------------------------------------
-- KPI 6: Documentation Error Impact
-- -----------------------------------------------------------------------------
SELECT
    CASE
        WHEN documentation_errors = 0 THEN 'No Errors'
        WHEN documentation_errors = 1 THEN '1 Error'
        ELSE '2+ Errors'
    END                                                     AS error_category,
    COUNT(*)                                                AS total_shipments,
    ROUND(
        SUM(CASE WHEN is_delayed THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1
    )                                                       AS delay_rate_pct,
    ROUND(AVG(delay_cost_nzd), 0)                          AS avg_delay_cost_nzd
FROM shipments
GROUP BY error_category
ORDER BY delay_rate_pct DESC;
