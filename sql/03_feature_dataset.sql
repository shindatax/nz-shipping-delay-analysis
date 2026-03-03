-- =============================================================================
-- 03_feature_dataset.sql
-- NZ Shipping Delay Analysis — ML Feature Engineering View
-- Prepares a clean, analysis-ready dataset for the prediction model
-- =============================================================================


-- -----------------------------------------------------------------------------
-- View: ML-ready feature dataset
-- Encodes categorical variables and adds engineered features
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_ml_features AS
SELECT
    shipment_id,

    -- Encoded categorical features
    CASE transport_mode
        WHEN 'Sea Freight' THEN 0
        WHEN 'Air Cargo'   THEN 1
        WHEN 'Road'        THEN 2
        WHEN 'Rail'        THEN 3
    END                                         AS transport_mode_enc,

    CASE origin_port
        WHEN 'Port of Auckland'  THEN 0
        WHEN 'Port of Tauranga'  THEN 1
        WHEN 'Lyttelton Port'    THEN 2
        WHEN 'Port Otago'        THEN 3
        WHEN 'Wellington Port'   THEN 4
    END                                         AS origin_port_enc,

    CASE cargo_type
        WHEN 'General Merchandise' THEN 0
        WHEN 'Refrigerated'        THEN 1
        WHEN 'Hazardous'           THEN 2
        WHEN 'Bulk'                THEN 3
        WHEN 'Automotive'          THEN 4
        WHEN 'Pharmaceuticals'     THEN 5
    END                                         AS cargo_type_enc,

    CASE carrier_reliability
        WHEN 'High'   THEN 2
        WHEN 'Medium' THEN 1
        WHEN 'Low'    THEN 0
    END                                         AS carrier_reliability_enc,

    CASE customs_complexity
        WHEN 'Simple'   THEN 0
        WHEN 'Moderate' THEN 1
        WHEN 'Complex'  THEN 2
    END                                         AS customs_complexity_enc,

    CASE season
        WHEN 'Summer' THEN 0
        WHEN 'Autumn' THEN 1
        WHEN 'Winter' THEN 2
        WHEN 'Spring' THEN 3
    END                                         AS season_enc,

    -- Numerical features
    lead_time_days,
    shipment_weight_kg,
    distance_km,
    port_congestion_idx,
    weather_severity,
    customs_days,
    documentation_errors,

    -- Engineered features
    CASE WHEN documentation_errors > 0 THEN 1 ELSE 0 END  AS has_doc_errors,
    ROUND(shipment_weight_kg / NULLIF(distance_km, 0), 4)  AS weight_per_km,
    port_congestion_idx * weather_severity                  AS risk_score,

    -- Target variables
    CASE WHEN is_delayed THEN 1 ELSE 0 END      AS is_delayed,
    delay_hours,
    delay_cost_nzd

FROM shipments;


-- -----------------------------------------------------------------------------
-- View: High-risk shipment summary for operational dashboard
-- Flags shipments with multiple delay risk factors
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW vw_high_risk_shipments AS
SELECT
    shipment_id,
    transport_mode,
    origin_port,
    cargo_type,
    carrier_reliability,
    port_congestion_idx,
    weather_severity,
    documentation_errors,
    customs_complexity,

    -- Risk score: count of high-risk factors
    (
        CASE WHEN transport_mode = 'Sea Freight'       THEN 1 ELSE 0 END +
        CASE WHEN port_congestion_idx > 7              THEN 1 ELSE 0 END +
        CASE WHEN weather_severity > 7                 THEN 1 ELSE 0 END +
        CASE WHEN documentation_errors > 0             THEN 1 ELSE 0 END +
        CASE WHEN customs_complexity = 'Complex'       THEN 1 ELSE 0 END +
        CASE WHEN carrier_reliability = 'Low'          THEN 1 ELSE 0 END
    )                                                   AS risk_factor_count,

    is_delayed,
    delay_hours,
    delay_cost_nzd

FROM shipments
WHERE (
    CASE WHEN transport_mode = 'Sea Freight'       THEN 1 ELSE 0 END +
    CASE WHEN port_congestion_idx > 7              THEN 1 ELSE 0 END +
    CASE WHEN weather_severity > 7                 THEN 1 ELSE 0 END +
    CASE WHEN documentation_errors > 0             THEN 1 ELSE 0 END +
    CASE WHEN customs_complexity = 'Complex'       THEN 1 ELSE 0 END +
    CASE WHEN carrier_reliability = 'Low'          THEN 1 ELSE 0 END
) >= 3
ORDER BY risk_factor_count DESC;
