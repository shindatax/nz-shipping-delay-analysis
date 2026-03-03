-- =============================================================================
-- 01_create_tables.sql
-- NZ Shipping Delay Analysis — Database Schema
-- =============================================================================

CREATE TABLE IF NOT EXISTS shipments (
    shipment_id         SERIAL PRIMARY KEY,
    transport_mode      VARCHAR(20)     NOT NULL,
    origin_port         VARCHAR(50)     NOT NULL,
    destination_region  VARCHAR(50)     NOT NULL,
    cargo_type          VARCHAR(30)     NOT NULL,
    season              VARCHAR(10)     NOT NULL,
    carrier_reliability VARCHAR(10)     NOT NULL,
    customs_complexity  VARCHAR(10)     NOT NULL,
    lead_time_days      DECIMAL(5,1)    NOT NULL,
    shipment_weight_kg  DECIMAL(8,1)    NOT NULL,
    distance_km         DECIMAL(8,0)    NOT NULL,
    port_congestion_idx DECIMAL(4,2)    NOT NULL,
    weather_severity    DECIMAL(4,2)    NOT NULL,
    customs_days        DECIMAL(4,1)    NOT NULL,
    documentation_errors INT            NOT NULL DEFAULT 0,
    is_delayed          BOOLEAN         NOT NULL,
    delay_hours         DECIMAL(6,1)    NOT NULL DEFAULT 0,
    base_cost_nzd       DECIMAL(10,2)   NOT NULL,
    delay_cost_nzd      DECIMAL(10,2)   NOT NULL DEFAULT 0,
    total_cost_nzd      DECIMAL(10,2)   NOT NULL,
    created_at          TIMESTAMP       DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common query patterns
CREATE INDEX idx_shipments_transport_mode  ON shipments(transport_mode);
CREATE INDEX idx_shipments_origin_port     ON shipments(origin_port);
CREATE INDEX idx_shipments_is_delayed      ON shipments(is_delayed);
CREATE INDEX idx_shipments_cargo_type      ON shipments(cargo_type);
CREATE INDEX idx_shipments_season          ON shipments(season);
