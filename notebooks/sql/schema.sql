BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS customers_raw (
    customer_id            TEXT PRIMARY KEY,
    tenure_months          INTEGER,
    monthly_charges        REAL,
    total_charges          REAL,
    num_support_calls      INTEGER,
    contract_type          TEXT,
    internet_service       TEXT,
    payment_method         TEXT,
    auto_pay               INTEGER,
    has_paperless_billing  INTEGER,
    is_senior_citizen      INTEGER,
    has_partner            INTEGER,
    has_dependents         INTEGER,
    churned                INTEGER
);

CREATE TABLE IF NOT EXISTS customers_features (
    customer_id            TEXT PRIMARY KEY,
    -- Core engineered features
    tenure_months          INTEGER,
    monthly_charges        REAL,
    total_charges          REAL,
    avg_monthly_spend      REAL,
    support_calls_rate     REAL,
    is_auto_pay            INTEGER,
    is_paperless           INTEGER,
    is_senior_citizen      INTEGER,
    has_partner            INTEGER,
    has_dependents         INTEGER,
    contract_type          TEXT,
    internet_service       TEXT,
    payment_method         TEXT,
    churned                INTEGER
);

COMMIT;

