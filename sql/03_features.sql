CREATE OR REPLACE TABLE features AS
SELECT
    * EXCLUDE (amenities, first_review, last_review, last_scraped),
    date_diff('day', last_review, last_scraped)  AS days_since_last_review,
    date_diff('day', first_review, last_scraped) AS listing_age_days
FROM clean_listings;

COPY features TO '${FEATURES_PATH}' (FORMAT PARQUET);
