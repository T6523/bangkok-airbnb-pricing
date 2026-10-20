-- Feature table: one row per listing, built from clean_listings.

-- 1. Amenities: JSON-array string -> one row per (listing, amenity), name made column-safe.
CREATE OR REPLACE TABLE amenity_long AS
SELECT DISTINCT id, 'am_' || left(trim(regexp_replace(lower(a), '[^a-z0-9]+', '_', 'g'), '_'), 40) AS col
FROM (SELECT id, unnest(from_json(amenities, '["VARCHAR"]')) AS a FROM clean_listings)
WHERE col <> 'am_';

-- 2. Keep amenities present in 3%-97% of listings (rarer/commoner ones carry no signal),
--    the 25 closest to 50% (most informative for a binary flag).
CREATE OR REPLACE TABLE amenity_flags AS
SELECT col, freq
FROM (SELECT col, count(*) * 1.0 / (SELECT count(*) FROM clean_listings) AS freq
      FROM amenity_long GROUP BY col)
WHERE freq BETWEEN 0.03 AND 0.97
ORDER BY abs(freq - 0.5)
LIMIT 25;

COPY amenity_flags TO '${RESULTS_DIR}/amenity_flags.csv' (HEADER);

-- Wide 0/1 matrix, PIVOT leaves NULL where absent, filled with 0 in the final select.
CREATE OR REPLACE TABLE amenity_wide AS
PIVOT (SELECT id, col FROM amenity_long WHERE col IN (SELECT col FROM amenity_flags))
ON col USING max(1);

-- 3. Main table
CREATE OR REPLACE TABLE features AS
WITH host_stats AS (
    -- listings per host within the modelled scope
    SELECT id, COUNT(*) OVER (PARTITION BY host_id) AS host_listing_count
    FROM clean_listings
),
rating_rank AS (
    -- within-neighbourhood rating percentile; uses no price information
    SELECT id, PERCENT_RANK() OVER (PARTITION BY neighbourhood_cleansed ORDER BY review_scores_rating) AS rating_pct_in_hood
    FROM clean_listings
    WHERE review_scores_rating IS NOT NULL
),
top_types AS (
    SELECT property_type FROM clean_listings GROUP BY 1 ORDER BY count(*) DESC LIMIT 10
),
amen_count AS (
    SELECT id, count(*) AS amenity_count FROM amenity_long GROUP BY id
)
SELECT
    c.* EXCLUDE (amenities, property_type, first_review, last_review, last_scraped, minimum_nights, maximum_nights),

    -- size
    c.beds / NULLIF(c.accommodates, 0)       AS beds_per_person,
    c.bedrooms / NULLIF(c.accommodates, 0)   AS bedrooms_per_person,

    -- location: haversine km to city centre
    2 * 6371 * asin(sqrt(
        pow(sin(radians(c.latitude - ${REF_LAT}) / 2), 2)
        + cos(radians(${REF_LAT})) * cos(radians(c.latitude))
          * pow(sin(radians(c.longitude - ${REF_LON}) / 2), 2)
    ))                                       AS dist_to_center_km,

    -- type: rare property types -> 'Other'
    CASE WHEN c.property_type IN (SELECT property_type FROM top_types)
         THEN c.property_type ELSE 'Other' END AS property_type_grouped,

    -- host
    h.host_listing_count,
    h.host_listing_count > 1                 AS is_multi_listing_host,

    -- stay rules, capped (some raw values are absurd)
    LEAST(c.minimum_nights, 365)             AS minimum_nights,
    LEAST(c.maximum_nights, 365)             AS maximum_nights,

    -- reviews: ~1/3 of listings have none, so scores are NULL for them
    c.number_of_reviews > 0                  AS has_reviews,
    date_diff('day', c.last_review, c.last_scraped)  AS days_since_last_review,
    date_diff('day', c.first_review, c.last_scraped) AS listing_age_days,
    r.rating_pct_in_hood,
    (c.availability_365 = 0 AND c.number_of_reviews_ltm = 0) AS is_inactive_like,

    -- amenities (flags: 1 = present)
    coalesce(a.amenity_count, 0)             AS amenity_count,
    coalesce(COLUMNS(x -> x LIKE 'am\_%' ESCAPE '\'), 0)
FROM clean_listings c
JOIN host_stats h USING (id)
LEFT JOIN rating_rank r USING (id)
LEFT JOIN amen_count a USING (id)
LEFT JOIN amenity_wide w USING (id);

COPY features TO '${FEATURES_PATH}' (FORMAT PARQUET);
