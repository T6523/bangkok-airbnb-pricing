-- Cleaned listings: typed columns, no free text/URLs, price filtered.

CREATE OR REPLACE TABLE clean_listings AS
SELECT
    TRY_CAST(id AS BIGINT)                                   AS id,
    TRY_CAST(host_id AS BIGINT)                              AS host_id,
    TRY_CAST(replace(replace(price, '$$', ''), ',', '') AS DOUBLE) AS price,

    -- host
    12 * TRY_CAST(hosts_time_as_host_years AS INT)
       + TRY_CAST(hosts_time_as_host_months AS INT)          AS host_tenure_months,
    host_is_superhost = 't'                                  AS host_is_superhost,
    host_identity_verified = 't'                             AS host_identity_verified,
    host_has_profile_pic = 't'                               AS host_has_profile_pic,
    TRY_CAST(host_listings_count AS INT)                     AS host_listings_count,
    TRY_CAST(calculated_host_listings_count AS INT)          AS calculated_host_listings_count,

    -- location & type
    neighbourhood_cleansed,
    TRY_CAST(latitude AS DOUBLE)                             AS latitude,
    TRY_CAST(longitude AS DOUBLE)                            AS longitude,
    property_type,
    room_type,

    -- size: bathrooms_text like '1 bath', '1.5 baths', 'Half-bath', 'Shared half-bath'
    TRY_CAST(accommodates AS INT)                            AS accommodates,
    TRY_CAST(bedrooms AS DOUBLE)                             AS bedrooms,
    TRY_CAST(beds AS DOUBLE)                                 AS beds,
    CASE
        WHEN bathrooms_text ILIKE '%half%' THEN 0.5
        ELSE TRY_CAST(regexp_extract(bathrooms_text, '(\d+(\.\d+)?)', 1) AS DOUBLE)
    END                                                      AS bathrooms,
    coalesce(bathrooms_text ILIKE '%shared%', FALSE)         AS bathroom_shared,
    amenities,

    -- stay rules
    TRY_CAST(minimum_nights AS INT)                          AS minimum_nights,
    TRY_CAST(maximum_nights AS INT)                          AS maximum_nights,

    -- availability
    TRY_CAST(availability_30 AS INT)                         AS availability_30,
    TRY_CAST(availability_60 AS INT)                         AS availability_60,
    TRY_CAST(availability_90 AS INT)                         AS availability_90,
    TRY_CAST(availability_365 AS INT)                        AS availability_365,

    -- reviews
    TRY_CAST(number_of_reviews AS INT)                       AS number_of_reviews,
    TRY_CAST(number_of_reviews_ltm AS INT)                   AS number_of_reviews_ltm,
    TRY_CAST(first_review AS DATE)                           AS first_review,
    TRY_CAST(last_review AS DATE)                            AS last_review,
    TRY_CAST(review_scores_rating AS DOUBLE)                 AS review_scores_rating,
    TRY_CAST(review_scores_accuracy AS DOUBLE)               AS review_scores_accuracy,
    TRY_CAST(review_scores_cleanliness AS DOUBLE)            AS review_scores_cleanliness,
    TRY_CAST(review_scores_checkin AS DOUBLE)                AS review_scores_checkin,
    TRY_CAST(review_scores_communication AS DOUBLE)          AS review_scores_communication,
    TRY_CAST(review_scores_location AS DOUBLE)               AS review_scores_location,
    TRY_CAST(review_scores_value AS DOUBLE)                  AS review_scores_value,
    TRY_CAST(reviews_per_month AS DOUBLE)                    AS reviews_per_month,

    TRY_CAST(last_scraped AS DATE)                           AS last_scraped
FROM raw_listings
WHERE TRY_CAST(replace(replace(price, '$$', ''), ',', '') AS DOUBLE) BETWEEN ${PRICE_MIN} AND ${PRICE_MAX}
  AND room_type IN ('Entire home/apt', 'Private room', 'Shared room', 'Hotel room')
  AND TRY_CAST(accommodates AS INT) IS NOT NULL;
