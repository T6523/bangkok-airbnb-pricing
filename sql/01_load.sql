CREATE OR REPLACE TABLE raw_listings AS 
SELECT * FROM read_csv('${listing_path}', all_varchar=true);

CREATE OR REPLACE TABLE raw_neighbors AS 
SELECT * FROM read_csv('${neighbor_path}', all_varchar=true);