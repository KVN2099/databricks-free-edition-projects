-- LakeFlow Pipeline: NYC Taxi Bronze-Silver-Gold Layers

-- Bronze Layer: Ingest raw NYC Taxi data as streaming table
CREATE OR REFRESH STREAMING TABLE bronze_nyc_taxi AS
SELECT * FROM STREAM workspace.default.nyc_taxi;

-- Silver Layer: Clean and filter records (e.g., filter invalid passenger counts and trip distances)
CREATE OR REFRESH STREAMING TABLE silver_nyc_taxi AS
SELECT * FROM STREAM bronze_nyc_taxi
WHERE Passenger_Count > 0 AND Trip_Distance > 0 AND Pickup_DateTime IS NOT NULL AND Dropoff_DateTime IS NOT NULL;

-- Gold Layer: Aggregated daily total fares per day
CREATE OR REFRESH MATERIALIZED VIEW gold_nyc_taxi_daily_fare AS
SELECT DATE(Pickup_DateTime) AS trip_date,
       COUNT(*) AS num_trips,
       SUM(Total_Amount) AS total_fare,
       AVG(Trip_Distance) AS avg_trip_distance
FROM silver_nyc_taxi
GROUP BY DATE(Pickup_DateTime)
ORDER BY trip_date;
