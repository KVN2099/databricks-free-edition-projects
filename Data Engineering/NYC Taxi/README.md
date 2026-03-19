## NYC Taxi — Bronze / Silver / Gold (Lakeflow)

This example shows a **medallion architecture** on top of the public **NYC yellow taxi** sample that ships with Databricks. You first materialize a **batch** Delta table from the sample CSVs, then declare a **Lakeflow-style** pipeline in SQL: bronze ingests continuously, silver applies **data quality** filters, gold exposes **daily aggregates** for dashboards or downstream features.

### Why three layers

- **Bronze** preserves a faithful stream from the source table (append-oriented, minimal rules) so you can replay or audit.
- **Silver** enforces **business rules** (e.g. positive passengers and distance, non-null timestamps) so downstream metrics are trustworthy.
- **Gold** precomputes **aggregates** (by day) so BI tools and ML features read small, fast tables instead of scanning raw trips.

Conceptual flow:

```text
Sample CSVs (DBFS) → Delta base table → bronze (streaming) → silver (streaming) → gold (MV)
```

### Run order

1. **`00 Setup.ipynb`**  
   - Reads from `dbfs:/databricks-datasets/nyctaxi/tripdata/yellow` with an explicit **StructType** schema (column names and types match the public sample).  
   - Writes with `saveAsTable` to **`workspace.default.nyc_taxi`** in the stock notebook—change this to your **catalog.schema.table** if you avoid `workspace.default`.

2. **`01 Pipeline.sql`**  
   - **Bronze:** `CREATE OR REFRESH STREAMING TABLE bronze_nyc_taxi AS SELECT * FROM STREAM workspace.default.nyc_taxi`  
   - **Silver:** filters invalid rows into `silver_nyc_taxi`  
   - **Gold:** `gold_nyc_taxi_daily_fare` materialized view with daily trip counts, total fare, average distance  

Run the setup notebook **once** (or when you need to rebuild the base table). Deploy or run the SQL pipeline in your Lakeflow / SDP product UI or job as your workspace requires.

### Prerequisites

- **Sample data** available at the `dbfs:/databricks-datasets/...` path (standard on many workspaces; if missing, copy an equivalent dataset to a volume and adjust the read path).
- **Permissions** to create **streaming tables** and **materialized views** in the catalog you target, and to run the pipeline workload.
- **Product support** for the SQL syntax you use (`CREATE OR REFRESH STREAMING TABLE`, etc.) on your runtime—align with current Lakeflow / SDP documentation for your region.

### Customization (important)

The same **three-part table name** must match in both places:

- The **output** of `00 Setup.ipynb` (`saveAsTable`)
- The **source** in `01 Pipeline.sql` (`FROM STREAM catalog.schema.table`)

If you only change one, the pipeline will point at an empty or wrong table.

You may also relocate gold/silver to a **dedicated schema** (e.g. `nyc_taxi_bronze`, `nyc_taxi_gold`) for cleaner ACLs—update all `CREATE` statements consistently.

### Operations notes

- **Backfill:** rebuilding the base Delta table may require a full refresh of downstream streaming tables depending on product behavior; plan maintenance windows for production.
- **Monitoring:** use pipeline run history and data expectations (if you add them later) to catch schema drift in the source files.
- **Cost:** streaming + MVs consume compute on schedule or trigger; tune pipeline clusters for your data volume.
