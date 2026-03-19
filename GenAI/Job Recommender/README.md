## Job recommender

This folder is a **two-stage data prep** demo: first **land** job-related data into **Unity Catalog**, then **inspect** the resulting table so you understand distributions, nulls, and keys before you invest in feature engineering or a ranking model. It is useful as a template for “bring your CSV or API export → UC → validation” workflows.

### What you will practice

- **Ingest:** moving from a file or upstream source to a **managed Delta table** with a clear schema.
- **Exploration:** using Spark SQL or DataFrame APIs to summarize columns, row counts, and data quality—typical precursors to a recommender (user id, job id, timestamps, categorical attributes).
- **Governance:** keeping the canonical dataset in UC so downstream notebooks share one definition of “jobs.”

### Notebooks

| Order | Notebook | What to expect |
| ----- | -------- | ---------------- |
| 1 | `Ingestar Dataset.ipynb` | Read the configured source (path/URL/API as implemented in cells), apply schema or casts, write to `catalog.schema.table` (names via widgets or variables). May include deduplication or basic cleansing. |
| 2 | `Detallar tabla.ipynb` | Profile the ingested table: schema, sample rows, possibly simple aggregates (e.g. jobs per category, missing rates). Use this output to decide joins and labels for a future model. |

Run **1 → 2**. Notebook 2 assumes the table from notebook 1 exists and is readable with your permissions.

### Prerequisites

- **Unity Catalog** with CREATE TABLE (or write access to an existing managed table) on your chosen catalog/schema.
- **Compute** attached to both notebooks (same cluster or Serverless for parity).
- **Libraries:** install any packages referenced in the **first code cells** (e.g. `requests`, `pandas`, cloud SDKs)—the notebooks may pull from public URLs or files you provide.

### Configuration

- Set **catalog**, **schema**, and **table** names consistently in both notebooks.
- If the ingest uses **secrets** (API keys, tokens), store them in Databricks secret scopes and reference them with `dbutils.secrets`, not plain text.
- For large files, consider **Auto Loader** or **COPY INTO** patterns; this demo may use a simpler read for clarity—scale the pattern as needed.

### Next steps (outside this repo)

A full “job recommender” usually adds: user and job embeddings or tabular features, negative sampling, offline metrics (nDCG, recall@k), and optionally **MLflow** + **Feature Store** for production. This repository stops at a clean, queryable UC table plus profiling so you can plug in your own modeling notebooks.
