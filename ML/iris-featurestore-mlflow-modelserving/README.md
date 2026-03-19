## Iris Feature Store + MLflow + Model Serving (Databricks)

This is a **minimal but complete** ML lifecycle on Databricks: ingest a public dataset, store features in **Unity Catalog**, train with features resolved through **Feature Store**, register the model in **MLflow** (Unity Catalog registry), promote with **aliases**, then consume the same model for **batch** and **real-time** scoring.

### Why this demo exists

- **Feature Store** separates “feature tables in UC” from “training code,” so you can reuse features and audit what the model saw.
- **Aliases** (`champion`, `challenger`) give you a safe promotion story without hard-coding version numbers in every notebook.
- **Model Serving** shows how the registered artifact becomes an HTTP API for applications.

### Architecture (high level)

```text
Kaggle (Iris) → Delta table (UC) → FeatureLookup + training → MLflow model + aliases
                                                      ↓
                              Batch: models:/…@champion     Realtime: Serving endpoint
```

### Run order (required)

1. `1. Feature Store.ipynb`
2. `2. Model Training.ipynb`
3. `3.1 Batch Inference.ipynb`
4. `3.2 Realtime Inference.ipynb`

Skipping steps will fail: later notebooks expect the Delta table, registered model name, and (for 3.2) a deployable artifact.

### Where this runs

- **Workspace:** any Databricks workspace with UC and Model Registry support for your account.
- **Compute:** Serverless or classic cluster—match what your org allows for Feature Store and Serving.
- **Artifacts:** Delta in UC; models and aliases in MLflow tied to UC.

### Required libraries

Install on the compute that runs the notebooks (Libraries UI or environment panel):

| Package | Role |
| ------- | ---- |
| `kagglehub` | Download Iris from Kaggle without manual files |
| `databricks-feature-engineering` | `FeatureEngineeringClient`, `FeatureLookup` |
| `scikit-learn` | `RandomForestClassifier` |
| `databricks-sdk` | Create/update serving endpoint, sample request |

`mlflow`, `pyspark`, and notebook utilities are normally on the runtime already.

### Parameters (widgets)

Set widgets **before** “Run all” so paths stay consistent across notebooks.

**1. Feature Store**

- `catalog`, `schema`, `dataset` — target Delta table `catalog.schema.dataset`
- `kaggle_dataset_id` — e.g. `uciml/iris`
- `kaggle_dataset_path` — file inside the dataset, e.g. `Iris.csv`

**2. Model Training**

- `catalog`, `schema`, `dataset` — same table as step 1

**3.1 Batch Inference**

- `catalog`, `schema`, `dataset`
- `model_name` — e.g. `iris_rf_classifier`
- `production_alias` — e.g. `champion` (must exist after training promotion logic)

**3.2 Realtime Inference**

- Same as 3.1, plus `endpoint_name` — e.g. `iris_endpoint` (must be unique in the workspace region)

### How to run (checklist)

1. Create or choose a UC **catalog** and **schema** where you have **CREATE TABLE** and **CREATE MODEL** (or equivalent).
2. Attach compute; install the PyPI packages above.
3. Run notebook 1 fully; confirm the Delta table in the Catalog UI.
4. Run notebook 2; confirm MLflow run and registered model with aliases.
5. Run 3.1 to validate batch scoring from the registry URI.
6. Run 3.2; wait for the endpoint to become ready, then inspect the sample response.

### Notebook behavior (summary)

| Notebook | What it does |
| -------- | -------------- |
| 1. Feature Store | Loads Iris via `kagglehub`, normalizes columns/types, writes UC Delta, sets up primary key `id`. |
| 2. Model Training | Builds training dataframe with `FeatureLookup`, trains sklearn RF, logs to MLflow, registers in UC, sets `challenger`, conditionally promotes to `champion`. |
| 3.1 Batch Inference | Reads features, loads `models:/catalog.schema.model_name@alias`, outputs predictions. |
| 3.2 Realtime Inference | Provisions/updates a Model Serving endpoint and sends a test payload via SDK. |

### Troubleshooting

- **Permission errors on UC:** grant USE CATALOG / USE SCHEMA / SELECT / MODIFY on the feature table and model registration permissions for your user or group.
- **Serving endpoint stuck:** check quota, region, and that the model flavor is supported for serving; review endpoint events in the UI.
- **Wrong file path for Kaggle:** open the dataset on Kaggle and match `kaggle_dataset_path` to the actual CSV path inside the bundle.
