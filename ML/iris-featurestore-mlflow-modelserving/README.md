## Iris Feature Store + MLflow + Model Serving (Databricks)

A small, end‑to‑end workflow that loads the Iris dataset into Unity Catalog, trains and registers a model with MLflow/Feature Store, then performs batch and realtime inference using Databricks Model Serving.

### Run order
1. `ML/iris-featurestore-mlflow-modelserving/1. Feature Store.ipynb`
2. `ML/iris-featurestore-mlflow-modelserving/2. Model Training.ipynb`
3. `ML/iris-featurestore-mlflow-modelserving/3.1 Batch Inference.ipynb`
4. `ML/iris-featurestore-mlflow-modelserving/3.2 Realtime Inference.ipynb`

### Where these run
- **Environment**: Databricks workspace (Serverless or Standard compute).
- **Storage/registry**: Unity Catalog tables and MLflow Model Registry.

### Required libraries (install via compute side panel)
Install the following PyPI packages on the notebook’s compute (Compute side panel → Libraries/Packages → PyPI):
- `kagglehub`
- `databricks-feature-engineering`
- `scikit-learn`
- `databricks-sdk`

Notes:
- `mlflow`, `pyspark`, and Databricks utilities are available on Databricks.
- If using Serverless, use the right‑hand “Environment/Libraries” panel to add packages.

### Parameters (widgets)
All notebooks are parameterized using Databricks widgets. Set these at the top of each notebook before running.

- 1. Feature Store
  - `catalog`, `schema`, `dataset`
  - `kaggle_dataset_id` (e.g., `uciml/iris`), `kaggle_dataset_path` (e.g., `Iris.csv`)

- 2. Model Training
  - `catalog`, `schema`, `dataset`

- 3.1 Batch Inference
  - `catalog`, `schema`, `dataset`
  - `model_name` (e.g., `iris_rf_classifier`), `production_alias` (e.g., `champion`)

- 3.2 Realtime Inference
  - `catalog`, `schema`, `dataset`
  - `model_name` (e.g., `iris_rf_classifier`), `production_alias` (e.g., `champion`)
  - `endpoint_name` (e.g., `iris_endpoint`)

### How to run
1. Choose/attach compute. Install the packages listed above on that compute.
2. Open notebook 1, set widgets, then Run All. Repeat for notebooks 2 → 3.1 → 3.2.
3. Ensure the `catalog.schema.dataset` you choose is accessible and you have UC permissions to create tables and register models.

### Notebook summaries
- 1. Feature Store
  - Loads Iris from Kaggle via `kagglehub` into a Spark DataFrame, normalizes column names/types, writes a Delta table to Unity Catalog, and enforces `id` as primary key.

- 2. Model Training
  - Builds a training set with `FeatureEngineeringClient` and `FeatureLookup`, trains a `RandomForestClassifier`, logs metrics/artifacts to MLflow, registers the model to Unity Catalog, assigns the `challenger` alias, and (if it outperforms) promotes it to `champion`.

- 3.1 Batch Inference
  - Reads features from the UC table, converts to pandas, loads the registered model by alias (`models:/catalog.schema.model_name@alias`), and produces predictions in‑notebook.

- 3.2 Realtime Inference
  - Creates/updates a Databricks Model Serving endpoint for the registered model and sends a sample request using the Databricks SDK, printing predictions.
