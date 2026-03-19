## Machine learning demos

Projects here focus on **tabular ML** on the Databricks Lakehouse: training data in **Unity Catalog**, optional **Databricks Feature Store** for training-time joins, **MLflow** for the model lifecycle, and **Model Serving** for production-style inference.

### Projects

| Project | What it demonstrates | README |
| ------- | -------------------- | ------ |
| [iris-featurestore-mlflow-modelserving](iris-featurestore-mlflow-modelserving/) | End-to-end Iris: Kaggle → UC Delta, `FeatureEngineeringClient` + `FeatureLookup`, sklearn model, UC Model Registry aliases (`challenger` / `champion`), batch scoring, then a **serving endpoint** with the Databricks SDK. | [iris-featurestore-mlflow-modelserving/README.md](iris-featurestore-mlflow-modelserving/README.md) |

### Skills you will exercise

- Wiring **widgets** so the same notebooks work across dev/test catalogs.
- Treating UC as the **single source of truth** for features and registered models.
- Comparing **batch** inference (Spark/pandas + `models:/…@alias`) with **online** inference (REST).

### Before you start

- Confirm you can **create tables** and **register models** in the catalog you choose.
- Install libraries listed in the project README on your cluster or Serverless environment.

If you add a new ML demo, keep the same pattern: one README per project with run order, widget list, and required PyPI packages.
