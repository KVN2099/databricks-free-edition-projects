## Databricks Free Edition Practical Demos

Curated, hands‑on demonstrations of Databricks capabilities intended for aspiring professionals using Databricks Free Edition. Each demo focuses on a concrete use case and showcases best‑practice patterns across the Lakehouse: Unity Catalog, Feature Engineering, MLflow, and Model Serving.

### Who this is for
- **Learners** who want practical examples they can run end‑to‑end in a Free Edition workspace.
- **Practitioners** looking for reference implementations to adapt for their own projects.

### Repository structure
- `common/`
  - Reusable helpers and utilities, e.g. `import_kaggle_dataset.py`.
- `ML/`
  - Machine learning demos.
  - `iris-featurestore-mlflow-modelserving/` — End‑to‑end Iris classification using Feature Store, MLflow, Batch + Realtime inference.
    - See the project README in that folder for details.

### Prerequisites
- A Databricks workspace (Free Edition is sufficient).
- Access to or ability to use Unity Catalog (or the `workspace` catalog in Free Edition).
- A compute cluster or Serverless runtime. You must be able to install PyPI libraries on the attached compute via the right‑hand “Libraries/Packages” panel.

Recommended PyPI libraries (install on the compute running the notebooks):
- `kagglehub`
- `databricks-feature-engineering`
- `scikit-learn`
- `databricks-sdk`

Note: `mlflow`, `pyspark`, and Databricks utilities are provided by the Databricks runtime.

### How to use this repository
1. Clone/import this repo into Databricks Repos or import the notebooks directly.
2. Attach/choose compute. Install the recommended PyPI libraries on that compute.
3. Open a project folder (for example `ML/iris-featurestore-mlflow-modelserving/`).
4. Follow the project’s README. Most notebooks are parameterized with widgets and are designed to run in numeric order.

### Included projects
- Iris Feature Store + MLflow + Model Serving
  - Path: `ML/iris-featurestore-mlflow-modelserving/`
  - Summary: Loads the Iris dataset to Unity Catalog, builds features with the Feature Engineering client, trains and registers a model with MLflow, then runs batch and realtime inference via Databricks Model Serving.

### Contributing
Contributions and improvements are welcome. Please keep examples minimal, well‑documented, and runnable on Databricks Free Edition when possible.


