## Databricks Free Edition Practical Demos

This repository collects **hands-on, end-to-end examples** you can run in a Databricks workspace (including **Free Edition** where the underlying features are available). Each project is scoped to a real pattern—loading data into **Unity Catalog**, training or serving models, streaming or declarative pipelines, and GenAI with **Vector Search**—so you can copy the structure into your own work rather than starting from a blank notebook.

### What you can practice here

- **Governance and storage:** Delta tables, volumes, and UC three-part names (`catalog.schema.object`).
- **Machine learning:** Feature Store lookups, MLflow registration and aliases, batch scoring, HTTP inference via **Model Serving**.
- **Deep learning:** Image pipelines (X-ray), classic MNIST with Keras/TensorFlow and MLflow.
- **Data engineering:** Bronze → silver → gold layering with **Lakeflow** SQL (streaming tables + materialized views).
- **Generative AI:** Embeddings, Vector Search endpoints/indexes, and **DSPy** for RAG-style flows.
- **Shared code:** Reusable Kaggle → Delta ingestion in `common/`.

Not every product capability exists on every SKU or region; each project README calls out what to verify in your workspace.

### Who this is for

- **Learners** who want runnable notebooks with a documented order and parameters.
- **Practitioners** who need a reference layout (widgets, UC names, bundle variables) to adapt for customer or personal projects.

### Repository structure

| Area | Path | What’s inside |
| ---- | ---- | ------------- |
| Shared utilities | [`common/`](common/) | Python helpers (e.g. Kaggle → Delta). [Details →](common/README.md) |
| Machine learning | [`ML/`](ML/) | Classical ML on the Lakehouse. [Details →](ML/README.md) |
| Deep learning | [`DL/`](DL/) | CNNs, TensorFlow/Keras. [Details →](DL/README.md) |
| Generative AI | [`GenAI/`](GenAI/) | Vector Search, RAG, ingest demos. [Details →](GenAI/README.md) |
| Data engineering | [`Data Engineering/`](Data%20Engineering/) | Medallion / Lakeflow examples. [Details →](Data%20Engineering/README.md) |

### Included projects (summary)

| Project | Path | In one sentence |
| ------- | ---- | ---------------- |
| Iris + Feature Store + MLflow + Serving | [`ML/iris-featurestore-mlflow-modelserving/`](ML/iris-featurestore-mlflow-modelserving/) | Tabular Iris data through Feature Store training, MLflow aliases, batch and REST inference. |
| MNIST classifier | [`DL/mnist-app-classifier/`](DL/mnist-app-classifier/) | Digit classification with Keras/TensorFlow and MLflow tracking. |
| Chest X-ray classification | [`DL/x-ray-classification/`](DL/x-ray-classification/) | Medical imaging EDA → preprocessing → CNN training; optional **Databricks Asset Bundle**. |
| NYC Taxi medallion | [`Data Engineering/NYC Taxi/`](Data%20Engineering/NYC%20Taxi/) | Built-in taxi sample → Delta, then declarative bronze/silver/gold pipeline. |
| Recetas RAG (DSPy) | [`GenAI/Recetas de Cocina con DSPy/`](GenAI/Recetas%20de%20Cocina%20con%20DSPy/) | Spanish recipe corpus, Vector Search, DSPy for RAG. |
| Job recommender | [`GenAI/Job Recommender/`](GenAI/Job%20Recommender/) | Ingest job data and inspect quality in Unity Catalog. |

Each folder has its own README: **notebook order**, **widgets**, **libraries**, and **customization** notes.

### Prerequisites (workspace-wide)

- A Databricks workspace; use **Unity Catalog** (often the `workspace` catalog on Free Edition).
- **Compute:** interactive cluster or Serverless, depending on the demo and your entitlements.
- **Libraries:** install per project README via the cluster UI or `%pip` in the first notebook cell. Baseline packages that appear often:
  - `kagglehub` — public Kaggle datasets without manual download
  - `databricks-feature-engineering` — Feature Store client (Iris demo)
  - `scikit-learn`, `databricks-sdk` — training and serving APIs

Runtime images already include **PySpark**, **MLflow** client, and Databricks helpers; only add what the README lists.

### How to use this repository

1. **Get the code in:** clone with **Databricks Repos**, or upload/export notebooks manually.
2. **Pick one project** and read its README end-to-end (order matters when tables and models depend on earlier steps).
3. **Attach compute** and install libraries before running “Run all.”
4. **Rename** catalogs, schemas, endpoints, and experiments to your naming convention so you do not collide with other users in a shared workspace.

### Contributing

Improvements are welcome: keep examples **minimal**, **documented**, and **runnable** on Free Edition when the required features are available.
