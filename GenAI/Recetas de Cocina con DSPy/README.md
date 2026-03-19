## Recetas de cocina con DSPy (RAG)

This demo walks through a **retrieval-augmented** workflow in Spanish: cooking **recipes** as the knowledge base, **Databricks Vector Search** for semantic retrieval, and **DSPy** to structure prompts and modules around that retrieval. It is aimed at learners who already know notebooks and want to see how Lakehouse data, embeddings, and LLM orchestration connect on one platform.

### What you will build

1. **Ground truth in the Lakehouse:** recipe records (and any file or volume paths you configure) live under **Unity Catalog**, not only in local files.
2. **A Vector Search endpoint and index:** text is embedded and indexed so queries retrieve the most relevant recipes (or chunks), not the whole corpus every time.
3. **A DSPy-oriented RAG step:** DSPy helps you treat “retrieve → compose answer” as a programmable flow rather than one-off string concatenation.

Conceptually:

```text
Recetas (Delta / UC) → embedding + sync → VS Index → DSPy (retrieve + generate)
```

### Run order

| Step | Notebook | Purpose |
| ---- | -------- | ------- |
| 1 | `1. Cargar datos de recetas.ipynb` | Load and persist recipe data; prepare text fields for downstream embedding. |
| 2 | `2. Crear VS Endpoint.ipynb` | Provision a **Vector Search endpoint** (throughput/capacity per your workspace limits). |
| 3 | `3. Crear VS Index.ipynb` | Create the index, point it at your source table or delta pipeline, run **sync** so vectors stay current. |
| 4 | `4. Crear RAG DSPy.ipynb` | Connect retrieval to DSPy (and related libraries) for question answering over recipes. |

Do not reorder: the index depends on data and schema from step 1; DSPy assumes a working endpoint and index from steps 2–3.

### Prerequisites

- **Unity Catalog** with permission to create or use the catalog/schema (and **volumes** if the notebooks store raw files there).
- **Vector Search** enabled for the workspace where you run this demo (check entitlement and region).
- **Compute** that can call embedding APIs or run local embedding code as written in the notebooks—some paths assume access to a **foundation model** or embedding model configured in the workspace.
- **Dependencies:** install from `requirements.txt` on the cluster or Serverless image. Notable pins include:
  - `databricks-vectorsearch` — index and query APIs
  - `dspy` / `dspy-ai` — declarative LLM programs
  - `pydantic` — structured configs
  - `mlflow` — optional experiment tracking if used in the notebooks
  - `databricks-agents` — if the sample wires Agent Framework pieces

Use `%pip install -r requirements.txt` in a setup cell or attach libraries at cluster level for reproducibility.

### Configuration tips

- Replace placeholder **catalog**, **schema**, **endpoint name**, and **index name** with values unique to you—shared workspaces collide easily.
- If sync is slow, reduce batch size or schedule syncs; for demos, a small slice of recipes is enough.
- Notebook prose and comments are in **Spanish**; UC object names can still use your team’s English convention (`main.recipes.silver_recipes`, etc.).

### Troubleshooting

- **403 / permission on Vector Search:** ask an admin to grant Vector Search entitlements and UC privileges on the backing tables.
- **Empty retrieval:** confirm the index **sync completed**, embedding column matches query embedding model, and filters are not excluding all rows.
- **DSPy version drift:** align `dspy` / `dspy-ai` with the notebook’s import style; breaking changes between major versions are common—use the repo’s `requirements.txt` as the source of truth.
