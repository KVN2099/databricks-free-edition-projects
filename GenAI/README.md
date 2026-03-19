## Generative AI demos

This section groups examples where **large language models**, **retrieval**, or **Vector Search** meet the Lakehouse: data lands in Unity Catalog, embeddings and indexes are managed as first-class platform objects, and notebooks show how to go from raw text to a **queryable RAG-style** application pattern.

### Projects

| Project | Focus | README |
| ------- | ------ | ------ |
| [Recetas de Cocina con DSPy](Recetas%20de%20Cocina%20con%20DSPy/) | Spanish recipe corpus → embeddings → **Vector Search** endpoint and index → **DSPy** program for RAG. | [Recetas …/README.md](Recetas%20de%20Cocina%20con%20DSPy/README.md) |
| [Job Recommender](Job%20Recommender/) | **Ingest** tabular job data into UC, then **profile** the table (schema, samples, data quality). Foundation for ranking or recommendation models later. | [Job Recommender/README.md](Job%20Recommender/README.md) |

### Skills you will exercise

- Choosing **catalog/schema** (and volumes when storing files or chunks).
- Creating and syncing **Vector Search** assets (where the product is enabled).
- Connecting retrieval to an orchestration layer (**DSPy** in the Recetas project).
- Treating UC tables as the **contract** between data engineering and GenAI notebooks.

### Workspace considerations

- **Vector Search**, **foundation model** endpoints, and **Serverless** GPU may vary by edition and region—confirm in your admin docs before relying on a specific notebook cell.
- Spanish notebook titles and markdown in **Recetas** are intentional; variable names and UC objects can still follow your English naming standards.

Start with the README inside the project you chose; notebook order is documented there.
