## Data engineering demos

Examples here focus on **moving data through layers** on the Lakehouse: raw or lightly trusted data lands in **Delta**, then **declarative SQL** (Lakeflow / SDP-style patterns) expresses **bronze → silver → gold** transformations with clear ownership of freshness and quality.

### Projects

| Project | Idea | README |
| ------- | ---- | ------ |
| [NYC Taxi](NYC%20Taxi/) | Use the built-in **NYC yellow taxi** sample, persist as a Delta table, then layer **streaming** and **aggregated** tables for analytics. | [NYC Taxi/README.md](NYC%20Taxi/README.md) |

### Skills you will exercise

- **Declarative pipelines:** SQL that defines **streaming tables** and **materialized views** instead of only one-off notebook writes.
- **Medallion thinking:** bronze (raw), silver (cleaned/conformed), gold (aggregates for BI or downstream ML).
- **Unity Catalog** as the namespace for every layer so ACLs and lineage stay coherent.

### When to use these patterns

- You need **incremental** updates as new taxi files arrive (streaming), while analysts consume stable **gold** tables or views.
- You want SQL assets that can be **reviewed**, **versioned**, and **scheduled** like other Lakehouse jobs.

Open the project README for exact **run order** (setup notebook before pipeline SQL) and **customization** for your catalog.
