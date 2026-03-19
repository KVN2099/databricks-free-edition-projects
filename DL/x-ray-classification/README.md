## Chest X-ray classification

This project is an **image ML** pipeline on the Lakehouse: radiology-style **chest X-rays** land in **Unity Catalog** (via tables and/or **volumes**), you explore and clean the data in notebooks, then train a **neural network** with metrics and runs suitable for production-style tracking. A **Databricks Asset Bundle** ties experiments, jobs, and UC metadata to versioned YAML so you can reproduce or promote work across environments.

### What you will learn

- How to organize **non-tabular** assets (DICOM or image files) alongside **metadata** in UC.
- A conventional **notebook sequence**: load → EDA → preprocessing → training—with clear handoffs between steps.
- How **MLflow** records experiments when training deep models on Databricks.
- Optional **bundle-driven** deployment: one place to define variables (catalog, volume, experiment name) and deploy jobs consistently.

### Run order (`src/` notebooks)

| Step | Notebook | Purpose |
| ---- | -------- | ------- |
| 1 | `1. Load data to UC.ipynb` | Download or attach the dataset (e.g. Kaggle chest X-ray splits), write paths or features to UC **volumes**/tables, establish labels for classification. |
| 2 | `2. EDA.ipynb` | Class balance, sample visualizations, path integrity, label noise—decide filtering rules before expensive training. |
| 3 | `3. Preprocessing.ipynb` | Resize, normalization, augmentations, train/val/test splits—outputs consumable by the training notebook (paths, Delta tables, or in-memory patterns as coded). |
| 4 | `4. Neural Network Training.ipynb` | Build the CNN (or transfer learning head), train with GPU where available, log metrics and artifacts to **MLflow**; align experiment name with bundle variables if you use the bundle. |

Run strictly **1 → 4** unless you already have equivalent UC objects from a prior run.

### Asset Bundle (`databricks.yml` + `resources/`)

The bundle centralizes **workspace host**, **deployment path**, and **variables** such as:

- **catalog** / **schema** — where UC tables live
- **table** — cleaned or feature table name
- **volume** — raw or processed image storage
- **experiment** — MLflow experiment name

**Before you deploy**

1. Replace **personal** defaults in `databricks.yml` (host, `root_path`, catalog/schema defaults) with **your** workspace and UC layout.
2. Run `databricks bundle validate` from this directory to catch schema errors.
3. Use `databricks bundle deploy` when you are ready to push jobs and resources defined under `resources/*.yml`.

Keep secrets (Kaggle tokens, cloud keys) in **secret scopes**, not in YAML.

### Prerequisites

- **GPU** recommended for training (cluster policy or Serverless GPU if available in your region).
- **Unity Catalog** permissions on the catalog, schema, volume, and MLflow experiment you configure.
- **Dataset access:** many chest X-ray datasets require **Kaggle** acceptance of license terms; configure credentials and paths as in notebook 1.
- **Dependencies:** `requirements.txt` is **large** (full DL stack). For exploratory runs, install only what the active notebook imports, or use a **Databricks ML/DL** runtime image that already includes PyTorch/TensorFlow and trim duplicates.

### Troubleshooting

- **OOM during training:** lower batch size, image resolution, or use mixed precision if the code supports it.
- **Slow volume reads:** coalesce small files or cache preprocessed tensors to Delta where appropriate.
- **Bundle deploy fails:** verify CLI auth (`databricks auth`), host URL, and that `root_path` exists or is creatable for your user.
