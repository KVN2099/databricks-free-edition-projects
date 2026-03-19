## Common utilities

This folder holds **reusable Python** intended to be imported from notebooks or small jobs elsewhere in the repo. The goal is to avoid duplicating “download dataset → normalize → write Delta” logic in every demo.

### `import_kaggle_dataset.py`

**Purpose:** `KaggleDatasetImporter` downloads a dataset through **[kagglehub](https://github.com/Kaggle/kagglehub)** (pandas adapter path), then materializes it as a **Delta** table using the notebook’s active `SparkSession`. That matches how many Lakehouse tutorials want data: one canonical table in **Unity Catalog** instead of scattered CSVs on DBFS.

**When to use it**

- You have a Kaggle dataset ID (e.g. `uciml/iris`) and know the path to the file inside the archive.
- You want consistent column handling and `saveAsTable` behavior across demos.

**Install**

KaggleHub needs the pandas extra for the typical pandas-based load path:

```text
%pip install kagglehub[pandas-datasets]
```

If Kaggle asks for credentials, configure them per [KaggleHub authentication](https://github.com/Kaggle/kagglehub#authentication) (API token or supported methods).

**Minimal example**

```python
from pyspark.sql import SparkSession
from common.import_kaggle_dataset import KaggleDatasetImporter

spark = SparkSession.builder.getOrCreate()
importer = KaggleDatasetImporter(spark)
df = importer.import_to_delta(
    kaggle_dataset_id="owner/dataset",
    kaggle_dataset_path="files/data.csv",
    table_name="catalog.schema.table_name",
)
```

**What to read next**

- Class and method docstrings in `import_kaggle_dataset.py` for parameters (overwrite mode, optional transformations, logging).
- The Iris demo under `ML/iris-featurestore-mlflow-modelserving/`, which uses a similar Kaggle load pattern in-line in the Feature Store notebook.

**Operational tips**

- Prefer a dedicated **schema** per demo so you can `DROP` or overwrite without touching production tables.
- If import fails with path errors, verify the **exact** file path inside the Kaggle dataset on the dataset’s page or by listing the downloaded folder once locally.
