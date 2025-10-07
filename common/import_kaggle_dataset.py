"""Utilities for importing Kaggle datasets and writing them as Delta tables.

Install dependencies as needed (example):
    %pip install kagglehub[pandas-datasets]

This module defines `KaggleDatasetImporter`, a class you can import and use from
other Python programs to load a Kaggle dataset (via pandas) and persist it as a
Delta table using an existing SparkSession.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional, TYPE_CHECKING

import kagglehub
from kagglehub import KaggleDatasetAdapter

if TYPE_CHECKING:  # Avoid hard dependency on pyspark at import time for type hints
    from pyspark.sql import DataFrame, SparkSession


logger = logging.getLogger(__name__)


class KaggleDatasetImporter:
    """Importer for Kaggle datasets to Delta tables.

    Usage example:
        from pyspark.sql import SparkSession
        from common.import_kaggle_dataset import KaggleDatasetImporter

        spark = SparkSession.builder.getOrCreate()
        importer = KaggleDatasetImporter(spark)
        df = importer.import_to_delta(
            kaggle_dataset_id="owner/dataset",
            kaggle_dataset_path="files/data.csv",
            table_name="schema.table_name",
        )
    """

    def __init__(
        self,
        spark: "SparkSession",
        enable_change_data_feed: bool = True,
    ) -> None:
        self.spark = spark
        self.enable_change_data_feed = enable_change_data_feed

    def import_to_delta(
        self,
        kaggle_dataset_id: str,
        kaggle_dataset_path: str,
        table_name: str,
        *,
        mode: str = "overwrite",
        columns_to_drop: Optional[Iterable[str]] = ("Unnamed: 6",),
        sql_query: Optional[str] = None,
        pandas_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "DataFrame":
        """Load a Kaggle dataset to pandas, convert to Spark, and save as Delta.

        Args:
            kaggle_dataset_id: Like "owner/dataset".
            kaggle_dataset_path: Path within the dataset, e.g. "files/data.csv".
            table_name: Fully qualified Delta table name to write to.
            mode: Spark save mode (default "overwrite").
            columns_to_drop: Optional columns to drop if present. Defaults to
                ("Unnamed: 6",) for convenience with typical CSVs where an
                unnamed index column appears.
            sql_query: Optional SQL query if the adapter supports it.
            pandas_kwargs: Optional extra kwargs forwarded to pandas loader.

        Returns:
            The Spark DataFrame that was written.
        """
        pandas_kwargs = pandas_kwargs or {}

        # Load the latest version of the dataset as a pandas DataFrame
        pdf = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            kaggle_dataset_id,
            kaggle_dataset_path,
            sql_query=sql_query,
            pandas_kwargs=pandas_kwargs,
        )

        # Convert pandas DataFrame to Spark DataFrame
        df = self.spark.createDataFrame(pdf)

        if columns_to_drop:
            present = [c for c in columns_to_drop if c in df.columns]
            if present:
                df = df.drop(*present)

        # Write to Delta table
        df.write.format("delta").mode(mode).saveAsTable(table_name)

        # Optionally enable Change Data Feed on the target table
        if self.enable_change_data_feed:
            try:
                self.spark.sql(
                    f"ALTER TABLE {table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
                )
            except Exception as exc:  # noqa: BLE001 - log and continue since CDF might be already enabled/unsupported
                logger.debug(
                    "Could not enable Change Data Feed for table %s: %s",
                    table_name,
                    exc,
                )

        return df


__all__ = ["KaggleDatasetImporter"]