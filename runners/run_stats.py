import os
import sys

from extractor import load_crime_data
from spark_utils import build_spark
from data_stats import (
    general_dataset_statistics,
    numeric_statistics,
    feature_informativeness_analysis,
    missing_and_duplicates_analysis,
)


def start_stats(df, source_path: str, tag: str):
    os.makedirs("stats", exist_ok=True)

    with open(f"stats/dataset_stats_{tag}.txt", "w", encoding="utf-8") as f:
        old_stdout = sys.stdout
        sys.stdout = f
        try:
            total_rows, nulls_row = general_dataset_statistics(
                df,
                source_path=source_path,
                cache_df=False
            )
            numeric_statistics(df)
            missing_and_duplicates_analysis(
                df,
                total_rows=total_rows,
                nulls_row=nulls_row
            )
            feature_informativeness_analysis(
                df,
                total_rows=total_rows
            )
        finally:
            sys.stdout = old_stdout


def run_stats():
    spark = build_spark("ChicagoCrimes-Stats")

    raw_path = "data/chicago_crime.csv"
    clean_path = "data/processed/chicago_crimes_clean"

    df_raw = load_crime_data(spark, raw_path)
    start_stats(df_raw, source_path=raw_path, tag="raw")

    df_clean = spark.read.parquet(clean_path)
    start_stats(df_clean, source_path=clean_path, tag="clean")


if __name__ == "__main__":
    run_stats()