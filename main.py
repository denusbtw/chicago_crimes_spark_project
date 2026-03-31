import os
import sys
from pyspark.sql import SparkSession

from extractor import load_crime_data, validate_dataframe
from data_processing import drop_columns, handle_missing, remove_structural_anomalies
from data_stats import (
    general_dataset_statistics,
    numeric_statistics,
    feature_informativeness_analysis,
    missing_and_duplicates_analysis,
    compute_correlation_matrix,
    plot_correlation_matrix,
    find_high_correlations,
)
from pipeline import run_analysis
from queries import get_lookup_tables


def start_stats(df, source_path: str, tag: str):
    os.makedirs("stats", exist_ok=True)

    with open(f"stats/dataset_stats_{tag}.txt", "w", encoding="utf-8") as f:
        old_stdout = sys.stdout
        sys.stdout = f
        try:
            total_rows, nulls_row = general_dataset_statistics(df, source_path=source_path, cache_df=False)
            numeric_statistics(df)
            missing_and_duplicates_analysis(df, total_rows=total_rows, nulls_row=nulls_row)
            feature_informativeness_analysis(df, total_rows=total_rows)
        finally:
            sys.stdout = old_stdout


def start_corr(df, threshold=0.9, tag="clean"):
    os.makedirs("stats", exist_ok=True)

    corr_df = compute_correlation_matrix(df)
    if corr_df is None:
        return

    corr_df.to_csv(f"stats/corr_matrix_{tag}.csv", index=True)
    plot_correlation_matrix(corr_df, f"stats/corr_matrix_{tag}.png")

    pairs = find_high_correlations(corr_df, threshold=threshold)
    with open(f"stats/high_correlations_{tag}.txt", "w", encoding="utf-8") as f:
        f.write(f"High correlations (|r| > {threshold})\n")
        if pairs:
            for a, b, r in pairs:
                f.write(f"{a} <-> {b}: {r}\n")
        else:
            f.write("No high-correlation pairs found.\n")


def build_spark():
    py = sys.executable

    os.environ.setdefault("PYSPARK_PYTHON", py)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", py)
    os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("ChicagoCrimes")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")
    return spark


def main():
    spark = build_spark()

    path = "data/chicago_crimes.csv"
    df = load_crime_data(spark, path)

    start_stats(df, source_path=path, tag="raw")

    df_clean = drop_columns(df)
    df_clean = handle_missing(df_clean)
    df_clean = remove_structural_anomalies(df_clean)

    validate_dataframe(df_clean)

    start_stats(df_clean, source_path=path, tag="clean")
    start_corr(df_clean, threshold=0.9, tag="clean")

    severity_df, districts_df = get_lookup_tables(spark)

    run_analysis(df_clean, severity_df, districts_df)


if __name__ == "__main__":
    main()