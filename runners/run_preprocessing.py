import os

from extractor import load_crime_data, validate_dataframe
from data_processing import (
    drop_columns,
    extract_block_features,
    handle_missing,
    remove_structural_anomalies
)
from spark_utils import build_spark


def save_clean_dataset(df, out_path="data/processed/chicago_crimes_clean"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df.write \
        .mode("overwrite") \
        .parquet(out_path)

    print(f"Clean dataset saved to: {out_path}")


def run_preprocessing():
    spark = build_spark("ChicagoCrimes-Preprocessing")

    path = "data/chicago_crime.csv"
    df = load_crime_data(spark, path)

    df_clean = drop_columns(df)
    df_clean = extract_block_features(df_clean)
    df_clean = handle_missing(df_clean)
    df_clean = remove_structural_anomalies(df_clean)

    validate_dataframe(df_clean)
    save_clean_dataset(df_clean, out_path="data/processed/chicago_crimes_clean")


if __name__ == "__main__":
    run_preprocessing()