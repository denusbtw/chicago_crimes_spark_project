from spark_utils import build_spark
from famd_analysis import run_famd_analysis


def run_famd():
    spark = build_spark("ChicagoCrimes-FAMD")

    clean_path = "data/processed/chicago_crimes_clean"
    df_clean = spark.read.parquet(clean_path)

    numeric_cols = [
        "Beat",
        "District",
        "Ward",
        "Community Area",
        "Latitude",
        "Longitude"
    ]

    categorical_cols = [
        "Primary Type",
        "FBI Code",
        "Location Description",
        "Arrest",
        "Domestic"
    ]

    collapse_top_n = {
        "Location Description": 20
    }

    run_famd_analysis(
        df_clean,
        out_dir="famd_results",
        sample_fraction=0.1,
        sample_max_rows=50000,
        n_components=6,
        seed=42,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        collapse_top_n=collapse_top_n
    )


if __name__ == "__main__":
    run_famd()