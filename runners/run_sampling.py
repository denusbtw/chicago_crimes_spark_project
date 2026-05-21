from spark_utils import build_spark
from sampling_analysis import run_sampling_analysis


def run_sampling():
    spark = build_spark("ChicagoCrimes-Sampling")

    clean_path = "data/processed/chicago_crimes_clean"
    df_clean = spark.read.parquet(clean_path)

    run_sampling_analysis(
        df_clean,
        out_dir="sampling_results",
        target_cols=("Domestic", "Arrest"),
        methods=("oversampling", "undersampling"),
        seed=42,
        pca_sample_max_rows=50000
    )


if __name__ == "__main__":
    run_sampling()