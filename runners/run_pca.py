from spark_utils import build_spark
from pca_analysis import run_pca_analysis


def run_pca():
    spark = build_spark("ChicagoCrimes-PCA")

    clean_path = "data/processed/chicago_crimes_clean"
    df_clean = spark.read.parquet(clean_path)

    run_pca_analysis(
        df_clean,
        out_dir="pca_results",
        n_components=10,
        sample_fraction=0.1,
        sample_max_rows=50000,
        seed=42
    )


if __name__ == "__main__":
    run_pca()