import os

from spark_utils import build_spark
from data_stats import (
    compute_correlation_matrix,
    plot_correlation_matrix,
    find_high_correlations,
)


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


def run_correlation():
    spark = build_spark("ChicagoCrimes-Correlation")

    clean_path = "data/processed/chicago_crimes_clean"
    df_clean = spark.read.parquet(clean_path)

    start_corr(df_clean, threshold=0.9, tag="clean")


if __name__ == "__main__":
    run_correlation()