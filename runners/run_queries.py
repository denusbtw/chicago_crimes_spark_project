from spark_utils import build_spark
from pipeline import run_analysis
from queries import get_lookup_tables


def run_queries():
    spark = build_spark("ChicagoCrimes-Queries")

    clean_path = "data/processed/chicago_crimes_clean"
    df_clean = spark.read.parquet(clean_path)

    severity_df, districts_df = get_lookup_tables(spark)

    run_analysis(
        df=df_clean,
        severity_df=severity_df,
        districts_df=districts_df,
        out_dir="output"
    )


if __name__ == "__main__":
    run_queries()