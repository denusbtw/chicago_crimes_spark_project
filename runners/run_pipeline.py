from pathlib import Path
from pyspark.sql import SparkSession
from pipeline import run_analysis


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"


def run_pipeline():
    spark = (
        SparkSession.builder
        .appName("ChicagoCrime")
        .getOrCreate()
    )

    df = (
        spark.read
        .parquet(str(DATA_DIR / "processed" / "chicago_crimes_clean"))
    )

    run_analysis(df, spark, out_dir=str(OUTPUT_DIR))
    spark.stop()


if __name__ == "__main__":
    run_pipeline()