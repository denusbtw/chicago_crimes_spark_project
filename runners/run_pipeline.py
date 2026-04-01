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
        .option("header", True)
        .option("inferSchema", True)
        .option("sep", ";")
        .csv(str(DATA_DIR / "chicago_crime.csv"))
    )

    run_analysis(df, spark, out_dir=str(OUTPUT_DIR))

    spark.stop()


if __name__ == "__main__":
    run_pipeline()