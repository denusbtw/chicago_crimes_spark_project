import os
from collections import Counter
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, IntegerType, DoubleType, BooleanType, TimestampType, StringType

from extractor import load_crime_data, validate_dataframe


def _dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


def _human_gb(size_bytes: int) -> float:
    return round(size_bytes / (1024 ** 3), 3)


def general_dataset_statistics(df, source_path = "data/chicago_crimes.csv"):
    print("===== SCHEMA =====")
    df.printSchema()

    print("===== ROW COUNT =====")
    row_count = df.count()
    print(row_count)

    print("===== COLUMN COUNT =====")
    col_count = len(df.columns)
    print(col_count)

    print("===== SAMPLE =====")
    df.show(5, truncate=False)

    print("===== DATA TYPES DISTRIBUTION =====")
    types = [field.dataType.simpleString() for field in df.schema.fields]
    type_counts = Counter(types)
    for dtype, count in type_counts.items():
        print(f"{dtype}: {count}")

    print("===== NUMERIC VS NON-NUMERIC =====")
    numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Non-numeric columns: {col_count - len(numeric_cols)}")

    print("===== FILE SIZE ON DISK =====")
    if source_path is None:
        print("Source path not provided.")
    else:
        if os.path.isfile(source_path):
            size_bytes = os.path.getsize(source_path)
            print(f"Source file size: {_human_gb(size_bytes)} GB")
        elif os.path.isdir(source_path):
            size_bytes = _dir_size_bytes(source_path)
            print(f"Source directory size: {_human_gb(size_bytes)} GB")
        else:
            print("Source path not found.")

    print("===== TOTAL NULL VALUES =====")
    total_nulls = df.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c)
        for c in df.columns
    ])
    total_nulls.show(truncate=False)


def numeric_statistics(df):
    numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]
    print("NUMERIC COLUMNS:", numeric_cols)
    if numeric_cols:
        df.select(numeric_cols).describe().show()


def drop_columns(df):
    columns_to_drop = [
        "ID",
        "Case Number",
        "X Coordinate",
        "Y Coordinate",
        "Location",
        "Year",
    ]
    return df.drop(*columns_to_drop)


def feature_informativeness_analysis(df):
    print("===== FEATURE INFORMATIVENESS ANALYSIS =====")

    total_rows = df.count()

    for field in df.schema.fields:
        col_name = field.name
        dtype = field.dataType

        print(f"\n--- Column: {col_name} ({dtype.simpleString()}) ---")

        if isinstance(dtype, StringType):

            distinct_count = df.select(col_name).distinct().count()

            top_freq = (
                df.groupBy(col_name)
                  .count()
                  .orderBy(F.desc("count"))
                  .limit(1)
            ).collect()

            if top_freq:
                top_value = top_freq[0][0]
                top_count = top_freq[0][1]
                ratio = round(top_count / total_rows, 4)

                print(f"Distinct values: {distinct_count}")
                print(f"Most frequent value: {top_value}")
                print(f"Dominance ratio: {ratio}")
            else:
                print("No data.")

        elif isinstance(dtype, (IntegerType, DoubleType)):

            stats = df.select(
                F.variance(col_name).alias("variance"),
                F.stddev(col_name).alias("stddev")
            ).collect()[0]

            print(f"Variance: {stats['variance']}")
            print(f"Stddev: {stats['stddev']}")

        elif isinstance(dtype, (BooleanType, TimestampType)):
            print("Skipped (boolean/timestamp).")

        else:
            print("Skipped (other type).")


if __name__ == "__main__":
    spark = SparkSession.builder.appName("ChicagoCrimes").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    path = "data/chicago_crimes.csv"
    df = load_crime_data(spark, path)

    validate_dataframe(df)
    feature_informativeness_analysis(df)
