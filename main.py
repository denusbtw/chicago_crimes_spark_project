import glob
import os
import shutil
from collections import Counter
from pyspark.sql import SparkSession
from pyspark.sql.types import NumericType, IntegerType, DoubleType, BooleanType, TimestampType, StringType

from extractor import load_crime_data, validate_dataframe
from queries import *


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


def missing_and_duplicates_analysis(df):
    missing_summary = df.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c)
        for c in df.columns
    ])
    print("Пропущені значення по колонках:")
    missing_summary.show(truncate=False)

    total_rows = df.count()
    missing_percent = df.select([
        (F.count(F.when(F.col(c).isNull(), c)) / total_rows * 100).alias(c)
        for c in df.columns
    ])
    print("Відсоток пропусків по колонках:")
    missing_percent.show(truncate=False)

    duplicate_count = df.count() - df.dropDuplicates().count()
    print(f"Кількість дублікатів: {duplicate_count}")

    return df


def handle_missing(df):
    df = df.fillna({
        "Primary Type": "Unknown",
        "Description": "Unknown",
        "Location Description": "Unknown"
    })

    df = df.filter(
        (F.col("Date").isNotNull()) &
        (F.col("Latitude").isNotNull()) &
        (F.col("Longitude").isNotNull())
    )

    df = df.withColumn(
        "Ward",
        F.when(F.col("Ward").isNull() | (F.col("Ward") == 0), F.lit(-1)).otherwise(F.col("Ward"))
    )

    df = df.withColumn(
        "Community Area",
        F.when(F.col("Community Area").isNull() | (F.col("Community Area") == 0), F.lit(-1)).otherwise(F.col("Community Area"))
    )

    df = df.filter(F.col("District").isNotNull() & (F.col("District") != "0"))

    return df


def write_single_csv(df, path, filename):
    temp_dir = path + "_tmp"
    os.makedirs(temp_dir, exist_ok=True)

    df.coalesce(1).write.csv(temp_dir, header=True, mode="overwrite")

    tmp_csv = glob.glob(os.path.join(temp_dir, "*.csv"))[0]

    shutil.move(tmp_csv, os.path.join(path, filename))

    shutil.rmtree(temp_dir)


def run_analysis(df):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    queries = [
        ("Q1: Крадіжки на суму (>$500)", lambda: q1_high_value_thefts(df)),
        ("Q2: Побудове насильство без проведеного арешту", lambda: q2_domestic_violence_no_arrest(df)),
        ("Q3: Злочини, що відбувалися в ресторанах", lambda: q3_crimes_in_restaurants(df)),
        ("Q4: Злочини за 2026 рік", lambda: q4_crimes_by_year_2026(df)),
        ("Q5: Вуличні злочини в нічний час", lambda: q5_crimes_on_streets_at_night(df)),
        ("Q6: Випадки, пов'язані з наркотиками", lambda: q6_narcotics_cases(df)),
    ]

    for i, (title, query_func) in enumerate(queries, start=1):
        print(f"\n{'=' * 80}")
        print(f"Бізнес-питання: {title}")
        print(f"{'=' * 80}")

        result_df = query_func()
        result_df.explain()

        filename = f"Q{i}.csv"
        write_single_csv(result_df, "output", filename)


if __name__ == "__main__":
    spark = SparkSession.builder.appName("ChicagoCrimes").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    path = "data/chicago_crimes.csv"
    df = load_crime_data(spark, path)

    validate_dataframe(df)
    run_analysis(df)