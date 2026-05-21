import os
from collections import Counter
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.storagelevel import StorageLevel
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, IntegerType, DoubleType, BooleanType, TimestampType, StringType


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


def _null_counts_row(df):
    exprs = [F.count(F.when(F.col(c).isNull(), 1)).alias(c) for c in df.columns]
    return df.select(exprs).collect()[0]


def general_dataset_statistics(df, source_path=None, cache_df=False):
    if cache_df:
        df = df.persist(StorageLevel.MEMORY_AND_DISK)

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

    print("===== TOTAL NULL VALUES (BY COLUMN) =====")
    nulls_row = _null_counts_row(df)
    df.sparkSession.createDataFrame([nulls_row.asDict()]).show(truncate=False)

    return row_count, nulls_row


def numeric_statistics(df):
    numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]
    print("NUMERIC COLUMNS:", numeric_cols)
    if numeric_cols:
        df.select(numeric_cols).describe().show()


def feature_informativeness_analysis(df, total_rows=None):
    print("===== FEATURE INFORMATIVENESS ANALYSIS =====")
    if total_rows is None:
        total_rows = df.count()

    string_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
    int_double_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, (IntegerType, DoubleType))]

    if string_cols:
        distinct_exprs = [F.countDistinct(F.col(c)).alias(c) for c in string_cols]
        distinct_row = df.select(distinct_exprs).collect()[0]
        distinct_map = distinct_row.asDict()
    else:
        distinct_map = {}

    for field in df.schema.fields:
        col_name = field.name
        dtype = field.dataType

        print(f"\n--- Column: {col_name} ({dtype.simpleString()}) ---")

        if isinstance(dtype, StringType):
            distinct_count = distinct_map.get(col_name)

            top_freq = (
                df.groupBy(col_name)
                  .count()
                  .orderBy(F.desc("count"))
                  .limit(1)
                  .collect()
            )

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


def missing_and_duplicates_analysis(df, total_rows=None, nulls_row=None):
    if total_rows is None:
        total_rows = df.count()

    if nulls_row is None:
        nulls_row = _null_counts_row(df)

    print("Пропущені значення по колонках:")
    df.sparkSession.createDataFrame([nulls_row.asDict()]).show(truncate=False)

    print("Відсоток пропусків по колонках:")
    percent_raw = {
        k: (float(v) / float(total_rows) * 100.0)
        for k, v in nulls_row.asDict().items()
    }

    percent_formatted = {}
    for k, v in percent_raw.items():
        if v >= 0.1:
            percent_formatted[k] = f"~{v:.1f}%"
        elif v > 0:
            percent_formatted[k] = f"{v:.5f}%"
        else:
            percent_formatted[k] = "0.0%"

    df.sparkSession.createDataFrame([percent_formatted]).show(truncate=False)

    duplicate_count = total_rows - df.dropDuplicates().count()
    print(f"Кількість дублікатів: {duplicate_count}")

def compute_correlation_matrix(df):
    import pandas as pd
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.stat import Correlation
    from pyspark.sql.types import NumericType

    numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]

    if len(numeric_cols) < 2:
        print("Not enough numeric columns for correlation.")
        return None

    assembler = VectorAssembler(
        inputCols=numeric_cols,
        outputCol="features",
        handleInvalid="skip"
    )

    vector_df = assembler.transform(df.select(numeric_cols)).select("features")

    corr_matrix = Correlation.corr(vector_df, "features", method="pearson").head()[0]
    corr_df = pd.DataFrame(corr_matrix.toArray(), index=numeric_cols, columns=numeric_cols)

    return corr_df

def plot_correlation_matrix(
    corr_df,
    out_path,
    title="Correlation Matrix",
    figsize=(12, 10),
    decimals=1,
    font_size=9
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patheffects as path_effects

    data = corr_df.values
    n = data.shape[0]

    plt.figure(figsize=figsize)

    im = plt.imshow(data, interpolation="nearest", cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.title(title)

    ticks = np.arange(n)
    plt.xticks(ticks, corr_df.columns, rotation=90)
    plt.yticks(ticks, corr_df.index)

    for i in range(n):
        for j in range(n):
            v = data[i, j]
            if np.isnan(v):
                txt = "NaN"
            else:
                txt = f"{v:.{decimals}f}"

            t = plt.text(
                j, i, txt,
                ha="center",
                va="center",
                fontsize=font_size,
                fontweight="bold",
                color="white"
            )
            t.set_path_effects([
                path_effects.Stroke(linewidth=1.6, foreground="black"),
                path_effects.Normal()
            ])

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def find_high_correlations(corr_df, threshold=0.9):
    high_corr_pairs = []

    for i in range(len(corr_df.columns)):
        for j in range(i + 1, len(corr_df.columns)):
            val = corr_df.iloc[i, j]
            if abs(val) > threshold:
                high_corr_pairs.append(
                    (corr_df.columns[i], corr_df.columns[j], val)
                )

    return high_corr_pairs

from pyspark.sql import functions as F


def check_structural_anomalies(df):

    total_rows = df.count()
    print(f"Total rows: {total_rows}")
    print("===== STRUCTURAL ANOMALY CHECK =====")

    # 1. Координати поза межами Chicago
    invalid_coordinates = df.filter(
        (F.col("Latitude").isNotNull()) &
        (
            (F.col("Latitude") < 41.5) |
            (F.col("Latitude") > 42.1) |
            (F.col("Longitude") < -88.0) |
            (F.col("Longitude") > -87.4)
        )
    ).count()

    print(f"Invalid coordinates: {invalid_coordinates}")

    # 2. Некоректні District (Chicago має 1–31)
    invalid_district = df.filter(
        (F.col("District").isNotNull()) &
        (
            (F.col("District") < 1) |
            (F.col("District") > 31)
        )
    ).count()

    print(f"Invalid District values: {invalid_district}")

    # 3. Некоректні Ward (1–50)
    invalid_ward = df.filter(
        (F.col("Ward").isNotNull()) &
        (
            (F.col("Ward") < 1) |
            (F.col("Ward") > 50)
        )
    ).count()

    print(f"Invalid Ward values: {invalid_ward}")

    # 4. Некоректні Community Area (1–77)
    invalid_community = df.filter(
        (F.col("Community Area").isNotNull()) &
        (
            (F.col("Community Area") < 1) |
            (F.col("Community Area") > 77)
        )
    ).count()

    print(f"Invalid Community Area values: {invalid_community}")

    # 5. Підозрілі дати (до 2000 року)
    invalid_dates = df.filter(
        F.year("Date") < 2000
    ).count()

    print(f"Suspicious dates (< 2000): {invalid_dates}")

    print("===== CHECK COMPLETE =====")