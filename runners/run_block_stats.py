import os

from pyspark.sql import functions as F
from pyspark.sql import Window

from spark_utils import build_spark


def _write_top_values(df, col_name, out_file, top_n=30):
    rows = (
        df.groupBy(col_name)
        .count()
        .orderBy(F.desc("count"))
        .limit(top_n)
        .collect()
    )

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"===== TOP {top_n} VALUES: {col_name} =====\n\n")
        for row in rows:
            value = row[col_name]
            count = row["count"]
            f.write(f"{value}: {count}\n")


def run_block_stats():
    spark = build_spark("ChicagoCrimes-BlockStats")

    clean_path = "data/processed/chicago_crimes_clean"
    out_dir = "stats/block_stats"
    os.makedirs(out_dir, exist_ok=True)

    df = spark.read.parquet(clean_path)

    cols = ["street_name", "street_type", "direction", "block_number"]

    total_rows = df.count()

    null_counts_before = (
        df.select([
            F.count(F.when(F.col(c).isNull(), 1)).alias(c)
            for c in cols
        ])
        .collect()[0]
        .asDict()
    )

    exact_mode = (
        df.filter(
            F.col("block_number").isNotNull() &
            F.col("street_name").isNotNull() &
            F.col("street_type").isNotNull() &
            F.col("direction").isNotNull() &
            F.col("District").isNotNull()
        )
        .groupBy("street_name", "street_type", "direction", "District", "block_number")
        .count()
    )

    exact_window = Window.partitionBy(
        "street_name", "street_type", "direction", "District"
    ).orderBy(F.desc("count"), F.asc("block_number"))

    exact_mode = (
        exact_mode
        .withColumn("rn", F.row_number().over(exact_window))
        .filter(F.col("rn") == 1)
        .select(
            "street_name",
            "street_type",
            "direction",
            "District",
            F.col("block_number").alias("block_number_imputed_exact")
        )
    )

    fallback_mode = (
        df.filter(
            F.col("block_number").isNotNull() &
            F.col("street_name").isNotNull() &
            F.col("street_type").isNotNull() &
            F.col("direction").isNotNull()
        )
        .groupBy("street_name", "street_type", "direction", "block_number")
        .count()
    )

    fallback_window = Window.partitionBy(
        "street_name", "street_type", "direction"
    ).orderBy(F.desc("count"), F.asc("block_number"))

    fallback_mode = (
        fallback_mode
        .withColumn("rn", F.row_number().over(fallback_window))
        .filter(F.col("rn") == 1)
        .select(
            "street_name",
            "street_type",
            "direction",
            F.col("block_number").alias("block_number_imputed_fallback")
        )
    )

    missing_before_df = (
        df.filter(F.col("block_number").isNull())
        .select(
            "Block",
            "street_name",
            "street_type",
            "direction",
            "block_number",
            "District",
            "Ward",
            "Community Area",
            "Beat",
            "Latitude",
            "Longitude",
            "Primary Type",
            "Description"
        )
    )

    missing_before_count = missing_before_df.count()

    if missing_before_count > 0:
        missing_before_df.toPandas().to_csv(
            os.path.join(out_dir, "missing_block_number_rows_before.csv"),
            index=False
        )

    df = df.join(
        exact_mode,
        on=["street_name", "street_type", "direction", "District"],
        how="left"
    )

    df = df.join(
        fallback_mode,
        on=["street_name", "street_type", "direction"],
        how="left"
    )

    df = df.withColumn(
        "block_number_original",
        F.col("block_number")
    )

    df = df.withColumn(
        "block_number",
        F.when(
            F.col("block_number").isNull(),
            F.coalesce(
                F.col("block_number_imputed_exact"),
                F.col("block_number_imputed_fallback")
            )
        ).otherwise(F.col("block_number"))
    )

    imputed_rows_df = (
        df.filter(
            F.col("block_number_original").isNull() &
            F.col("block_number").isNotNull()
        )
        .select(
            "Block",
            "street_name",
            "street_type",
            "direction",
            "District",
            "block_number_original",
            "block_number_imputed_exact",
            "block_number_imputed_fallback",
            "block_number"
        )
    )

    imputed_count = imputed_rows_df.count()

    if imputed_count > 0:
        imputed_rows_df.toPandas().to_csv(
            os.path.join(out_dir, "imputed_block_number_rows.csv"),
            index=False
        )

    still_missing_df = (
        df.filter(F.col("block_number").isNull())
        .select(
            "Block",
            "street_name",
            "street_type",
            "direction",
            "District",
            "Ward",
            "Community Area",
            "Beat",
            "Latitude",
            "Longitude",
            "Primary Type",
            "Description"
        )
    )

    still_missing_count = still_missing_df.count()

    if still_missing_count > 0:
        still_missing_df.toPandas().to_csv(
            os.path.join(out_dir, "missing_block_number_rows_after.csv"),
            index=False
        )

    df = df.drop(
        "block_number_imputed_exact",
        "block_number_imputed_fallback",
        "block_number_original"
    )

    updated_path = "data/processed/chicago_crimes_clean"
    temp_path = "data/processed/chicago_crimes_clean_block_fix_tmp"

    df.write.mode("overwrite").parquet(temp_path)

    import shutil
    if os.path.exists(updated_path):
        shutil.rmtree(updated_path)
    os.rename(temp_path, updated_path)

    null_counts_after = (
        df.select([
            F.count(F.when(F.col(c).isNull(), 1)).alias(c)
            for c in cols
        ])
        .collect()[0]
        .asDict()
    )

    with open(os.path.join(out_dir, "block_feature_summary.txt"), "w", encoding="utf-8") as f:
        f.write("===== BLOCK-DERIVED FEATURES SUMMARY =====\n\n")
        f.write(f"Total rows: {total_rows}\n\n")

        f.write("Null counts BEFORE imputation\n")
        f.write("-----------------------------\n")
        for c in cols:
            count = null_counts_before[c]
            pct = count / total_rows * 100 if total_rows else 0.0
            f.write(f"{c}: {count} ({pct:.8f}%)\n")

        f.write("\nNull counts AFTER imputation\n")
        f.write("----------------------------\n")
        for c in cols:
            count = null_counts_after[c]
            pct = count / total_rows * 100 if total_rows else 0.0
            f.write(f"{c}: {count} ({pct:.8f}%)\n")

        f.write("\nDistinct counts AFTER imputation\n")
        f.write("-------------------------------\n")
        for c in cols:
            distinct_count = df.select(c).distinct().count()
            f.write(f"{c}: {distinct_count}\n")

        f.write("\nImputation result\n")
        f.write("-----------------\n")
        f.write(f"Rows with missing block_number before: {missing_before_count}\n")
        f.write(f"Rows successfully imputed: {imputed_count}\n")
        f.write(f"Rows still missing after imputation: {still_missing_count}\n")

    for c in cols:
        _write_top_values(
            df=df,
            col_name=c,
            out_file=os.path.join(out_dir, f"top_{c}.txt"),
            top_n=30
        )

    with open(os.path.join(out_dir, "missing_block_number_summary.txt"), "w", encoding="utf-8") as f:
        f.write("===== MISSING block_number =====\n\n")
        f.write(f"Rows with missing block_number before: {missing_before_count}\n")
        f.write(f"Rows imputed: {imputed_count}\n")
        f.write(f"Rows with missing block_number after: {still_missing_count}\n")

    print(f"Block stats saved to: {out_dir}")
    print(f"Updated clean dataset saved to: {updated_path}")


if __name__ == "__main__":
    run_block_stats()