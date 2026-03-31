import os
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StringType,
    BooleanType,
    IntegerType,
    LongType,
    ShortType,
    ByteType
)


def _is_low_cardinality_numeric(field):
    return isinstance(field.dataType, (IntegerType, LongType, ShortType, ByteType))


def _classify_imbalance(dominance_ratio):
    if dominance_ratio >= 0.9:
        return "almost_constant"
    if dominance_ratio >= 0.6:
        return "strong_imbalance"
    if dominance_ratio >= 0.3:
        return "moderate_imbalance"
    return "relatively_balanced"


def _prepare_spatial_grid(df, lat_col="Latitude", lon_col="Longitude", grid_size=0.01):
    factor = int(round(1 / grid_size))

    return df.withColumn(
        "geo_grid",
        F.when(
            F.col(lat_col).isNotNull() & F.col(lon_col).isNotNull(),
            F.concat_ws(
                "_",
                F.floor(F.col(lat_col) * F.lit(factor)).cast("int"),
                F.floor(F.col(lon_col) * F.lit(factor)).cast("int")
            )
        )
    )


def _prepare_time_features(df, date_col="Date"):
    return (
        df.withColumn(
            "crime_year",
            F.when(F.col(date_col).isNotNull(), F.year(F.col(date_col)))
        )
        .withColumn(
            "crime_month",
            F.when(F.col(date_col).isNotNull(), F.month(F.col(date_col)))
        )
        .withColumn(
            "crime_hour",
            F.when(F.col(date_col).isNotNull(), F.hour(F.col(date_col)))
        )
        .withColumn(
            "crime_dayofweek",
            F.when(F.col(date_col).isNotNull(), F.dayofweek(F.col(date_col)))
        )
    )


def _should_analyze_column(
    df,
    field,
    max_distinct=100,
    exclude_cols=None,
    force_include_numeric=None,
    auxiliary_used_cols=None
):
    if exclude_cols is None:
        exclude_cols = []

    if force_include_numeric is None:
        force_include_numeric = []

    if auxiliary_used_cols is None:
        auxiliary_used_cols = []

    col_name = field.name

    if col_name in auxiliary_used_cols:
        return False, None, "used_for_derived_feature"

    if col_name in exclude_cols:
        return False, None, "excluded"

    if isinstance(field.dataType, (StringType, BooleanType)):
        distinct_count = df.select(col_name).distinct().count()
        if distinct_count <= 1:
            return False, distinct_count, "skipped_constant"
        return True, distinct_count, "categorical"

    if col_name in force_include_numeric and _is_low_cardinality_numeric(field):
        distinct_count = df.select(col_name).distinct().count()
        if distinct_count <= 1:
            return False, distinct_count, "skipped_constant"
        return True, distinct_count, "forced_numeric"

    if _is_low_cardinality_numeric(field):
        distinct_count = df.select(col_name).distinct().count()
        if distinct_count <= 1:
            return False, distinct_count, "skipped_constant"
        if distinct_count <= max_distinct:
            return True, distinct_count, "low_cardinality_numeric"
        return False, distinct_count, "skipped_high_cardinality_numeric"

    return False, None, "skipped_unsupported_type"


def _analyze_single_column(df, col_name, distinct_count, total_rows, top_n=10):
    freq_df = (
        df.groupBy(col_name)
        .count()
        .orderBy(F.desc("count"))
    )

    top_rows = freq_df.limit(top_n).collect()

    if not top_rows:
        return None

    top_value = top_rows[0][0]
    top_count = top_rows[0][1]
    dominance_ratio = top_count / total_rows if total_rows > 0 else 0.0
    imbalance_label = _classify_imbalance(dominance_ratio)

    distribution_rows = []
    cumulative_top_n = 0

    for row in top_rows:
        value = row[0]
        count = row[1]
        ratio = count / total_rows if total_rows > 0 else 0.0
        cumulative_top_n += count
        distribution_rows.append({
            "feature": col_name,
            "value": str(value),
            "count": int(count),
            "ratio": round(float(ratio), 6)
        })

    top_n_ratio = cumulative_top_n / total_rows if total_rows > 0 else 0.0

    return {
        "feature": col_name,
        "distinct_count": int(distinct_count),
        "top_value": str(top_value),
        "top_count": int(top_count),
        "dominance_ratio": round(float(dominance_ratio), 6),
        "top_n_ratio": round(float(top_n_ratio), 6),
        "imbalance_label": imbalance_label,
        "distribution_rows": distribution_rows
    }


def _write_section(f, title, section_df, top_n):
    f.write(title + "\n")
    f.write("-" * len(title) + "\n")

    if section_df.empty:
        f.write("None\n\n")
        return

    for _, row in section_df.iterrows():
        f.write(
            f"{row['feature']}: "
            f"type={row['analysis_type']}, "
            f"distinct={row['distinct_count']}, "
            f"top_value={row['top_value']}, "
            f"top_count={row['top_count']}, "
            f"dominance_ratio={float(row['dominance_ratio']):.6f}, "
            f"top_{top_n}_ratio={float(row['top_n_ratio']):.6f}, "
            f"label={row['imbalance_label']}\n"
        )
    f.write("\n")


def run_imbalance_analysis(
    df,
    out_dir="imbalance_results",
    max_distinct_numeric=100,
    top_n=10,
    exclude_cols=None,
    force_include_numeric=None,
    add_spatial_grid=True,
    grid_size=0.01,
    add_time_features=True,
    date_col="Date"
):
    os.makedirs(out_dir, exist_ok=True)

    if exclude_cols is None:
        exclude_cols = [
            "ID",
            "Case Number",
            "Location",
            "X Coordinate",
            "Y Coordinate",
            "Updated On"
        ]

    if force_include_numeric is None:
        force_include_numeric = [
            "Beat",
            "District",
            "Ward",
            "Community Area",
            "crime_year",
            "crime_month",
            "crime_hour",
            "crime_dayofweek",
            "block_number"
        ]

    auxiliary_used_cols = []
    work_df = df

    if add_spatial_grid:
        work_df = _prepare_spatial_grid(work_df, grid_size=grid_size)
        auxiliary_used_cols.extend(["Latitude", "Longitude"])

    if add_time_features:
        work_df = _prepare_time_features(work_df, date_col=date_col)
        auxiliary_used_cols.append(date_col)

    total_rows = work_df.count()
    summary_rows = []
    distribution_all = []

    for field in work_df.schema.fields:
        should_analyze, distinct_count, reason = _should_analyze_column(
            work_df,
            field,
            max_distinct=max_distinct_numeric,
            exclude_cols=exclude_cols,
            force_include_numeric=force_include_numeric,
            auxiliary_used_cols=auxiliary_used_cols
        )

        if not should_analyze:
            summary_rows.append({
                "feature": field.name,
                "analysis_type": reason,
                "distinct_count": int(distinct_count) if distinct_count is not None else "",
                "top_value": "",
                "top_count": "",
                "dominance_ratio": "",
                "top_n_ratio": "",
                "imbalance_label": "skipped"
            })
            continue

        result = _analyze_single_column(
            work_df,
            col_name=field.name,
            distinct_count=distinct_count,
            total_rows=total_rows,
            top_n=top_n
        )

        if result is None:
            continue

        summary_rows.append({
            "feature": result["feature"],
            "analysis_type": reason,
            "distinct_count": result["distinct_count"],
            "top_value": result["top_value"],
            "top_count": result["top_count"],
            "dominance_ratio": result["dominance_ratio"],
            "top_n_ratio": result["top_n_ratio"],
            "imbalance_label": result["imbalance_label"]
        })

        distribution_all.extend(result["distribution_rows"])

    summary_df = pd.DataFrame(summary_rows)
    distribution_df = pd.DataFrame(distribution_all)

    label_order = {
        "almost_constant": 0,
        "strong_imbalance": 1,
        "moderate_imbalance": 2,
        "relatively_balanced": 3,
        "skipped": 4
    }

    if not summary_df.empty:
        summary_df["label_rank"] = summary_df["imbalance_label"].map(label_order)
        summary_df["dominance_sort"] = pd.to_numeric(
            summary_df["dominance_ratio"],
            errors="coerce"
        ).fillna(-1)

        summary_df = summary_df.sort_values(
            by=["label_rank", "dominance_sort", "feature"],
            ascending=[True, False, True]
        ).drop(columns=["label_rank", "dominance_sort"])

    summary_csv_path = os.path.join(out_dir, "imbalance_summary.csv")
    summary_txt_path = os.path.join(out_dir, "imbalance_summary.txt")
    distribution_csv_path = os.path.join(out_dir, "imbalance_top_values.csv")

    summary_df.to_csv(summary_csv_path, index=False)

    if not distribution_df.empty:
        distribution_df.to_csv(distribution_csv_path, index=False)
    else:
        pd.DataFrame(
            columns=["feature", "value", "count", "ratio"]
        ).to_csv(distribution_csv_path, index=False)

    analyzed = summary_df[summary_df["imbalance_label"] != "skipped"].copy()
    skipped = summary_df[
        (summary_df["imbalance_label"] == "skipped") &
        (~summary_df["analysis_type"].isin(["used_for_derived_feature"]))
    ].copy()
    auxiliary_df = summary_df[
        summary_df["analysis_type"] == "used_for_derived_feature"
    ].copy()

    almost_constant_df = analyzed[analyzed["imbalance_label"] == "almost_constant"]
    strong_df = analyzed[analyzed["imbalance_label"] == "strong_imbalance"]
    moderate_df = analyzed[analyzed["imbalance_label"] == "moderate_imbalance"]
    balanced_df = analyzed[analyzed["imbalance_label"] == "relatively_balanced"]

    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("===== DATASET IMBALANCE ANALYSIS =====\n\n")
        f.write(f"Total rows: {total_rows}\n")
        f.write(f"Total features in report: {len(summary_df)}\n")
        f.write(f"Analyzed features: {len(analyzed)}\n")
        f.write(f"Skipped features: {len(skipped)}\n")
        f.write(f"Auxiliary source fields: {len(auxiliary_df)}\n")
        if add_spatial_grid:
            f.write(f"Derived spatial feature: geo_grid (grid_size={grid_size})\n")
        else:
            f.write("Derived spatial feature: not used\n")
        if add_time_features:
            f.write("Derived time features: crime_year, crime_month, crime_hour, crime_dayofweek\n")
        else:
            f.write("Derived time features: not used\n")
        f.write("Additional block-derived features considered: block_number, direction, street_type, street_name\n\n")

        f.write("Overall conclusion\n")
        f.write("------------------\n")
        f.write(
            "The analysis covers categorical features, selected administrative numeric features, "
            "derived spatial and temporal features, and additional components extracted from Block.\n"
        )

        if len(almost_constant_df) == 0 and len(strong_df) == 0:
            f.write("No critically imbalanced features were detected among the analyzed features.\n")
        else:
            f.write(
                f"Detected {len(almost_constant_df)} almost constant and "
                f"{len(strong_df)} strongly imbalanced features.\n"
            )

        if add_spatial_grid:
            f.write(
                "Latitude and Longitude were not analyzed as raw fields; they were used only "
                "to construct geo_grid for spatial concentration analysis.\n"
            )

        if add_time_features:
            f.write(
                f"{date_col} was not analyzed as a raw timestamp; it was discretized into "
                "year, month, hour, and day-of-week features.\n"
            )

        if "street_name" in analyzed["feature"].values:
            f.write(
                "street_name is a high-cardinality categorical feature, therefore its imbalance "
                "metrics should be interpreted cautiously and not treated as a primary summary indicator.\n"
            )

        if "geo_grid" in analyzed["feature"].values:
            f.write(
                "The geo_grid result reflects dominance of individual spatial cells, not "
                "uniformity of the full geographic distribution.\n"
            )

        f.write("\n")

        _write_section(f, "Almost constant features", almost_constant_df, top_n)
        _write_section(f, "Strongly imbalanced features", strong_df, top_n)
        _write_section(f, "Moderately imbalanced features", moderate_df, top_n)
        _write_section(f, "Relatively balanced features", balanced_df, top_n)

        f.write("Skipped features\n")
        f.write("----------------\n")
        if skipped.empty:
            f.write("None\n\n")
        else:
            for _, row in skipped.iterrows():
                f.write(f"{row['feature']}: {row['analysis_type']}\n")
            f.write("\n")

        f.write("Auxiliary source fields\n")
        f.write("-----------------------\n")
        if auxiliary_df.empty:
            f.write("None\n")
        else:
            for _, row in auxiliary_df.iterrows():
                f.write(f"{row['feature']}: used_for_derived_feature\n")

    return {
        "summary_path_csv": summary_csv_path,
        "summary_path_txt": summary_txt_path,
        "distribution_path_csv": distribution_csv_path,
        "summary_df": summary_df,
        "distribution_df": distribution_df
    }