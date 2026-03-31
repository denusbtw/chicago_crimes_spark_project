from pyspark.sql import functions as F
from pyspark.sql import Window
import os


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


def extract_block_features(df):
    df = df.withColumn(
        "block_number_raw",
        F.regexp_extract(F.col("Block"), r"^(\d+)", 1)
    )

    df = df.withColumn(
        "block_number",
        F.when(
            F.col("block_number_raw") != "",
            F.col("block_number_raw").cast("int")
        )
    ).drop("block_number_raw")

    df = df.withColumn(
        "direction",
        F.regexp_extract(F.col("Block"), r"\b(N|S|E|W)\b", 1)
    )

    df = df.withColumn(
        "street_type",
        F.upper(
            F.regexp_extract(
                F.col("Block"),
                r"\b(ST|AVE|AV|RD|BLVD|DR|LN|CT|PL|PKWY|HWY|TER|CIR|WAY)\b\.?$",
                1
            )
        )
    )

    df = df.withColumn(
        "street_name",
        F.trim(
            F.regexp_replace(
                F.regexp_replace(
                    F.regexp_replace(
                        F.col("Block"),
                        r"^\d+XX\s*",
                        ""
                    ),
                    r"\b(N|S|E|W)\b\s*",
                    ""
                ),
                r"\b(ST|AVE|AV|RD|BLVD|DR|LN|CT|PL|PKWY|HWY|TER|CIR|WAY)\b\.?$",
                ""
            )
        )
    )

    df = df.withColumn(
        "direction",
        F.when(F.col("direction") == "", F.lit("UNKNOWN")).otherwise(F.col("direction"))
    )

    df = df.withColumn(
        "street_type",
        F.when(F.col("street_type") == "", F.lit("UNKNOWN")).otherwise(F.col("street_type"))
    )

    df = df.withColumn(
        "street_name",
        F.when(
            F.col("street_name").isNull() | (F.col("street_name") == ""),
            F.lit("UNKNOWN")
        ).otherwise(F.col("street_name"))
    )

    return df


def handle_missing(df):
    location_mode = (
        df.filter(F.col("Location Description").isNotNull())
        .groupBy("Primary Type", "Description", "Location Description")
        .count()
    )

    location_window = Window.partitionBy("Primary Type", "Description").orderBy(F.desc("count"))

    location_mode_ranked = (
        location_mode
        .withColumn("rn", F.row_number().over(location_window))
        .withColumn("group_total", F.sum("count").over(Window.partitionBy("Primary Type", "Description")))
        .withColumn("dominance_ratio", F.col("count") / F.col("group_total"))
        .filter((F.col("rn") == 1) & (F.col("dominance_ratio") >= 0.8))
        .select(
            "Primary Type",
            "Description",
            F.col("Location Description").alias("Location_Description_imputed")
        )
    )

    df = df.join(location_mode_ranked, on=["Primary Type", "Description"], how="left")

    df = df.withColumn(
        "Location Description",
        F.when(
            F.col("Location Description").isNull(),
            F.coalesce(F.col("Location_Description_imputed"), F.lit("Unknown"))
        ).otherwise(F.col("Location Description"))
    ).drop("Location_Description_imputed")

    district_mapping = (
        df.filter(F.col("District").isNotNull() & (F.col("District") != 0))
        .groupBy("Beat")
        .agg(F.first("District", ignorenulls=True).alias("District_from_beat"))
    )

    df = df.join(district_mapping, on="Beat", how="left")

    df = df.withColumn(
        "District",
        F.when(
            F.col("District").isNull() | (F.col("District") == 0),
            F.col("District_from_beat")
        ).otherwise(F.col("District"))
    ).drop("District_from_beat")

    community_means = (
        df.filter(
            F.col("Community Area").isNotNull() &
            (F.col("Community Area") != 0) &
            F.col("Latitude").isNotNull() &
            F.col("Longitude").isNotNull()
        )
        .groupBy("Community Area")
        .agg(
            F.avg("Latitude").alias("lat_ca"),
            F.avg("Longitude").alias("lon_ca")
        )
    )

    ward_means = (
        df.filter(
            F.col("Ward").isNotNull() &
            (F.col("Ward") != 0) &
            F.col("Latitude").isNotNull() &
            F.col("Longitude").isNotNull()
        )
        .groupBy("Ward")
        .agg(
            F.avg("Latitude").alias("lat_ward"),
            F.avg("Longitude").alias("lon_ward")
        )
    )

    district_means = (
        df.filter(
            F.col("District").isNotNull() &
            (F.col("District") != 0) &
            F.col("Latitude").isNotNull() &
            F.col("Longitude").isNotNull()
        )
        .groupBy("District")
        .agg(
            F.avg("Latitude").alias("lat_dist"),
            F.avg("Longitude").alias("lon_dist")
        )
    )

    global_means = (
        df.filter(
            F.col("Latitude").isNotNull() &
            F.col("Longitude").isNotNull()
        )
        .agg(
            F.avg("Latitude").alias("lat_global"),
            F.avg("Longitude").alias("lon_global")
        )
        .collect()[0]
    )

    lat_global = global_means["lat_global"]
    lon_global = global_means["lon_global"]

    df = df.join(community_means, on="Community Area", how="left")
    df = df.join(ward_means, on="Ward", how="left")
    df = df.join(district_means, on="District", how="left")

    df = df.withColumn(
        "Latitude",
        F.when(
            F.col("Latitude").isNull(),
            F.coalesce(
                F.col("lat_ca"),
                F.col("lat_ward"),
                F.col("lat_dist"),
                F.lit(lat_global)
            )
        ).otherwise(F.col("Latitude"))
    )

    df = df.withColumn(
        "Longitude",
        F.when(
            F.col("Longitude").isNull(),
            F.coalesce(
                F.col("lon_ca"),
                F.col("lon_ward"),
                F.col("lon_dist"),
                F.lit(lon_global)
            )
        ).otherwise(F.col("Longitude"))
    )

    df = df.drop(
        "lat_ca", "lon_ca",
        "lat_ward", "lon_ward",
        "lat_dist", "lon_dist"
    )

    df = df.withColumn(
        "Ward",
        F.when(
            F.col("Ward").isNull() | (F.col("Ward") == 0),
            F.lit(-1)
        ).otherwise(F.col("Ward"))
    )

    df = df.withColumn(
        "Community Area",
        F.when(
            F.col("Community Area").isNull() | (F.col("Community Area") == 0),
            F.lit(-1)
        ).otherwise(F.col("Community Area"))
    )

    df = df.withColumn(
        "District",
        F.when(
            F.col("District").isNull() | (F.col("District") == 0),
            F.lit(-1)
        ).otherwise(F.col("District"))
    )

    return df


def remove_structural_anomalies(df, save_report=True):
    total_rows = df.count()

    invalid_coordinates = df.filter(
        (F.col("Latitude").isNotNull()) &
        (
            (F.col("Latitude") < 41.5) |
            (F.col("Latitude") > 42.1) |
            (F.col("Longitude") < -88.0) |
            (F.col("Longitude") > -87.4)
        )
    )

    invalid_district = df.filter(
        (F.col("District").isNotNull()) &
        ((F.col("District") < 1) | (F.col("District") > 31))
    )

    invalid_ward = df.filter(
        (F.col("Ward").isNotNull()) &
        ((F.col("Ward") < 1) | (F.col("Ward") > 50))
    )

    invalid_community = df.filter(
        (F.col("Community Area").isNotNull()) &
        ((F.col("Community Area") < 1) | (F.col("Community Area") > 77))
    )

    coord_count = invalid_coordinates.count()
    district_count = invalid_district.count()
    ward_count = invalid_ward.count()
    community_count = invalid_community.count()

    total_anomalies = coord_count + district_count + ward_count + community_count

    report = []
    report.append("===== STRUCTURAL ANOMALY REPORT =====")
    report.append(f"Total rows: {total_rows}")
    report.append(f"Invalid coordinates: {coord_count}")
    report.append(f"Invalid District values: {district_count}")
    report.append(f"Invalid Ward values: {ward_count}")
    report.append(f"Invalid Community Area values: {community_count}")
    report.append(f"Total detected anomalies (raw sum): {total_anomalies}")

    if total_anomalies == 0:
        report.append("VERDICT: No structural anomalies detected.")
    else:
        percent = (total_anomalies / total_rows) * 100
        report.append(f"VERDICT: Structural anomalies detected ({percent:.4f}% of data).")

    if save_report:
        os.makedirs("stats", exist_ok=True)
        with open("stats/structural_anomalies_report.txt", "w", encoding="utf-8") as f:
            for line in report:
                f.write(line + "\n")

    df_clean = df.filter(
        (F.col("Latitude").between(41.5, 42.1)) &
        (F.col("Longitude").between(-88.0, -87.4)) &
        (F.col("District").between(1, 31)) &
        (F.col("Ward").between(1, 50)) &
        (F.col("Community Area").between(1, 77))
    )

    return df_clean