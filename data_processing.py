from pyspark.sql import functions as F
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


def handle_missing(df):
    df = df.fillna({
        "Location Description": "Unknown"
    })

    df = df.filter(
        F.col("Date").isNotNull() &
        F.col("Latitude").isNotNull() &
        F.col("Longitude").isNotNull()
    )

    df = df.withColumn(
        "Ward",
        F.when(F.col("Ward").isNull() | (F.col("Ward") == 0), F.lit(-1)).otherwise(F.col("Ward"))
    )

    df = df.withColumn(
        "Community Area",
        F.when(F.col("Community Area").isNull() | (F.col("Community Area") == 0), F.lit(-1)).otherwise(F.col("Community Area"))
    )

    mapping = (
        df.filter(F.col("District").isNotNull() & (F.col("District") != 0))
          .groupBy("Beat")
          .agg(F.first("District", ignorenulls=True).alias("District_from_beat"))
    )

    df = df.join(mapping, on="Beat", how="left")

    df = df.withColumn(
        "District",
        F.when(F.col("District").isNull() | (F.col("District") == 0), F.col("District_from_beat")).otherwise(F.col("District"))
    ).drop("District_from_beat")

    df = df.filter(F.col("District").isNotNull() & (F.col("District") != 0))

    return df

from pyspark.sql import functions as F
import os


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