from pyspark.sql import functions as F
from schemas import crime_schema

def load_crime_data(spark, path: str):
    df = (
        spark.read
        .option("header", True)
        .option("sep", ";")
        .option("quote", '"')
        .option("timestampFormat", "MM/dd/yyyy hh:mm:ss a")
        .schema(crime_schema)
        .csv(path)
    )

    return df


def validate_dataframe(df):
    if df.rdd.isEmpty():
        raise ValueError("DataFrame is empty")

    dtypes = dict(df.dtypes)

    if "Date" not in dtypes:
        raise ValueError("Date column missing")

    if dtypes["Date"] != "timestamp":
        raise TypeError("Date column is not timestamp")

    required_cols = ["Date", "Latitude", "Longitude", "District"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    invalid_rows = df.filter(
        F.col("Date").isNull() |
        F.col("Latitude").isNull() |
        F.col("Longitude").isNull() |
        F.col("District").isNull() |
        (F.col("District") < 1) |
        (F.col("District") > 31)
    ).limit(1).count()

    if invalid_rows > 0:
        raise ValueError("DataFrame contains invalid critical values")

    print("DataFrame validation passed.")
