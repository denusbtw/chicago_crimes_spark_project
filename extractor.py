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
    if df.count() == 0:
        raise ValueError("DataFrame is empty")

    if "ID" not in df.columns:
        raise ValueError("ID column missing")

    if dict(df.dtypes)["Date"] != "timestamp":
        raise TypeError("Date column is not timestamp")

    print("DataFrame validation passed.")
