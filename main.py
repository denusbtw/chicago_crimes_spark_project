from pyspark.sql import SparkSession
from schemas import crime_schema


if __name__ == "__main__":
    spark = SparkSession.builder.appName("ChicagoCrimes").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    path = "data/chicago_crimes.csv"
    df = (
        spark.read
        .option("header", True)
        .option("sep", ";")
        .option("quote", '"')
        .option("timestampFormat", "MM/dd/yyyy hh:mm:ss a")
        .schema(crime_schema)
        .csv(path)
    )

    if df.count() == 0:
        raise ValueError("DataFrame is empty")

    if "ID" not in df.columns:
        raise ValueError("ID column missing")

    if dict(df.dtypes)["Date"] != "timestamp":
        raise TypeError("Date column is not timestamp")

    print("DataFrame validation passed.")
