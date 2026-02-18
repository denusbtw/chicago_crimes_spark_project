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
