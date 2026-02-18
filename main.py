from pyspark.sql import SparkSession

from extractor import load_crime_data, validate_dataframe


if __name__ == "__main__":
    spark = SparkSession.builder.appName("ChicagoCrimes").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    df = load_crime_data(spark, "data/chicago_crimes.csv")

    validate_dataframe(df)
