import os
from pyspark.sql import SparkSession


def build_spark(app_name="ChicagoCrimes"):
    os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName(app_name)
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark