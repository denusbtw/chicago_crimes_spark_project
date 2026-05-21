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

        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")

        .config("spark.sql.shuffle.partitions", "50")
        .config("spark.default.parallelism", "4")

        .config("spark.local.dir", "/tmp/spark-temp")

        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")

        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")
    return spark