from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

spark = SparkSession.builder \
    .appName("Test DataFrame") \
    .master("local[*]") \
    .getOrCreate()

schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("name", StringType(), True),
    StructField("age", IntegerType(), True)
])

data = [
    (1, "Vadym", 19),
    (2, "Maryna", 19),
    (3, "Yan", 19),
    (4, "Denys", 19)
]

df = spark.createDataFrame(data, schema=schema)

df.show()
df.printSchema()

spark.stop()