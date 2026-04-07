from pyspark.sql import functions as F
from pyspark.sql import Window, DataFrame
from enum import Enum


def get_lookup_tables(spark):
    pass


def q1_night_crimes(df):
    return df.filter((F.col("hour") >= 0) & (F.col("hour") < 6)) \
             .groupBy("Primary Type").count().orderBy(F.desc("count"))

def q2_top_theft_districts(df):
    return df.filter(F.col("Primary Type") == "THEFT") \
             .groupBy("District").count().orderBy(F.desc("count")).limit(10)

def q3_domestic_monthly(df):
    return df.filter(F.col("Domestic") == "true") \
        .withColumn("MonthNum", F.month(F.to_timestamp("Date", "MM/dd/yyyy hh:mm:ss a"))) \
        .groupBy("MonthNum") \
        .count() \
        .orderBy("MonthNum")

def q4_arrest_by_type(df):
    return df.filter(F.col("Arrest") == "true") \
             .groupBy("Primary Type").count().orderBy(F.desc("count"))

def q5_weekend_vs_weekday(df):
    return df.withColumn(
                "is_weekend",
                F.when(F.col("day_of_week").isin("Sat", "Sun"), "weekend").otherwise("weekday")
           ).groupBy("is_weekend").count()

def q6_criminal_damage_locations(df):
    return df.filter(F.col("Primary Type") == "CRIMINAL DAMAGE") \
             .groupBy("Location Description").count().orderBy(F.desc("count")).limit(10)

def q7_hourly_all_city(df):
    return df.groupBy("hour").count().orderBy("hour")

def q8_street_crimes(df):
    return df.filter(F.col("Location Description") == "STREET") \
             .groupBy("Primary Type").count().orderBy(F.desc("count"))

def q9_domestic_by_district(df):
    return df.filter(F.col("Domestic") == "true") \
             .groupBy("District").count().orderBy(F.desc("count"))

def q10_day_crimes(df):
    return df.filter((F.col("hour") >= 12) & (F.col("hour") < 18)) \
             .groupBy("Primary Type").count().orderBy(F.desc("count"))

def q11_arrest_share_2025(df):
    window_spec = Window.partitionBy("Primary Type")

    return df.withColumn("Year", F.year(F.to_timestamp("Date", "MM/dd/yyyy hh:mm:ss a"))) \
             .filter(F.col("Year") == 2025) \
             .groupBy("Primary Type", "Arrest") \
             .count() \
             .withColumn("total_count", F.sum("count").over(window_spec)) \
             .withColumn("share", F.col("count") / F.col("total_count")) \
             .filter(F.col("Arrest") == "true") \
             .select("Primary Type", "share") \
             .orderBy(F.desc("share"))

def q12_top_streets(df):
    return df.groupBy("street_name").count().orderBy(F.desc("count")).limit(10)
