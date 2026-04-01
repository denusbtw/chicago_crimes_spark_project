from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def get_lookup_tables(spark):
    severity_df = spark.createDataFrame([
        ("HOMICIDE", "Critical"),
        ("ROBBERY", "High"),
        ("THEFT", "Medium"),
        ("BATTERY", "Medium")
    ], ["Primary Type", "Severity"])

    districts_df = spark.createDataFrame([
        ("001", "Central"),
        ("002", "Wentworth"),
        ("012", "Near West")
    ], ["District", "District Name"])

    return severity_df, districts_df


def q1_high_value_thefts(df: DataFrame):
    """Крадіжки на суму понад $500."""
    res = df.filter((F.col("Primary Type") == "THEFT") & (F.col("Description") == "OVER $500"))
    return res

def q2_domestic_violence_no_arrest(df: DataFrame):
    """Побутове насильство без проведеного арешту."""
    res = df.filter((F.col("Domestic") == True) & (F.col("Arrest") == False))
    return res

def q3_crimes_in_restaurants(df: DataFrame):
    """Злочини, що відбулися в ресторанах."""
    res = df.filter(F.col("Location Description") == "RESTAURANT")
    return res

def q4_crimes_by_year_2026(df: DataFrame):
    """Злочини за конкретний рік."""
    res = df.filter(F.year("Date") == 2026)
    return res

def q5_crimes_on_streets_at_night(df: DataFrame):
    """Вуличні злочини в нічний час (00:00 - 06:00)."""
    res = df.filter((F.col("Location Description") == "STREET") & (F.hour("Date") < 6))
    return res

def q6_narcotics_cases(df: DataFrame):
    """Випадки, пов'язані з наркотиками (Primary Type)."""
    res = df.filter(F.col("Primary Type") == "NARCOTICS")
    return res

def q7_crimes_with_valid_coordinates(df: DataFrame):
    """Злочини з наявними геокоординатами для мапування."""
    res = df.filter(F.col("Latitude").isNotNull() & F.col("Longitude").isNotNull())
    return res

def q8_specific_ward_analysis(df: DataFrame):
    """Злочини у 42-му варді."""
    res = df.filter(F.col("Ward") == 42)
    return res

def q9_count_by_crime_type(df: DataFrame):
    """Загальна кількість за типом злочину."""
    res = df.groupBy("Primary Type").count().orderBy(F.desc("count"))
    return res

def q10_arrest_rate_by_district(df: DataFrame):
    """Відсоток арештів по районах."""
    res = df.groupBy("District").agg(F.avg(F.col("Arrest").cast("double")).alias("arrest_rate"))
    return res

def q11_crimes_per_month(df: DataFrame):
    """Розподіл злочинів за місяцями."""
    res = df.groupBy(F.month("Date").alias("month")).count().orderBy("month")
    return res

def q12_top_location_types(df: DataFrame):
    """Топ-5 типів локацій за частотою злочинів."""
    res = df.groupBy("Location Description").count().orderBy(F.desc("count")).limit(5)
    return res

def q13_domestic_crimes_per_year(df: DataFrame):
    """Кількість побутових злочинів за роками."""
    res = df.filter(F.col("Domestic") == True).groupBy(F.year("Date").alias("year")).count()
    return res

def q14_fbi_code_distribution(df: DataFrame):
    """Статистика за кодами FBI."""
    res = df.groupBy("FBI Code").count().orderBy(F.desc("count"))
    return res

def q15_hourly_crime_frequency(df: DataFrame):
    """Пікові години злочинності."""
    res = df.groupBy(F.hour("Date").alias("hour")).count().orderBy(F.desc("count"))
    return res

def q16_district_ward_combinations(df: DataFrame):
    """Кількість унікальних справ для пар District/Ward."""
    res = df.groupBy("District", "Ward").count()
    return res

def q17_crimes_with_severity(df: DataFrame, severity_df: DataFrame):
    """Злочини з доданим рівнем критичності."""
    res = df.join(severity_df, "Primary Type", "left")
    return res

def q18_named_districts_report(df: DataFrame, districts_df: DataFrame):
    """Звіт з назвами районів замість номерів."""
    res = df.join(districts_df, "District", "inner")
    return res

def q19_critical_crimes_in_central(df: DataFrame, severity_df: DataFrame, districts_df: DataFrame):
    """Критичні злочини в центральному районі."""
    res = df.join(severity_df, "Primary Type").join(districts_df, "District") \
            .filter((F.col("Severity") == "Critical") & (F.col("District Name") == "Central"))
    return res

def q20_unmatched_crime_types(df: DataFrame, severity_df: DataFrame):
    """Типи злочинів, які відсутні в довіднику пріоритетів (Anti-join)."""
    res = df.join(severity_df, "Primary Type", "left_anti")
    return res

def q21_rank_crimes_by_date_in_district(df: DataFrame):
    """Порядковий номер злочину в межах кожного району за часом."""
    window = Window.partitionBy("District").orderBy("Date")
    res = df.withColumn("crime_order", F.row_number().over(window))
    return res

def q22_cumulative_crime_count_by_ward(df: DataFrame):
    """Накопичувальний підсумок злочинів для кожного варду."""
    window = Window.partitionBy("Ward").orderBy("Date").rowsBetween(Window.unboundedPreceding, Window.currentRow)
    res = df.withColumn("running_total", F.count("Primary Type").over(window))
    return res

def q23_time_diff_between_crimes(df: DataFrame):
    """Різниця в часі (в секундах) між поточним та попереднім злочином у районі."""
    window = Window.partitionBy("District").orderBy("Date")
    res = df.withColumn("prev_crime_time", F.lag("Date").over(window)) \
            .withColumn("diff_seconds", F.unix_timestamp("Date") - F.unix_timestamp("prev_crime_time"))
    return res

def q24_top_crime_type_per_district(df: DataFrame):
    """Найпопулярніший тип злочину для кожного району."""
    counts = df.groupBy("District", "Primary Type").count()
    window = Window.partitionBy("District").orderBy(F.desc("count"))
    res = counts.withColumn("rank", F.rank().over(window)).filter(F.col("rank") == 1)
    return res
