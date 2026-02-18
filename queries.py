from pyspark.sql import DataFrame
from pyspark.sql import functions as F


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
