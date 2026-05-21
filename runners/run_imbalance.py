from spark_utils import build_spark
from imbalance_analysis import run_imbalance_analysis


def run_imbalance():
    spark = build_spark("ChicagoCrimes-Imbalance")

    clean_path = "data/processed/chicago_crimes_clean"
    df_clean = spark.read.parquet(clean_path)

    run_imbalance_analysis(
        df_clean,
        out_dir="imbalance_results",
        max_distinct_numeric=100,
        top_n=10,
        exclude_cols=[
            "ID",
            "Case Number",
            "Location",
            "X Coordinate",
            "Y Coordinate",
            "Updated On"
        ],
        force_include_numeric=[
            "Beat",
            "District",
            "Ward",
            "Community Area",
            "crime_year",
            "crime_month",
            "crime_hour",
            "crime_dayofweek",
            "block_number"
        ],
        add_spatial_grid=True,
        grid_size=0.01,
        add_time_features=True,
        date_col="Date"
    )


if __name__ == "__main__":
    run_imbalance()