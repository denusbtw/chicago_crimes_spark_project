from spark_utils import build_spark
from queries import get_lookup_tables
from business_plots import generate_question_plots
import queries as q


def run_business_plots():
    spark = build_spark("ChicagoCrimes-BusinessPlots")

    clean_path = "data/processed/chicago_crimes_clean"
    df = spark.read.parquet(clean_path)

    severity_df, districts_df = get_lookup_tables(spark)

    registry = [
        {
            "code": "Q1",
            "title": "Крадіжки на суму понад $500",
            "base_df": lambda: df.filter(
                (df["Primary Type"] == "THEFT") &
                (df["Description"] == "OVER $500")
            ),
            "result_df": lambda: q.q1_high_value_thefts(df)
        },
        {
            "code": "Q2",
            "title": "Побутове насильство без проведеного арешту",
            "base_df": lambda: df.filter(
                (df["Domestic"] == True) &
                (df["Arrest"] == False)
            ),
            "result_df": lambda: q.q2_domestic_violence_no_arrest(df)
        },
        {
            "code": "Q3",
            "title": "Злочини, що відбувалися в ресторанах",
            "base_df": lambda: df.filter(df["Location Description"] == "RESTAURANT"),
            "result_df": lambda: q.q3_crimes_in_restaurants(df)
        },
        {
            "code": "Q4",
            "title": "Злочини за 2026 рік",
            "base_df": lambda: df.filter(df["Date"].isNotNull()).filter(df["Date"].substr(1, 4) == "2026"),
            "result_df": lambda: q.q4_crimes_by_year_2026(df)
        },
        {
            "code": "Q5",
            "title": "Вуличні злочини в нічний час",
            "base_df": lambda: df.filter(
                (df["Location Description"] == "STREET")
            ),
            "result_df": lambda: q.q5_crimes_on_streets_at_night(df)
        },
        {
            "code": "Q6",
            "title": "Випадки, пов'язані з наркотиками",
            "base_df": lambda: df.filter(df["Primary Type"] == "NARCOTICS"),
            "result_df": lambda: q.q6_narcotics_cases(df)
        },
        {
            "code": "Q7",
            "title": "Злочини з коректними геокординатами",
            "base_df": lambda: df.filter(
                df["Latitude"].isNotNull() & df["Longitude"].isNotNull()
            ),
            "result_df": lambda: q.q7_crimes_with_valid_coordinates(df)
        },
        {
            "code": "Q8",
            "title": "Злочини у 42-му варді",
            "base_df": lambda: df.filter(df["Ward"] == 42),
            "result_df": lambda: df.filter(df["Ward"] == 42)
        },
        {
            "code": "Q9",
            "title": "Розподіл злочинів за Primary Type",
            "base_df": lambda: df,
            "result_df": lambda: q.q9_count_by_crime_type(df)
        },
        {
            "code": "Q10",
            "title": "Відсоток арештів по районах",
            "base_df": lambda: df,
            "result_df": lambda: q.q10_arrest_rate_by_district(df)
        },
        {
            "code": "Q11",
            "title": "Динаміка злочинності за місяцями",
            "base_df": lambda: df,
            "result_df": lambda: q.q11_crimes_per_month(df)
        },
        {
            "code": "Q12",
            "title": "Топ-5 типів локацій",
            "base_df": lambda: df,
            "result_df": lambda: q.q12_top_location_types(df)
        },
        {
            "code": "Q13",
            "title": "Динаміка побутових злочинів за роками",
            "base_df": lambda: df.filter(df["Domestic"] == True),
            "result_df": lambda: q.q13_domestic_crimes_per_year(df)
        },
        {
            "code": "Q14",
            "title": "Розподіл злочинів за FBI Code",
            "base_df": lambda: df,
            "result_df": lambda: q.q14_fbi_code_distribution(df)
        },
        {
            "code": "Q15",
            "title": "Пікові години злочинності",
            "base_df": lambda: df,
            "result_df": lambda: q.q15_hourly_crime_frequency(df)
        },
        {
            "code": "Q16",
            "title": "Кількість справ для District/Ward",
            "base_df": lambda: df,
            "result_df": lambda: q.q16_district_ward_combinations(df)
        },
        {
            "code": "Q17",
            "title": "Злочини з доданим рівнем критичності",
            "base_df": lambda: q.q17_crimes_with_severity(df, severity_df),
            "result_df": lambda: q.q17_crimes_with_severity(df, severity_df)
        },
        {
            "code": "Q18",
            "title": "Звіт з назвами районів",
            "base_df": lambda: q.q18_named_districts_report(df, districts_df),
            "result_df": lambda: q.q18_named_districts_report(df, districts_df)
        },
        {
            "code": "Q19",
            "title": "Критичні злочини в центральному районі",
            "base_df": lambda: q.q19_critical_crimes_in_central(df, severity_df, districts_df),
            "result_df": lambda: q.q19_critical_crimes_in_central(df, severity_df, districts_df)
        },
        {
            "code": "Q20",
            "title": "Типи злочинів, відсутні у довіднику",
            "base_df": lambda: q.q20_unmatched_crime_types(df, severity_df),
            "result_df": lambda: q.q20_unmatched_crime_types(df, severity_df)
        },
        {
            "code": "Q21",
            "title": "Порядковий номер злочину в межах району",
            "base_df": lambda: df,
            "result_df": lambda: q.q21_rank_crimes_by_date_in_district(df)
        },
        {
            "code": "Q22",
            "title": "Накопичувальний підсумок злочинів для Ward",
            "base_df": lambda: df,
            "result_df": lambda: q.q22_cumulative_crime_count_by_ward(df)
        },
        {
            "code": "Q23",
            "title": "Часовий інтервал між злочинами в районі",
            "base_df": lambda: df,
            "result_df": lambda: q.q23_time_diff_between_crimes(df)
        },
        {
            "code": "Q24",
            "title": "Найпопулярніший тип злочину в кожному District",
            "base_df": lambda: df,
            "result_df": lambda: q.q24_top_crime_type_per_district(df)
        },
    ]

    for item in registry:
        print(f"\n{'=' * 80}")
        print(f"{item['code']}: {item['title']}")
        print(f"{'=' * 80}")

        base_df = item["base_df"]()
        result_df = item["result_df"]()

        generate_question_plots(
            q_code=item["code"],
            title=item["title"],
            base_df=base_df,
            result_df=result_df,
            out_root="business_plots"
        )


if __name__ == "__main__":
    run_business_plots()