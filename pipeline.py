import glob
import os
import shutil

import queries as q


def write_single_csv(df, out_dir: str, filename: str):
    os.makedirs(out_dir, exist_ok=True)
    pdf = df.toPandas()
    pdf.to_csv(os.path.join(out_dir, filename), index=False)


def run_analysis(df, severity_df, districts_df, out_dir: str = "output"):
    queries_list = [
        ("Q1: Крадіжки на суму (>$500)", lambda: q.q1_high_value_thefts(df)),
        ("Q2: Побудове насильство без проведеного арешту", lambda: q.q2_domestic_violence_no_arrest(df)),
        ("Q3: Злочини, що відбувалися в ресторанах", lambda: q.q3_crimes_in_restaurants(df)),
        ("Q4: Злочини за 2026 рік", lambda: q.q4_crimes_by_year_2026(df)),
        ("Q5: Вуличні злочини в нічний час", lambda: q.q5_crimes_on_streets_at_night(df)),
        ("Q6: Випадки, пов'язані з наркотиками", lambda: q.q6_narcotics_cases(df)),
        ("Q7: Злочини з наявними геокординатами для мапування", lambda: q.q7_crimes_with_valid_coordinates(df)),
        ("Q8: Злочини у 42-му варді", lambda: q.q8_specific_ward_analysis(df)),
        ("Q9: Загальна кількість за типом злочину", lambda: q.q9_count_by_crime_type(df)),
        ("Q10: Відсоток арештів по районах", lambda: q.q10_arrest_rate_by_district(df)),
        ("Q11: Розподіл злочинів за місяцями", lambda: q.q11_crimes_per_month(df)),
        ("Q12: Топ-5 типів локації за частотою злочинів", lambda: q.q12_top_location_types(df)),
        ("Q13: Кількість побутових злочинів за роками", lambda: q.q13_domestic_crimes_per_year(df)),
        ("Q14: Статистика за кодами FBI", lambda: q.q14_fbi_code_distribution(df)),
        ("Q15: Пікові години злочинності", lambda: q.q15_hourly_crime_frequency(df)),
        ("Q16: Кількість унікальних справ для пар District/Ward", lambda: q.q16_district_ward_combinations(df)),
        ("Q17: Злочини з доданим рівнем критичності", lambda: q.q17_crimes_with_severity(df, severity_df)),
        ("Q18: Звіт з назвами районів замість номерів", lambda: q.q18_named_districts_report(df, districts_df)),
        ("Q19: Критичні злочини в центральному районі", lambda: q.q19_critical_crimes_in_central(df, severity_df, districts_df)),
        ("Q20: Типи злочинів, які відсутні в довіднику пріоритетів", lambda: q.q20_unmatched_crime_types(df, severity_df)),
        ("Q21: Порядковий номер злочину в межах району за часом", lambda: q.q21_rank_crimes_by_date_in_district(df)),
        ("Q22: Накопичувальний підсумок злочинів для кожного варду", lambda: q.q22_cumulative_crime_count_by_ward(df)),
        ("Q23: Різниця в часі (в секундах) між поточним та попереднім злочином в районі", lambda: q.q23_time_diff_between_crimes(df)),
        ("Q24: Найпопулярніший тип злочину для кожного району", lambda: q.q24_top_crime_type_per_district(df)),
    ]

    os.makedirs(out_dir, exist_ok=True)

    for i, (title, query_func) in enumerate(queries_list, start=1):
        print(f"\n{'=' * 80}")
        print(f"Бізнес-питання: {title}")
        print(f"{'=' * 80}")

        result_df = query_func()
        result_df.explain()

        write_single_csv(result_df, out_dir, f"Q{i}.csv")