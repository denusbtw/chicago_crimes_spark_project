import queries as q
from pyspark.sql import functions as F


def prepare_df(df):
    return df.withColumn(
        "ts", F.to_timestamp("Date", "MM/dd/yyyy hh:mm:ss a")
    ).withColumn(
        "hour", F.hour("ts")
    ).withColumn(
        "year", F.year("ts")
    ).withColumn(
        "month", F.month("ts")
    ).withColumn(
        "day_of_week", F.date_format("ts", "E")
    )


def run_analysis(df, spark, out_dir: str = "output"):
    df = prepare_df(df)

    tasks = [
        (q.Q1_D_Night_Crimes(out_dir), "Q1_D_Night_Crimes", "Нічні злочини (00-06)"),
        (q.Q2_D_Weekend_vs_Weekday(out_dir), "Q2_D_Weekend_vs_Weekday", "Розподіл: Вихідні vs Будні"),
        (q.Q3_D_Narcotics_vs_Assault_By_District(out_dir), "Q3_D_Narcotics_vs_Assault_By_District", "Співвідношення Narcotics vs Assault по Районах"),
        (q.Q4_D_Domestic_By_District(out_dir), "Q4_D_Domestic_By_District", "Домашнє насильство по районах"),
        (q.Q5_D_Arrest_Ratio_By_Ward(out_dir), "Q5_D_Arrest_Ratio_By_Ward", "Топ-15 Ward за відсотком арештів"),
        (q.Q6_D_Moving_Average_Crimes(out_dir), "Q6_D_Moving_Average_Crimes", "Динаміка злочинності (3-міс. ковзне середнє)"), 
        (q.Q1_J_Top_Theft_Districts(out_dir), "Q1_J_Top_Theft_Districts", "ТОП-10 районів за крадіжками"),
        (q.Q2_J_Domestic_Monthly(out_dir), "Q2_J_Domestic_Monthly", "Домашнє насильство по місяцях"),
        (q.Q3_J_Top5_Crime_Locations(out_dir), "Q3_J_Top5_Crime_Locations", "Локації ТОП-5 найпопулярніших типів злочинів"),
        (q.Q4_J_Top_Streets(out_dir), "Q4_J_Top_Streets", "ТОП-10 вулиць за кількістю злочинів"),
        (q.Q5_J_Top_Crime_Per_District(out_dir), "Q5_J_Top_Crime_Per_District", "Найпопулярніший тип злочину по районах"),
        (q.Q6_J_Most_Dangerous_Blocks_Share(out_dir), "Q6_J_Most_Dangerous_Blocks_Share", "Топ-10 найнебезпечніших кварталів (частка)"),
    ]
    
    for question_obj, folder, title in tasks:
        try:
            print(f"\n{'='*50}\nАналіз плану для: {folder}\n{'='*50}")
            result_df = question_obj.execute(df)
            result_df.explain("formatted")

            question_obj.run(df, title, folder)
        except Exception as e:
            print(f"Помилка в {title}: {e}")
