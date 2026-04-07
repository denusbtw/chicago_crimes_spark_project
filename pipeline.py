import os
import queries as q
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import functions as F
import glob
import shutil


def save_output(df, base_dir, q_folder, title, chart_type="bar"):
    target_path = os.path.join(base_dir, q_folder)
    os.makedirs(target_path, exist_ok=True)

    # Збереження CSV (залишаємо як було)
    temp_path = os.path.join(target_path, "temp_spark")
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(temp_path)
    part_file = glob.glob(os.path.join(temp_path, "part-*.csv"))[0]
    shutil.move(part_file, os.path.join(target_path, "result.csv"))
    shutil.rmtree(temp_path)

    # Візуалізація
    pdf = df.toPandas()
    if not pdf.empty:
        plt.figure(figsize=(12, 7))
        x_label = pdf.columns[0]
        y_label = pdf.columns[1]

        if chart_type == "bar":
            plt.bar(pdf[x_label].astype(str), pdf[y_label], color='skyblue', edgecolor='navy')

        elif chart_type == "barh":  # Горизонтальний (ідеально для довгих назв вулиць)
            plt.barh(pdf[x_label].astype(str), pdf[y_label], color='salmon')
            plt.gca().invert_yaxis()  # Щоб топ був зверху

        elif chart_type == "line":  # Для часових рядів
            plt.plot(pdf[x_label], pdf[y_label], marker='o', linestyle='-', color='green', linewidth=2)
            plt.grid(True, linestyle='--', alpha=0.6)

        elif chart_type == "pie":  # Для часток (Arrest, Domestic, Severity)
            plt.pie(pdf[y_label], labels=pdf[x_label].astype(str), autopct='%1.1f%%', startangle=140,
                    colors=plt.cm.Paired.colors)

        elif chart_type == "area":  # Для накопичувальних підсумків
            plt.fill_between(pdf[x_label], pdf[y_label], color="lightgreen", alpha=0.4)
            plt.plot(pdf[x_label], pdf[y_label], color="green", alpha=0.6)

        elif chart_type == "scatter":  # Для кореляцій (наприклад, Населення vs Злочини)
            plt.scatter(pdf[x_label], pdf[y_label], alpha=0.5, color='purple')
            for i, txt in enumerate(pdf[x_label]):  # Додамо підписи точок
                plt.annotate(txt, (pdf.iloc[i, 0], pdf.iloc[i, 1]), size=8)

        elif chart_type == "polar":
            # Перетворюємо години в радіани (2*pi радіан = 24 години)
            angles = [n / 24.0 * 2 * np.pi for n in pdf[x_label].astype(float)]
            # Щоб графік замкнувся, додаємо першу точку в кінець
            angles += angles[:1]
            values = pdf[y_label].tolist()
            values += values[:1]

            ax = plt.subplot(111, projection='polar')
            # Малюємо лінію або бари
            ax.plot(angles, values, color='green', linewidth=2)
            ax.fill(angles, values, color='green', alpha=0.25)

            # Налаштовуємо «циферблат»
            ax.set_theta_offset(np.pi / 2)  # Початок зверху (12:00)
            ax.set_theta_direction(-1)  # За годинниковою стрілкою

            # Назви годин на колі
            ax.set_xticks(np.linspace(0, 2 * np.pi, 24, endpoint=False))
            ax.set_xticklabels([f"{h}h" for h in range(24)])

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if chart_type != "pie":
            plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.savefig(os.path.join(target_path, "chart.png"))
        plt.close()


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
        (q.q1_night_crimes, [df], "Q01_Night_Crimes", "Нічні злочини за типами", "barh"),
        (q.q2_top_theft_districts, [df], "Q02_Theft_Districts", "ТОП районів за крадіжками", "bar"),
        (q.q3_domestic_monthly, [df], "Q03_Domestic_Monthly", "Домашнє насильство по місяцях", "line"),
        (q.q4_arrest_by_type, [df], "Q04_Arrests_By_Type", "Арешти за типами злочинів", "barh"),
        (q.q5_weekend_vs_weekday, [df], "Q05_Weekend_vs_Weekday", "Вихідні vs будні", "pie"),
        (q.q6_criminal_damage_locations, [df], "Q06_Criminal_Damage", "ТОП 10 CRIMINAL DAMAGE по локаціях", "barh"),
        (q.q7_hourly_all_city, [df], "Q07_Hourly_All_City", "Активність по годинах", "polar"),
        (q.q8_street_crimes, [df], "Q08_Street_Crimes", "Злочини на вулиці", "barh"),
        (q.q9_domestic_by_district, [df], "Q09_Domestic_District", "Домашнє насильство по районах", "bar"),
        (q.q10_day_crimes, [df], "Q10_Day_Crimes", "Денні злочини", "barh"),
        (q.q11_arrest_share_2025, [df], "Q11_Arrest_Share", "Арешти 2025 по типах", "barh"),
        (q.q12_top_streets, [df], "Q12_Top_Streets", "ТОП вулиць за злочинами", "barh"),
    ]

    for func, args, folder, title, c_type in tasks:
        print(f"--- Processing: {title} ---")
        result_df = func(*args)
        save_output(result_df, out_dir, folder, title, chart_type=c_type)
