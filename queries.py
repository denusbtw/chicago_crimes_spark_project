from pyspark.sql import functions as F
from pyspark.sql import Window
import os
import glob
import shutil
import matplotlib.pyplot as plt


def get_lookup_tables(spark):
    pass


class BaseQuestion:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def execute(self, df):
        raise NotImplementedError

    def draw_plot(self, pdf, title, path):
        raise NotImplementedError

    def run(self, df, title, folder):
        result_df = self.execute(df)

        if result_df is None:
            raise ValueError(f"Метод execute у {self.__class__.__name__} повернув None.")

        target_path = os.path.join(self.output_dir, folder)
        os.makedirs(target_path, exist_ok=True)

        self._save_csv(result_df, target_path)

        pdf = result_df.toPandas()
        if not pdf.empty:
            self.draw_plot(pdf, title, os.path.join(target_path, "chart.png"))

    def _save_csv(self, df, target_path):
        temp_path = os.path.join(target_path, "temp_spark")
        df.coalesce(1).write.mode("overwrite").option("header", True).csv(temp_path)

        part_file = glob.glob(os.path.join(temp_path, "part-*.csv"))[0]
        shutil.move(part_file, os.path.join(target_path, "result.csv"))

        shutil.rmtree(temp_path)


class Q1_D_Night_Crimes(BaseQuestion):
    """filter, group by"""
    def execute(self, df):
        return df.filter((F.col("hour") >= 0) & (F.col("hour") < 6)) \
            .groupBy("Primary Type").count().orderBy(F.desc("count"))

    def draw_plot(self, pdf, title, path):
        top_n = pdf.head(15)

        plt.figure(figsize=(12, 8))

        bars = plt.barh(top_n["Primary Type"], top_n["count"], color='midnightblue', edgecolor='black')

        plt.gca().invert_yaxis()

        plt.grid(axis='x', linestyle='--', alpha=0.6)

        for bar in bars:
            width = bar.get_width()
            plt.text(width + (width * 0.01), bar.get_y() + bar.get_height() / 2,
                     f'{int(width):,}', va='center', fontsize=10)

        plt.title(f"{title} (Top 15)", fontsize=14, fontweight='bold')
        plt.xlabel("Кількість випадків", fontsize=12)
        plt.ylabel("Тип злочину", fontsize=12)

        plt.tight_layout()
        plt.savefig(path)
        plt.close()


class Q2_D_Weekend_vs_Weekday(BaseQuestion): #Денис
    """groupby"""
    def execute(self, df, spark=None):
        return df.withColumn(
            "is_weekend",
            F.when(F.col("day_of_week").isin("Sat", "Sun"), "weekend").otherwise("weekday")
        ).groupBy("is_weekend").count()

    def draw_plot(self, pdf, title, path):
        plt.figure(figsize=(8, 8))

        colors = ['#5dade2', '#f39c12']

        plt.pie(
            pdf["count"],
            labels=pdf["is_weekend"],
            autopct=lambda p: f'{p:.1f}%\n({int(p * sum(pdf["count"]) / 100):,})',
            startangle=140,
            colors=colors,
            explode=(0.05, 0),
            shadow=True
        )

        plt.title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

class Q3_D_Narcotics_vs_Assault_By_District(BaseQuestion): #Денис
    """filter, group by, join"""
    def execute(self, df):
        df_narc = df.filter(F.col("Primary Type") == "NARCOTICS") \
            .groupBy("District").count().withColumnRenamed("count", "narcotics")

        df_assault = df.filter(F.col("Primary Type") == "ASSAULT") \
            .groupBy("District").count().withColumnRenamed("count", "assaults")

        return df_narc.join(df_assault, "District") \
            .withColumn("ratio", F.col("narcotics") / F.col("assaults")) \
            .orderBy(F.desc("narcotics")).limit(15)

    def draw_plot(self, pdf, title, path):
        plt.figure(figsize=(10, 6))
        plt.scatter(pdf["assaults"], pdf["narcotics"], s=pdf["ratio"]*100, alpha=0.6, color='blue')
        for i, txt in enumerate(pdf["District"]):
            plt.annotate(f"D:{txt}", (pdf["assaults"].iloc[i], pdf["narcotics"].iloc[i]))
        plt.title("Співвідношення Narcotics vs Assault по Районах")
        plt.xlabel("Кількість Assault")
        plt.ylabel("Кількість Narcotics")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()


class Q4_D_Domestic_By_District(BaseQuestion): #Денис
    """filter, group by"""
    def execute(self, df, spark=None):
        return df.filter(F.col("Domestic") == "true") \
            .groupBy("District").count() \
            .orderBy(F.desc("count"))

    def draw_plot(self, pdf, title, path):
        plt.figure(figsize=(14, 7))

        bars = plt.bar(pdf["District"].astype(str), pdf["count"],
                       color='rebeccapurple', edgecolor='black', alpha=0.8)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + (height * 0.01),
                     f'{int(height):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.title(title, fontsize=15, fontweight='bold')
        plt.xlabel("Номер району (District)", fontsize=12)
        plt.ylabel("Кількість випадків", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.4)

        plt.xticks(rotation=0)

        plt.tight_layout()
        plt.savefig(path)
        plt.close()


class Q5_D_Arrest_Ratio_By_Ward(BaseQuestion):
    """filter, group by, join, window"""
    def execute(self, df):
        df_total = df.filter(F.col("Ward").isNotNull()).groupBy("Ward").count().withColumnRenamed("count", "total_crimes")
        df_arrests = df.filter((F.col("Ward").isNotNull()) & (F.col("Arrest") == True)) \
            .groupBy("Ward").count().withColumnRenamed("count", "arrests")

        joined = df_total.join(df_arrests, "Ward") \
            .withColumn("arrest_rate", (F.col("arrests") / F.col("total_crimes")) * 100)

        window_spec = Window.orderBy(F.desc("arrest_rate"))
        return joined.withColumn("rank", F.dense_rank().over(window_spec)).limit(15)

    def draw_plot(self, pdf, title, path):
        plt.figure(figsize=(10, 6))
        plt.bar(pdf["Ward"].astype(str), pdf["arrest_rate"], color='coral')
        plt.title("Топ-15 Ward за відсотком арештів")
        plt.xlabel("Ward")
        plt.ylabel("Відсоток арештів (%)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()


class Q6_D_Moving_Average_Crimes(BaseQuestion): #Денис
    """filter, group by, window"""
    def execute(self, df):
        df_month = df.withColumn("YearMonth", F.date_format(F.col("Date"), "yyyy-MM")) \
            .filter(F.col("YearMonth").isNotNull())

        monthly_counts = df_month.groupBy("YearMonth").count()

        window_spec = Window.orderBy("YearMonth").rowsBetween(-2, 0)
        return monthly_counts.withColumn("moving_avg_3m", F.avg("count").over(window_spec)) \
            .orderBy("YearMonth")

    def draw_plot(self, pdf, title, path):
        plt.figure(figsize=(14, 6))

        plt.plot(pdf["YearMonth"], pdf["count"], label="Фактична кількість", color='lightgray', marker='.', alpha=0.6)
        plt.plot(pdf["YearMonth"], pdf["moving_avg_3m"], label="3-місячне ковзне середнє", color='red', linewidth=2.5)

        plt.title("Динаміка злочинності (Ковзне середнє)")
        plt.ylabel("Кількість злочинів")

        step = max(1, len(pdf) // 20)

        plt.xticks(ticks=range(0, len(pdf), step), labels=pdf["YearMonth"].iloc[::step], rotation=45)

        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
