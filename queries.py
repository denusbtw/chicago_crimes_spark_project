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


class Q2_D_Weekend_vs_Weekday(BaseQuestion):
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

class Q3_D_Narcotics_vs_Assault_By_District(BaseQuestion): 
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


class Q4_D_Domestic_By_District(BaseQuestion):
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


class Q6_D_Moving_Average_Crimes(BaseQuestion):
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


class Q1_J_Top_Theft_Districts(BaseQuestion):
    """filter, group by"""
    def execute(self, df, spark=None):
        return df.filter(F.col("Primary Type") == "THEFT") \
            .groupBy("District").count() \
            .orderBy(F.desc("count")) \
            .limit(10)

    def draw_plot(self, pdf, title, path):
        plt.figure(figsize=(10, 6))

        bars = plt.bar(pdf["District"].astype(str), pdf["count"],
                       color='skyblue', edgecolor='navy')

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 500,
                     f'{int(height):,}', ha='center', va='bottom', fontsize=10)

        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Номер району (District)", fontsize=12)
        plt.ylabel("Кількість крадіжок", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(path)
        plt.close()


class Q2_J_Domestic_Monthly(BaseQuestion):
    """filter, group by"""
    def execute(self, df, spark=None):
        return df.filter(F.col("Domestic") == "true") \
            .withColumn("MonthNum", F.month(F.to_timestamp("Date", "MM/dd/yyyy hh:mm:ss a"))) \
            .groupBy("MonthNum") \
            .count() \
            .orderBy("MonthNum")

    def draw_plot(self, pdf, title, path):
        plt.figure(figsize=(10, 6))

        pdf['MonthName'] = pdf['MonthNum'].apply(lambda x: calendar.month_name[int(x)])

        plt.plot(pdf['MonthName'], pdf['count'], marker='o', linestyle='-',
                 color='crimson', linewidth=2, markersize=8)

        plt.fill_between(pdf['MonthName'], pdf['count'], color='crimson', alpha=0.1)

        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Місяць", fontsize=12)
        plt.ylabel("Кількість випадків", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

class Q3_J_Top5_Crime_Locations(BaseQuestion):
    """filter, group by, join"""
    def execute(self, df):
        top_types = df.groupBy("Primary Type").count().orderBy(F.desc("count")).limit(5)

        joined = df.join(F.broadcast(top_types.select("Primary Type")), "Primary Type")

        return joined.filter(F.col("Location Description").isNotNull()) \
            .groupBy("Location Description").count() \
            .orderBy(F.desc("count")).limit(10)

    def draw_plot(self, pdf, title, path):
        plt.figure(figsize=(12, 8))
        plt.barh(pdf["Location Description"], pdf["count"], color='purple')
        plt.gca().invert_yaxis()
        plt.title("Топ-10 локацій для 5 найпопулярніших типів злочинів")
        plt.xlabel("Кількість")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

class Q4_J_Top_Streets(BaseQuestion):
    """group by"""
    def execute(self, df, spark=None):
        return df.groupBy("street_name").count() \
                 .orderBy(F.desc("count")) \
                 .limit(10)

    def draw_plot(self, pdf, title, path):
        plt.figure(figsize=(12, 7))

        bars = plt.barh(pdf["street_name"], pdf["count"], 
                        color='indianred', edgecolor='black')

        plt.gca().invert_yaxis()

        for bar in bars:
            width = bar.get_width()
            plt.text(width + (width * 0.005), bar.get_y() + bar.get_height()/2, 
                     f'{int(width):,}', va='center', fontsize=10, fontweight='bold')

        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Кількість злочинів", fontsize=12)
        plt.ylabel("Назва вулиці", fontsize=12)
        plt.grid(axis='x', linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.savefig(path)
        plt.close()


class Q5_J_Top_Crime_Per_District(BaseQuestion):
    """filter, group by, window"""
    def execute(self, df):
        window_spec = Window.partitionBy("District").orderBy(F.desc("count"))

        grouped = df.filter(F.col("District").isNotNull()) \
            .groupBy("District", "Primary Type").count()

        return grouped.withColumn("rank", F.rank().over(window_spec)) \
            .filter(F.col("rank") == 1) \
            .orderBy("District")

    def draw_plot(self, pdf, title, path):
        pdf = pdf.head(15)
        plt.figure(figsize=(12, 6))
        bars = plt.bar(pdf["District"].astype(str), pdf["count"], color='teal')
        plt.title("Найпопулярніший тип злочину по районах (Top 15 Districts)")
        plt.xlabel("District")
        plt.ylabel("Кількість")
        for bar, crime in zip(bars, pdf["Primary Type"]):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                     crime, ha='center', va='bottom', rotation=45, fontsize=8)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

class Q6_J_Most_Dangerous_Blocks_Share(BaseQuestion):
    """filter, group by, join, window"""
    def execute(self, df):
        dist_totals = df.filter(F.col("District").isNotNull()) \
            .groupBy("District").count().withColumnRenamed("count", "dist_total") \
            .filter(F.col("dist_total") > 1000) 

        block_counts = df.filter(F.col("Block").isNotNull()).groupBy("District", "Block").count()

        window_spec = Window.partitionBy("District").orderBy(F.desc("count"))
        top_blocks = block_counts.withColumn("rank", F.row_number().over(window_spec)) \
            .filter(F.col("rank") == 1)

        return top_blocks.join(dist_totals, "District") \
            .withColumn("share_percent", (F.col("count") / F.col("dist_total")) * 100) \
            .orderBy(F.desc("share_percent")).limit(10)

    def draw_plot(self, pdf, title, path):
        pdf_sorted = pdf.sort_values(by="share_percent", ascending=True)

        plt.figure(figsize=(12, 8))
        bars = plt.barh(pdf_sorted["Block"], pdf_sorted["share_percent"], color='steelblue', edgecolor='black')

        plt.title("Топ-10 кварталів-монополістів за часткою злочинів у своєму районі", fontsize=14)
        plt.xlabel("Частка від усіх злочинів району (%)")
        plt.ylabel("Квартал (Block)")

        plt.xlim(0, max(pdf_sorted["share_percent"]) * 1.4)

        for bar, count, dist_total, dist in zip(bars, pdf_sorted["count"], pdf_sorted["dist_total"], pdf_sorted["District"]):
            width = bar.get_width()
            label_text = f'{width:.1f}% ({count:,} з {dist_total:,} у Dist {dist})'
            plt.text(width + (max(pdf_sorted["share_percent"]) * 0.02), bar.get_y() + bar.get_height()/2, 
                     label_text, va='center', fontsize=9)

        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()


class Q1_M_Arrest_By_Type(BaseQuestion): 
    """filter, group by"""
    def execute(self, df, spark=None):
        return df.filter(F.col("Arrest") == "true") \
            .groupBy("Primary Type").count() \
            .orderBy(F.desc("count"))

    def draw_plot(self, pdf, title, path):
        top_pdf = pdf.head(20)

        plt.figure(figsize=(12, 9))

        bars = plt.barh(top_pdf["Primary Type"], top_pdf["count"],
                        color='forestgreen', edgecolor='black', alpha=0.8)

        plt.gca().invert_yaxis()

        for bar in bars:
            width = bar.get_width()
            plt.text(width + (width * 0.01), bar.get_y() + bar.get_height() / 2,
                     f'{int(width):,}', va='center', fontsize=10, fontweight='bold')

        plt.title(title + " (Top 20)", fontsize=14, fontweight='bold')
        plt.xlabel("Кількість арештів", fontsize=12)
        plt.ylabel("Тип злочину", fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(path)
        plt.close()


class Q2_M_Arrest_Share_2025(BaseQuestion): 
    """window"""
    def execute(self, df, spark=None):
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

    def draw_plot(self, pdf, title, path):
        top_pdf = pdf.head(20)
        plt.figure(figsize=(12, 9))

        plt.hlines(y=top_pdf["Primary Type"], xmin=0, xmax=top_pdf["share"], color='grey', alpha=0.5)
        plt.scatter(top_pdf["share"], top_pdf["Primary Type"], color='darkcyan', s=100, edgecolors='black')

        plt.gca().invert_yaxis()
        plt.title(title + " (Lollipop Chart)", fontsize=14, fontweight='bold')
        plt.xlabel("Частка арештів")
        plt.grid(axis='x', linestyle='--', alpha=0.3)

        for i, row in top_pdf.iterrows():
            plt.text(row['share'] + 0.01, i, f'{row["share"]*100:.1f}%', va='center')

        plt.tight_layout()
        plt.savefig(path)
        plt.close()


class Q3_M_Weekday_Vs_Weekend_By_District(BaseQuestion):
    """filter, group by, join"""
    def execute(self, df):
        df = df.withColumn("day_of_week", F.date_format(F.col("Date"), "E"))

        df_weekend = df.filter(F.col("day_of_week").isin("Sat", "Sun", "Сб", "Нд")) \
            .groupBy("District").count().withColumnRenamed("count", "weekend_count")

        df_weekday = df.filter(~F.col("day_of_week").isin("Sat", "Sun", "Сб", "Нд")) \
            .groupBy("District").count().withColumnRenamed("count", "weekday_count")

        return df_weekend.join(df_weekday, "District").orderBy(F.desc("weekend_count")).limit(10)

    def draw_plot(self, pdf, title, path):
        x = np.arange(len(pdf["District"]))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, pdf["weekday_count"], width, label='Будні', color='skyblue')
        plt.bar(x + width/2, pdf["weekend_count"], width, label='Вихідні', color='salmon')

        plt.title("Злочини: Будні vs Вихідні (Топ-10 District)")
        plt.xlabel("District")
        plt.ylabel("Кількість злочинів")
        plt.xticks(x, pdf["District"])
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()


class Q4_M_Domestic_Vs_NonDomestic_Streets(BaseQuestion): 
    """filter, group by, join, window"""
    def execute(self, df):
        df_dom = df.filter(F.col("Domestic") == True).groupBy("street_name").count().withColumnRenamed("count", "domestic")
        df_non = df.filter(F.col("Domestic") == False).groupBy("street_name").count().withColumnRenamed("count", "non_domestic")

        joined = df_dom.join(df_non, "street_name") \
            .withColumn("total", F.col("domestic") + F.col("non_domestic"))

        window_spec = Window.orderBy(F.desc("total"))
        return joined.withColumn("rank", F.rank().over(window_spec)) \
            .filter(F.col("rank") <= 10).orderBy("rank")

    def draw_plot(self, pdf, title, path):
        plt.figure(figsize=(12, 8))
        plt.barh(pdf["street_name"], pdf["non_domestic"], color='lightblue', label='Недомашні')
        plt.barh(pdf["street_name"], pdf["domestic"], left=pdf["non_domestic"], color='orange', label='Домашні')
        plt.gca().invert_yaxis()
        plt.title("Домашні та Недомашні злочини на Топ-10 вулицях")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()


class Q5_M_Crime_Type_Rank_In_Community(BaseQuestion): 
    """filter, group by, window"""
    def execute(self, df):
        filtered = df.filter(F.col("Community Area") == 40) \
            .groupBy("Primary Type").count()

        window_spec = Window.orderBy(F.desc("count"))
        return filtered.withColumn("rank", F.dense_rank().over(window_spec)) \
            .filter(F.col("rank") <= 10).orderBy("rank")

    def draw_plot(self, pdf, title, path):
        pdf_sorted = pdf.sort_values(by="count", ascending=True)

        plt.figure(figsize=(10, 6))
        bars = plt.barh(pdf_sorted["Primary Type"], pdf_sorted["count"], color='magenta', edgecolor='black')

        plt.title("Топ-10 злочинів у Community Area 40")
        plt.xlabel("Кількість злочинів")
        plt.ylabel("Тип злочину")

        for bar in bars:
            width = bar.get_width()
            plt.text(width + (max(pdf_sorted["count"]) * 0.01), bar.get_y() + bar.get_height()/2, 
                     f'{int(width):,}', va='center', fontsize=10)

        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()


class Q6_M_Hourly_All_City(BaseQuestion):
    """group by"""
    def execute(self, df, spark=None):
        return df.groupBy("hour").count().orderBy("hour")

    def draw_plot(self, pdf, title, path):
        plt.figure(figsize=(10, 10))

        angles = [n / 24.0 * 2 * np.pi for n in pdf["hour"].astype(float)]

        angles += angles[:1]
        values = pdf["count"].tolist()
        values += values[:1]

        ax = plt.subplot(111, projection='polar')

        ax.plot(angles, values, color='darkorange', linewidth=2, label='Кількість злочинів')
        ax.fill(angles, values, color='orange', alpha=0.3)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        ax.set_xticks(np.linspace(0, 2 * np.pi, 24, endpoint=False))
        ax.set_xticklabels([f"{h}h" for h in range(24)])

        plt.title(title, fontsize=15, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()