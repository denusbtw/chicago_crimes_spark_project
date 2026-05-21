import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import functions as F


def _to_pandas(df, limit_rows=200000):
    return df.limit(limit_rows).toPandas()


def _save_line_plot(pdf, x_col, y_col, title, xlabel, ylabel, out_path):
    if pdf.empty or x_col not in pdf.columns or y_col not in pdf.columns:
        return False

    plt.figure(figsize=(10, 6))
    plt.plot(pdf[x_col], pdf[y_col], marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    return True


def _save_area_plot(pdf, x_col, y_col, title, xlabel, ylabel, out_path):
    if pdf.empty or x_col not in pdf.columns or y_col not in pdf.columns:
        return False

    plt.figure(figsize=(10, 6))
    plt.fill_between(pdf[x_col], pdf[y_col], alpha=0.35)
    plt.plot(pdf[x_col], pdf[y_col], marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    return True


def _save_bar_plot(pdf, x_col, y_col, title, xlabel, ylabel, out_path, rotation=45):
    if pdf.empty or x_col not in pdf.columns or y_col not in pdf.columns:
        return False

    plt.figure(figsize=(11, 6))
    plt.bar(pdf[x_col].astype(str), pdf[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    return True


def _save_horizontal_bar_plot(pdf, x_col, y_col, title, xlabel, ylabel, out_path):
    if pdf.empty or x_col not in pdf.columns or y_col not in pdf.columns:
        return False

    pdf = pdf.copy()
    pdf = pdf.sort_values(by=x_col, ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(pdf[y_col].astype(str), pdf[x_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, axis="x")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    return True


def _save_hist_plot(pdf, col_name, title, xlabel, ylabel, out_path, bins=30):
    if pdf.empty or col_name not in pdf.columns:
        return False

    values = pdf[col_name].dropna()
    if values.empty:
        return False

    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    return True


def _save_boxplot(pdf, col_name, title, ylabel, out_path):
    if pdf.empty or col_name not in pdf.columns:
        return False

    values = pdf[col_name].dropna()
    if values.empty:
        return False

    plt.figure(figsize=(8, 6))
    plt.boxplot(values, vert=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    return True


def _save_scatter_plot(pdf, x_col, y_col, title, xlabel, ylabel, out_path):
    if pdf.empty or x_col not in pdf.columns or y_col not in pdf.columns:
        return False

    plt.figure(figsize=(10, 6))
    plt.scatter(pdf[x_col], pdf[y_col], alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    return True


def _save_pie_chart(pdf, labels_col, values_col, title, out_path):
    if pdf.empty or labels_col not in pdf.columns or values_col not in pdf.columns:
        return False

    plt.figure(figsize=(8, 8))
    plt.pie(pdf[values_col], labels=pdf[labels_col].astype(str), autopct="%1.1f%%", startangle=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    return True


def _save_heatmap(pdf, title, xlabel, ylabel, out_path):
    if pdf.empty:
        return False

    plt.figure(figsize=(12, 7))
    plt.imshow(pdf.values, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(range(len(pdf.columns)), [str(c) for c in pdf.columns], rotation=45, ha="right")
    plt.yticks(range(len(pdf.index)), [str(i) for i in pdf.index])
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    return True


def _plot_yearly_trend(df, out_dir, prefix):
    yearly_df = (
        df.groupBy(F.year("Date").alias("year"))
        .count()
        .orderBy("year")
    )
    pdf = _to_pandas(yearly_df)
    return _save_line_plot(
        pdf, "year", "count",
        f"{prefix}: динаміка за роками",
        "Рік", "Кількість",
        os.path.join(out_dir, f"{prefix}_yearly_trend.png")
    )


def _plot_yearly_area(df, out_dir, prefix):
    yearly_df = (
        df.groupBy(F.year("Date").alias("year"))
        .count()
        .orderBy("year")
    )
    pdf = _to_pandas(yearly_df)
    return _save_area_plot(
        pdf, "year", "count",
        f"{prefix}: площинна динаміка за роками",
        "Рік", "Кількість",
        os.path.join(out_dir, f"{prefix}_yearly_area.png")
    )


def _plot_monthly_distribution(df, out_dir, prefix):
    monthly_df = (
        df.groupBy(F.month("Date").alias("month"))
        .count()
        .orderBy("month")
    )
    pdf = _to_pandas(monthly_df)
    return _save_bar_plot(
        pdf, "month", "count",
        f"{prefix}: розподіл за місяцями",
        "Місяць", "Кількість",
        os.path.join(out_dir, f"{prefix}_monthly_distribution.png"),
        rotation=0
    )


def _plot_hourly_distribution(df, out_dir, prefix):
    hourly_df = (
        df.groupBy(F.hour("Date").alias("hour"))
        .count()
        .orderBy("hour")
    )
    pdf = _to_pandas(hourly_df)
    return _save_line_plot(
        pdf, "hour", "count",
        f"{prefix}: розподіл за годинами",
        "Година", "Кількість",
        os.path.join(out_dir, f"{prefix}_hourly_distribution.png")
    )


def _plot_hourly_bar(df, out_dir, prefix):
    hourly_df = (
        df.groupBy(F.hour("Date").alias("hour"))
        .count()
        .orderBy("hour")
    )
    pdf = _to_pandas(hourly_df)
    return _save_bar_plot(
        pdf, "hour", "count",
        f"{prefix}: гістограма по годинах",
        "Година", "Кількість",
        os.path.join(out_dir, f"{prefix}_hourly_bar.png"),
        rotation=0
    )


def _plot_district_horizontal(df, out_dir, prefix, top_n=15):
    if "District" not in df.columns:
        return False

    district_df = (
        df.groupBy("District")
        .count()
        .orderBy(F.desc("count"))
        .limit(top_n)
    )
    pdf = _to_pandas(district_df)
    return _save_horizontal_bar_plot(
        pdf,
        x_col="count",
        y_col="District",
        title=f"{prefix}: топ-{top_n} районів",
        xlabel="Кількість",
        ylabel="District",
        out_path=os.path.join(out_dir, f"{prefix}_district_horizontal.png")
    )


def _plot_location_horizontal(df, out_dir, prefix, top_n=10):
    if "Location Description" not in df.columns:
        return False

    location_df = (
        df.groupBy("Location Description")
        .count()
        .orderBy(F.desc("count"))
        .limit(top_n)
    )
    pdf = _to_pandas(location_df)
    return _save_horizontal_bar_plot(
        pdf,
        x_col="count",
        y_col="Location Description",
        title=f"{prefix}: топ-{top_n} локацій",
        xlabel="Кількість",
        ylabel="Location",
        out_path=os.path.join(out_dir, f"{prefix}_location_horizontal.png")
    )


def _plot_primary_type_horizontal(df, out_dir, prefix, top_n=10):
    if "Primary Type" not in df.columns:
        return False

    crime_type_df = (
        df.groupBy("Primary Type")
        .count()
        .orderBy(F.desc("count"))
        .limit(top_n)
    )
    pdf = _to_pandas(crime_type_df)
    return _save_horizontal_bar_plot(
        pdf,
        x_col="count",
        y_col="Primary Type",
        title=f"{prefix}: топ-{top_n} типів злочинів",
        xlabel="Кількість",
        ylabel="Primary Type",
        out_path=os.path.join(out_dir, f"{prefix}_primary_type_horizontal.png")
    )


def _plot_arrest_pie(df, out_dir, prefix):
    if "Arrest" not in df.columns:
        return False

    arrest_df = df.groupBy("Arrest").count().orderBy("Arrest")
    pdf = _to_pandas(arrest_df)
    return _save_pie_chart(
        pdf,
        labels_col="Arrest",
        values_col="count",
        title=f"{prefix}: частка арештів",
        out_path=os.path.join(out_dir, f"{prefix}_arrest_pie.png")
    )


def _plot_domestic_pie(df, out_dir, prefix):
    if "Domestic" not in df.columns:
        return False

    domestic_df = df.groupBy("Domestic").count().orderBy("Domestic")
    pdf = _to_pandas(domestic_df)
    return _save_pie_chart(
        pdf,
        labels_col="Domestic",
        values_col="count",
        title=f"{prefix}: частка domestic",
        out_path=os.path.join(out_dir, f"{prefix}_domestic_pie.png")
    )


def _plot_top_crime_types_from_result(result_df, out_dir, prefix, top_n=15):
    if "Primary Type" not in result_df.columns or "count" not in result_df.columns:
        return False

    pdf = _to_pandas(result_df.limit(top_n))
    return _save_horizontal_bar_plot(
        pdf,
        x_col="count",
        y_col="Primary Type",
        title=f"{prefix}: розподіл за Primary Type",
        xlabel="Кількість",
        ylabel="Primary Type",
        out_path=os.path.join(out_dir, f"{prefix}_result_primary_type.png")
    )


def _plot_top_fbi_codes_from_result(result_df, out_dir, prefix, top_n=15):
    if "FBI Code" not in result_df.columns or "count" not in result_df.columns:
        return False

    pdf = _to_pandas(result_df.limit(top_n))
    return _save_horizontal_bar_plot(
        pdf,
        x_col="count",
        y_col="FBI Code",
        title=f"{prefix}: розподіл за FBI Code",
        xlabel="Кількість",
        ylabel="FBI Code",
        out_path=os.path.join(out_dir, f"{prefix}_fbi_distribution.png")
    )


def _plot_month_counts_from_result(result_df, out_dir, prefix):
    if "month" not in result_df.columns or "count" not in result_df.columns:
        return False

    pdf = _to_pandas(result_df.orderBy("month"))
    return _save_area_plot(
        pdf,
        x_col="month",
        y_col="count",
        title=f"{prefix}: сезонна динаміка за місяцями",
        xlabel="Місяць",
        ylabel="Кількість",
        out_path=os.path.join(out_dir, f"{prefix}_result_months_area.png")
    )


def _plot_year_counts_from_result(result_df, out_dir, prefix):
    if "year" not in result_df.columns or "count" not in result_df.columns:
        return False

    pdf = _to_pandas(result_df.orderBy("year"))
    return _save_line_plot(
        pdf,
        x_col="year",
        y_col="count",
        title=f"{prefix}: динаміка за роками",
        xlabel="Рік",
        ylabel="Кількість",
        out_path=os.path.join(out_dir, f"{prefix}_result_years.png")
    )


def _plot_hour_counts_from_result(result_df, out_dir, prefix):
    if "hour" not in result_df.columns or "count" not in result_df.columns:
        return False

    pdf = _to_pandas(result_df.orderBy("hour"))
    return _save_bar_plot(
        pdf,
        x_col="hour",
        y_col="count",
        title=f"{prefix}: пікові години",
        xlabel="Година",
        ylabel="Кількість",
        out_path=os.path.join(out_dir, f"{prefix}_result_hours.png"),
        rotation=0
    )


def _plot_arrest_rate_from_result(result_df, out_dir, prefix, top_n=15):
    if "District" not in result_df.columns or "arrest_rate" not in result_df.columns:
        return False

    pdf = _to_pandas(result_df.orderBy(F.desc("arrest_rate")).limit(top_n))
    return _save_horizontal_bar_plot(
        pdf,
        x_col="arrest_rate",
        y_col="District",
        title=f"{prefix}: top-{top_n} arrest rate by district",
        xlabel="Arrest rate",
        ylabel="District",
        out_path=os.path.join(out_dir, f"{prefix}_arrest_rate.png")
    )


def _plot_rate_vs_volume_from_result(result_df, base_df, out_dir, prefix):
    if "District" not in result_df.columns or "arrest_rate" not in result_df.columns:
        return False
    if "District" not in base_df.columns:
        return False

    counts_df = (
        base_df.groupBy("District")
        .count()
        .withColumnRenamed("count", "crime_count")
    )

    merged_df = result_df.join(counts_df, on="District", how="inner")
    pdf = _to_pandas(merged_df)

    return _save_scatter_plot(
        pdf,
        x_col="crime_count",
        y_col="arrest_rate",
        title=f"{prefix}: arrest rate vs crime volume",
        xlabel="Crime count",
        ylabel="Arrest rate",
        out_path=os.path.join(out_dir, f"{prefix}_rate_vs_volume.png")
    )


def _plot_diff_seconds_hist_from_result(result_df, out_dir, prefix):
    if "diff_seconds" not in result_df.columns:
        return False

    pdf = _to_pandas(
        result_df.filter(F.col("diff_seconds").isNotNull()).select("diff_seconds"),
        limit_rows=100000
    )

    return _save_hist_plot(
        pdf,
        col_name="diff_seconds",
        title=f"{prefix}: розподіл часових інтервалів",
        xlabel="diff_seconds",
        ylabel="Частота",
        out_path=os.path.join(out_dir, f"{prefix}_diff_seconds_hist.png"),
        bins=40
    )


def _plot_diff_seconds_boxplot_from_result(result_df, out_dir, prefix):
    if "diff_seconds" not in result_df.columns:
        return False

    pdf = _to_pandas(
        result_df.filter(F.col("diff_seconds").isNotNull()).select("diff_seconds"),
        limit_rows=100000
    )

    return _save_boxplot(
        pdf,
        col_name="diff_seconds",
        title=f"{prefix}: boxplot часових інтервалів",
        ylabel="diff_seconds",
        out_path=os.path.join(out_dir, f"{prefix}_diff_seconds_boxplot.png")
    )


def _plot_district_ward_heatmap(base_df, out_dir, prefix):
    if "District" not in base_df.columns or "Ward" not in base_df.columns:
        return False

    matrix_df = (
        base_df.groupBy("District", "Ward")
        .count()
        .orderBy("District", "Ward")
    )

    pdf = _to_pandas(matrix_df)
    if pdf.empty:
        return False

    pivot = pdf.pivot(index="District", columns="Ward", values="count").fillna(0)
    return _save_heatmap(
        pivot,
        title=f"{prefix}: heatmap District x Ward",
        xlabel="Ward",
        ylabel="District",
        out_path=os.path.join(out_dir, f"{prefix}_district_ward_heatmap.png")
    )


def _plot_cumulative_sample_from_result(result_df, out_dir, prefix, top_n=10):
    if "running_total" not in result_df.columns or "Ward" not in result_df.columns or "Date" not in result_df.columns:
        return False

    wards = (
        result_df.groupBy("Ward")
        .agg(F.max("running_total").alias("max_total"))
        .orderBy(F.desc("max_total"))
        .limit(top_n)
        .select("Ward")
    )

    sample_df = result_df.join(wards, on="Ward", how="inner")
    pdf = _to_pandas(sample_df.select("Ward", "Date", "running_total"), limit_rows=50000)
    if pdf.empty:
        return False

    plt.figure(figsize=(11, 7))
    for ward, group in pdf.groupby("Ward"):
        group = group.sort_values("Date")
        plt.plot(group["Date"], group["running_total"], label=str(ward))

    plt.title(f"{prefix}: cumulative trajectories by Ward")
    plt.xlabel("Date")
    plt.ylabel("running_total")
    plt.grid(True)
    plt.legend(title="Ward", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_cumulative_sample.png"), dpi=220)
    plt.close()
    return True


def generate_question_plots(q_code, title, base_df, result_df, out_root="business_plots"):
    question_dir = os.path.join(out_root, q_code)
    os.makedirs(question_dir, exist_ok=True)

    created = []

    if q_code in {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q17", "Q19", "Q20"}:
        plot_specs = [
            (_plot_yearly_area, base_df),
            (_plot_monthly_distribution, base_df),
            (_plot_hourly_distribution, base_df),
            (_plot_location_horizontal, base_df),
            (_plot_district_horizontal, base_df),
            (_plot_arrest_pie, base_df),
            (_plot_domestic_pie, base_df),
            (_plot_primary_type_horizontal, base_df),
        ]

    elif q_code == "Q9":
        plot_specs = [
            (_plot_top_crime_types_from_result, result_df),
            (_plot_yearly_area, base_df),
            (_plot_location_horizontal, base_df),
            (_plot_arrest_pie, base_df),
        ]

    elif q_code == "Q10":
        plot_specs = [
            (_plot_arrest_rate_from_result, result_df),
            (_plot_rate_vs_volume_from_result, (result_df, base_df)),
            (_plot_yearly_trend, base_df),
            (_plot_monthly_distribution, base_df),
        ]

    elif q_code == "Q11":
        plot_specs = [
            (_plot_month_counts_from_result, result_df),
            (_plot_hourly_bar, base_df),
            (_plot_location_horizontal, base_df),
            (_plot_arrest_pie, base_df),
        ]

    elif q_code == "Q12":
        plot_specs = [
            (_plot_location_horizontal, base_df),
            (_plot_yearly_trend, base_df),
            (_plot_arrest_pie, base_df),
            (_plot_domestic_pie, base_df),
        ]

    elif q_code == "Q13":
        plot_specs = [
            (_plot_year_counts_from_result, result_df),
            (_plot_district_horizontal, base_df),
            (_plot_hourly_distribution, base_df),
            (_plot_location_horizontal, base_df),
        ]

    elif q_code == "Q14":
        plot_specs = [
            (_plot_top_fbi_codes_from_result, result_df),
            (_plot_yearly_trend, base_df),
            (_plot_primary_type_horizontal, base_df),
            (_plot_district_horizontal, base_df),
        ]

    elif q_code == "Q15":
        plot_specs = [
            (_plot_hour_counts_from_result, result_df),
            (_plot_yearly_trend, base_df),
            (_plot_location_horizontal, base_df),
            (_plot_arrest_pie, base_df),
        ]

    elif q_code == "Q16":
        plot_specs = [
            (_plot_district_ward_heatmap, base_df),
            (_plot_yearly_trend, base_df),
            (_plot_monthly_distribution, base_df),
        ]

    elif q_code == "Q21":
        plot_specs = [
            (_plot_yearly_trend, base_df),
            (_plot_district_horizontal, base_df),
            (_plot_hourly_distribution, base_df),
        ]

    elif q_code == "Q22":
        plot_specs = [
            (_plot_cumulative_sample_from_result, result_df),
            (_plot_yearly_trend, base_df),
            (_plot_monthly_distribution, base_df),
        ]

    elif q_code == "Q23":
        plot_specs = [
            (_plot_diff_seconds_hist_from_result, result_df),
            (_plot_diff_seconds_boxplot_from_result, result_df),
            (_plot_yearly_trend, base_df),
        ]

    elif q_code == "Q24":
        plot_specs = [
            (_plot_primary_type_horizontal, base_df),
            (_plot_district_horizontal, base_df),
            (_plot_yearly_trend, base_df),
        ]

    else:
        plot_specs = [
            (_plot_yearly_trend, base_df),
            (_plot_monthly_distribution, base_df),
            (_plot_district_horizontal, base_df),
        ]

    for item in plot_specs:
        try:
            plot_func = item[0]
            source = item[1]

            if isinstance(source, tuple):
                ok = plot_func(*source, question_dir, q_code)
            else:
                ok = plot_func(source, question_dir, q_code)

            if ok:
                created.append(plot_func.__name__)
        except Exception:
            continue

    with open(os.path.join(question_dir, "plots_info.txt"), "w", encoding="utf-8") as f:
        f.write("===== PLOTS INFO =====\n\n")
        f.write(f"Question: {q_code}\n")
        f.write(f"Title: {title}\n")
        f.write(f"Plots created: {len(created)}\n")
        for item in created:
            f.write(f"{item}\n")

    return question_dir