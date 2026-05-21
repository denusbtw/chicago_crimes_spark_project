import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import functions as F


def _sample_spark_df(df, fraction=0.01, max_rows=20000, seed=42):
    sampled = df.sample(withReplacement=False, fraction=fraction, seed=seed)
    if max_rows is not None:
        sampled = sampled.limit(max_rows)
    return sampled


def _collapse_top_categories(series, top_n=20, other_label="OTHER"):
    series = series.astype("object").fillna("Unknown")
    top_values = series.value_counts(dropna=False).head(top_n).index
    return series.where(series.isin(top_values), other_label)


def _prepare_famd_dataframe(
    pdf,
    numeric_cols,
    categorical_cols,
    collapse_top_n=None
):
    if numeric_cols is None or categorical_cols is None:
        raise ValueError("numeric_cols and categorical_cols must be provided explicitly.")

    existing_numeric = [c for c in numeric_cols if c in pdf.columns]
    existing_categorical = [c for c in categorical_cols if c in pdf.columns]

    missing_numeric = [c for c in numeric_cols if c not in pdf.columns]
    missing_categorical = [c for c in categorical_cols if c not in pdf.columns]

    if missing_numeric:
        print(f"Warning: missing numeric columns: {missing_numeric}")

    if missing_categorical:
        print(f"Warning: missing categorical columns: {missing_categorical}")

    selected_cols = existing_numeric + existing_categorical
    if not selected_cols:
        raise ValueError("No valid columns found for FAMD.")

    work = pdf[selected_cols].copy()

    for col in existing_numeric:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        median_value = work[col].median()
        work[col] = work[col].fillna(median_value)

    for col in existing_categorical:
        work[col] = work[col].astype("object").fillna("Unknown")

    if collapse_top_n is not None:
        for col, top_n in collapse_top_n.items():
            if col in work.columns:
                work[col] = _collapse_top_categories(work[col], top_n=top_n)

    for col in existing_categorical:
        work[col] = work[col].astype("category")

    return work, existing_numeric, existing_categorical


def _get_variance_info(model):
    explained_ratio = None
    cumulative_ratio = None

    if hasattr(model, "percentage_of_variance_"):
        explained_ratio = [float(x) / 100.0 for x in model.percentage_of_variance_]

    if hasattr(model, "cumulative_percentage_of_variance_"):
        cumulative_ratio = [float(x) / 100.0 for x in model.cumulative_percentage_of_variance_]

    if explained_ratio is None and hasattr(model, "eigenvalues_"):
        eigenvalues = [float(x) for x in model.eigenvalues_]
        total = sum(eigenvalues)
        if total > 0:
            explained_ratio = [x / total for x in eigenvalues]

    if cumulative_ratio is None and explained_ratio is not None:
        cumulative_ratio = []
        running = 0.0
        for value in explained_ratio:
            running += value
            cumulative_ratio.append(running)

    return explained_ratio, cumulative_ratio


def _save_summary(explained_ratio, cumulative_ratio, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"===== FAMD with {len(explained_ratio)} components =====\n\n")

        f.write("Explained variance by components:\n")
        f.write(
            f"{'Component':<12}"
            f"{'Explained variance ratio':<28}"
            f"{'Explained variance (%)':<24}\n"
        )

        for i, ev in enumerate(explained_ratio, start=1):
            f.write(
                f"{f'FAMD{i}':<12}"
                f"{ev:<28.6f}"
                f"{ev * 100:<24.4f}\n"
            )

        f.write("\n")
        f.write("Cumulative explained variance (%):\n")
        f.write(f"{'Component':<12}{'Cumulative explained variance (%)':<32}\n")

        for i, cum in enumerate(cumulative_ratio, start=1):
            f.write(
                f"{f'FAMD{i}':<12}"
                f"{cum * 100:<32.4f}\n"
            )

        f.write("\n")
        f.write(
            f"Total explained variance by first {len(explained_ratio)} components: "
            f"{cumulative_ratio[-1] * 100:.2f}%\n"
        )

        thresholds = [0.7, 0.8, 0.9, 0.95]
        f.write("\n")
        for threshold in thresholds:
            needed = None
            for i, cum in enumerate(cumulative_ratio, start=1):
                if cum >= threshold:
                    needed = i
                    break

            if needed is None:
                f.write(
                    f"Threshold {threshold * 100:.0f}% was not reached. "
                    f"Maximum cumulative variance = {cumulative_ratio[-1] * 100:.4f}%\n"
                )
            else:
                f.write(
                    f"Components needed for {threshold * 100:.0f}% variance: {needed}\n"
                )


def _save_dataframe_txt(df, out_path, title):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(title + "\n")
        f.write("=" * len(title) + "\n\n")
        f.write(df.to_string())
        f.write("\n")


def _plot_cumulative_variance(explained_ratio, cumulative_ratio, out_path):
    labels = [f"FAMD{i}" for i in range(1, len(explained_ratio) + 1)]
    y = [x * 100 for x in cumulative_ratio]

    plt.figure(figsize=(10, 6))
    plt.plot(labels, y, marker="o")
    plt.xlabel("Components")
    plt.ylabel("Cumulative Explained Variance (%)")
    plt.title("FAMD Cumulative Explained Variance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_2d(coords_df, out_path):
    if not {"Dim1", "Dim2"}.issubset(coords_df.columns):
        return

    plt.figure(figsize=(10, 7))
    plt.scatter(coords_df["Dim1"], coords_df["Dim2"], s=8, alpha=0.45)
    plt.xlabel("Dim1")
    plt.ylabel("Dim2")
    plt.title("FAMD 2D Projection")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_3d(coords_df, out_path):
    if not {"Dim1", "Dim2", "Dim3"}.issubset(coords_df.columns):
        return

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(coords_df["Dim1"], coords_df["Dim2"], coords_df["Dim3"], s=8, alpha=0.4)
    ax.set_xlabel("Dim1")
    ax.set_ylabel("Dim2")
    ax.set_zlabel("Dim3")
    ax.set_title("FAMD 3D Projection")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _safe_column_coordinates(model, df):
    if hasattr(model, "column_coordinates"):
        try:
            return model.column_coordinates(df)
        except Exception:
            return None
    return None


def _safe_column_contributions(model, df):
    if hasattr(model, "column_contributions_"):
        try:
            return model.column_contributions_
        except Exception:
            return None

    if hasattr(model, "column_contributions"):
        try:
            return model.column_contributions(df)
        except Exception:
            return None

    return None


def run_famd_analysis(
    spark_df,
    out_dir="famd_results",
    sample_fraction=0.01,
    sample_max_rows=20000,
    n_components=6,
    seed=42,
    numeric_cols=None,
    categorical_cols=None,
    collapse_top_n=None
):
    import prince

    if numeric_cols is None or categorical_cols is None:
        raise ValueError("You must explicitly provide numeric_cols and categorical_cols for FAMD.")

    os.makedirs(out_dir, exist_ok=True)

    sampled_spark_df = _sample_spark_df(
        spark_df,
        fraction=sample_fraction,
        max_rows=sample_max_rows,
        seed=seed
    )

    pdf = sampled_spark_df.toPandas()

    famd_df, used_numeric, used_categorical = _prepare_famd_dataframe(
        pdf,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        collapse_top_n=collapse_top_n
    )

    if famd_df.empty:
        raise ValueError("Prepared FAMD dataframe is empty.")

    effective_k = min(n_components, famd_df.shape[1], max(1, famd_df.shape[0] - 1))

    famd = prince.FAMD(
        n_components=effective_k,
        random_state=seed,
        engine="sklearn"
    )

    famd = famd.fit(famd_df)

    row_coords = famd.row_coordinates(famd_df)
    row_coords = row_coords.copy()
    row_coords.columns = [f"Dim{i}" for i in range(1, row_coords.shape[1] + 1)]
    row_coords.to_csv(os.path.join(out_dir, "famd_row_coordinates.csv"), index=False)

    explained_ratio, cumulative_ratio = _get_variance_info(famd)
    if explained_ratio is None or cumulative_ratio is None:
        raise ValueError("Could not extract explained variance from FAMD model.")

    _save_summary(
        explained_ratio,
        cumulative_ratio,
        os.path.join(out_dir, "famd_summary.txt")
    )

    column_coords = _safe_column_coordinates(famd, famd_df)
    if column_coords is not None and isinstance(column_coords, pd.DataFrame):
        column_coords.to_csv(os.path.join(out_dir, "famd_column_coordinates.csv"))
        _save_dataframe_txt(
            column_coords,
            os.path.join(out_dir, "famd_column_coordinates.txt"),
            "FAMD COLUMN COORDINATES"
        )

    column_contrib = _safe_column_contributions(famd, famd_df)
    if column_contrib is not None and isinstance(column_contrib, pd.DataFrame):
        column_contrib.to_csv(os.path.join(out_dir, "famd_column_contributions.csv"))
        _save_dataframe_txt(
            column_contrib,
            os.path.join(out_dir, "famd_column_contributions.txt"),
            "FAMD COLUMN CONTRIBUTIONS"
        )

    _plot_cumulative_variance(
        explained_ratio,
        cumulative_ratio,
        os.path.join(out_dir, "famd_cumulative_variance.png")
    )

    if effective_k >= 2:
        _plot_2d(row_coords, os.path.join(out_dir, "famd_2d_scatter.png"))

    if effective_k >= 3:
        _plot_3d(row_coords, os.path.join(out_dir, "famd_3d_scatter.png"))

    with open(os.path.join(out_dir, "famd_run_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"Input rows after sampling: {len(pdf)}\n")
        f.write(f"Used numeric features: {used_numeric}\n")
        f.write(f"Used categorical features: {used_categorical}\n")
        f.write(f"Requested components: {n_components}\n")
        f.write(f"Effective components: {effective_k}\n")
        f.write(f"Sample fraction: {sample_fraction}\n")
        f.write(f"Sample max rows: {sample_max_rows}\n")
        f.write(f"Final dataframe shape for FAMD: {famd_df.shape}\n")

    return {
        "out_dir": out_dir,
        "used_numeric": used_numeric,
        "used_categorical": used_categorical,
        "explained_ratio": explained_ratio,
        "cumulative_ratio": cumulative_ratio
    }