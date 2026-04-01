import os
import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, BooleanType
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.linalg import DenseVector, SparseVector


def _get_binary_distribution(df, target_col):
    rows = df.groupBy(target_col).count().collect()

    if len(rows) != 2:
        raise ValueError(f"Target '{target_col}' must have exactly 2 classes.")

    distribution = {row[target_col]: int(row["count"]) for row in rows}
    sorted_items = sorted(distribution.items(), key=lambda x: x[1])

    minority_value, minority_count = sorted_items[0]
    majority_value, majority_count = sorted_items[1]

    return {
        "distribution": distribution,
        "minority_value": minority_value,
        "minority_count": minority_count,
        "majority_value": majority_value,
        "majority_count": majority_count
    }


def _prepare_modeling_df(df, target_col):
    numeric_cols = [
        f.name for f in df.schema.fields
        if isinstance(f.dataType, NumericType) and f.name != target_col
    ]

    bool_cols = [
        f.name for f in df.schema.fields
        if isinstance(f.dataType, BooleanType) and f.name != target_col
    ]

    feature_cols = numeric_cols + bool_cols

    if len(feature_cols) < 2:
        raise ValueError(f"Not enough usable numeric/bool features for target '{target_col}'.")

    work_df = df

    for col_name in bool_cols:
        work_df = work_df.withColumn(col_name, F.col(col_name).cast("double"))

    if isinstance(dict(df.dtypes)[target_col], str) and dict(df.dtypes)[target_col] == "boolean":
        work_df = work_df.withColumn(target_col, F.col(target_col).cast("boolean"))

    return work_df.select(*feature_cols, target_col), feature_cols


def _vector_to_list(v):
    if isinstance(v, DenseVector):
        return list(v)
    if isinstance(v, SparseVector):
        return v.toArray().tolist()
    return list(v)


def _sample_exact_without_replacement(df, required_count, total_count, seed=42):
    if required_count >= total_count:
        return df

    fraction = min(1.0, required_count / total_count * 1.15)
    sampled = df.sample(withReplacement=False, fraction=fraction, seed=seed)
    sampled_count = sampled.count()

    attempts = 0
    while sampled_count < required_count and attempts < 5:
        fraction = min(1.0, fraction * 1.25)
        sampled = df.sample(withReplacement=False, fraction=fraction, seed=seed + attempts + 1)
        sampled_count = sampled.count()
        attempts += 1

    return sampled.limit(required_count)


def _sample_exact_with_replacement(df, required_count, base_count, seed=42):
    if required_count <= 0:
        return df.limit(0)

    fraction = max(1.0, required_count / base_count * 1.10)
    sampled = df.sample(withReplacement=True, fraction=fraction, seed=seed)
    sampled_count = sampled.count()

    attempts = 0
    while sampled_count < required_count and attempts < 5:
        fraction *= 1.25
        sampled = df.sample(withReplacement=True, fraction=fraction, seed=seed + attempts + 1)
        sampled_count = sampled.count()
        attempts += 1

    return sampled.limit(required_count)


def _oversample_to_balance(df, target_col, seed=42):
    info = _get_binary_distribution(df, target_col)

    minority_value = info["minority_value"]
    minority_count = info["minority_count"]
    majority_value = info["majority_value"]
    majority_count = info["majority_count"]

    minority_df = df.filter(F.col(target_col) == minority_value)
    majority_df = df.filter(F.col(target_col) == majority_value)

    oversampled_minority = _sample_exact_with_replacement(
        minority_df,
        required_count=majority_count,
        base_count=minority_count,
        seed=seed
    )

    result_df = majority_df.unionByName(oversampled_minority)

    return result_df, info


def _undersample_to_balance(df, target_col, seed=42):
    info = _get_binary_distribution(df, target_col)

    minority_value = info["minority_value"]
    minority_count = info["minority_count"]
    majority_value = info["majority_value"]
    majority_count = info["majority_count"]

    minority_df = df.filter(F.col(target_col) == minority_value)
    majority_df = df.filter(F.col(target_col) == majority_value)

    sampled_majority = _sample_exact_without_replacement(
        majority_df,
        required_count=minority_count,
        total_count=majority_count,
        seed=seed
    )

    result_df = minority_df.unionByName(sampled_majority)

    return result_df, info


def _build_pca_projection(df, target_col, sample_max_rows=50000, seed=42):
    work_df, feature_names = _prepare_modeling_df(df, target_col)

    total_count = work_df.count()
    if total_count == 0:
        raise ValueError(f"Empty dataframe for PCA for target '{target_col}'.")

    if total_count > sample_max_rows:
        fraction = min(1.0, sample_max_rows / total_count * 1.15)
        sampled_df = work_df.sample(withReplacement=False, fraction=fraction, seed=seed)
        sampled_df = sampled_df.limit(sample_max_rows)
    else:
        sampled_df = work_df

    sampled_df = sampled_df.dropna()

    assembler = VectorAssembler(
        inputCols=feature_names,
        outputCol="features",
        handleInvalid="skip"
    )

    assembled_df = assembler.transform(sampled_df)

    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaled_features",
        withStd=True,
        withMean=True
    )

    scaler_model = scaler.fit(assembled_df)
    scaled_df = scaler_model.transform(assembled_df)

    pca = PCA(
        k=2,
        inputCol="scaled_features",
        outputCol="pca_features"
    )

    pca_model = pca.fit(scaled_df)
    pca_df = pca_model.transform(scaled_df).select(target_col, "pca_features")

    rows = pca_df.collect()
    data = []

    for row in rows:
        coords = _vector_to_list(row["pca_features"])
        data.append({
            target_col: row[target_col],
            "PC1": coords[0],
            "PC2": coords[1]
        })

    pdf = pd.DataFrame(data)
    explained = [float(x) for x in pca_model.explainedVariance]

    return pdf, explained, feature_names


def _plot_pca_scatter(pdf, target_col, out_path, title):
    if pdf.empty:
        return

    plt.figure(figsize=(10, 7))

    for label, group in pdf.groupby(target_col):
        plt.scatter(group["PC1"], group["PC2"], s=8, alpha=0.5, label=str(label))

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend(title=target_col)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _save_distribution_report(before_info, after_df, target_col, method, out_path, feature_cols):
    after_counts = {
        row[target_col]: int(row["count"])
        for row in after_df.groupBy(target_col).count().collect()
    }

    before_distribution = before_info["distribution"]

    before_minority = before_info["minority_value"]
    before_minority_count = before_info["minority_count"]
    before_majority = before_info["majority_value"]
    before_majority_count = before_info["majority_count"]

    after_minority_count = after_counts.get(before_minority, 0)
    after_majority_count = after_counts.get(before_majority, 0)

    minority_delta = after_minority_count - before_minority_count
    majority_delta = after_majority_count - before_majority_count

    minority_ratio_before = before_minority_count / before_majority_count
    minority_ratio_after = after_minority_count / after_majority_count if after_majority_count != 0 else 0.0

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("===== SAMPLING REPORT =====\n\n")
        f.write(f"Target column: {target_col}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Modeling features count: {len(feature_cols)}\n")
        f.write(f"Modeling features: {feature_cols}\n\n")

        f.write("Before sampling\n")
        f.write("----------------\n")
        for key, value in sorted(before_distribution.items(), key=lambda x: str(x[0])):
            f.write(f"{key}: {value}\n")
        f.write("\n")

        f.write("Detected roles\n")
        f.write("--------------\n")
        f.write(f"Minority class: {before_minority} ({before_minority_count})\n")
        f.write(f"Majority class: {before_majority} ({before_majority_count})\n")
        f.write(f"Minority / Majority ratio before: {minority_ratio_before:.6f}\n\n")

        f.write("After sampling\n")
        f.write("--------------\n")
        for key, value in sorted(after_counts.items(), key=lambda x: str(x[0])):
            f.write(f"{key}: {value}\n")
        f.write("\n")

        f.write("Change summary\n")
        f.write("--------------\n")
        f.write(f"Minority class change: {minority_delta:+d}\n")
        f.write(f"Majority class change: {majority_delta:+d}\n")
        f.write(f"Minority / Majority ratio after: {minority_ratio_after:.6f}\n")

        if method == "oversampling":
            growth_pct = (minority_delta / before_minority_count * 100.0) if before_minority_count > 0 else 0.0
            f.write(f"Minority class increase (%): {growth_pct:.4f}\n")

        if method == "undersampling":
            shrink_pct = (-majority_delta / before_majority_count * 100.0) if before_majority_count > 0 else 0.0
            f.write(f"Majority class reduction (%): {shrink_pct:.4f}\n")


def _save_pca_report(explained_before, explained_after, out_path, target_col, method):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("===== PCA SAMPLING COMPARISON =====\n\n")
        f.write(f"Target column: {target_col}\n")
        f.write(f"Method: {method}\n\n")

        f.write("Before sampling\n")
        f.write("---------------\n")
        f.write(f"PC1 explained variance: {explained_before[0]:.6f}\n")
        f.write(f"PC2 explained variance: {explained_before[1]:.6f}\n")
        f.write(f"Cumulative: {sum(explained_before):.6f}\n\n")

        f.write("After sampling\n")
        f.write("--------------\n")
        f.write(f"PC1 explained variance: {explained_after[0]:.6f}\n")
        f.write(f"PC2 explained variance: {explained_after[1]:.6f}\n")
        f.write(f"Cumulative: {sum(explained_after):.6f}\n")


def _run_single_sampling_case(
    df,
    target_col,
    method,
    out_dir,
    seed=42,
    pca_sample_max_rows=50000
):
    os.makedirs(out_dir, exist_ok=True)

    modeling_df, feature_cols = _prepare_modeling_df(df, target_col)
    modeling_df = modeling_df.dropna()

    before_info = _get_binary_distribution(modeling_df, target_col)

    if method == "oversampling":
        sampled_df, before_info = _oversample_to_balance(modeling_df, target_col, seed=seed)
    elif method == "undersampling":
        sampled_df, before_info = _undersample_to_balance(modeling_df, target_col, seed=seed)
    else:
        raise ValueError("Method must be 'oversampling' or 'undersampling'.")

    sampled_path = os.path.join(out_dir, "sampled_dataset")
    sampled_df.write.mode("overwrite").parquet(sampled_path)

    _save_distribution_report(
        before_info=before_info,
        after_df=sampled_df,
        target_col=target_col,
        method=method,
        out_path=os.path.join(out_dir, "sampling_report.txt"),
        feature_cols=feature_cols
    )

    before_pca_pdf, explained_before, before_feature_cols = _build_pca_projection(
        modeling_df,
        target_col=target_col,
        sample_max_rows=pca_sample_max_rows,
        seed=seed
    )

    after_pca_pdf, explained_after, after_feature_cols = _build_pca_projection(
        sampled_df,
        target_col=target_col,
        sample_max_rows=pca_sample_max_rows,
        seed=seed
    )

    before_pca_pdf.to_csv(os.path.join(out_dir, "pca_before.csv"), index=False)
    after_pca_pdf.to_csv(os.path.join(out_dir, "pca_after.csv"), index=False)

    _plot_pca_scatter(
        before_pca_pdf,
        target_col=target_col,
        out_path=os.path.join(out_dir, "pca_before.png"),
        title=f"PCA before sampling: {target_col}"
    )

    _plot_pca_scatter(
        after_pca_pdf,
        target_col=target_col,
        out_path=os.path.join(out_dir, "pca_after.png"),
        title=f"PCA after {method}: {target_col}"
    )

    _save_pca_report(
        explained_before=explained_before,
        explained_after=explained_after,
        out_path=os.path.join(out_dir, "pca_report.txt"),
        target_col=target_col,
        method=method
    )

    with open(os.path.join(out_dir, "run_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"Target column: {target_col}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Feature columns used: {feature_cols}\n")
        f.write(f"PCA sample max rows: {pca_sample_max_rows}\n")
        f.write(f"Rows in modeling_df: {modeling_df.count()}\n")
        f.write(f"Rows in sampled_df: {sampled_df.count()}\n")
        f.write(f"PCA features before: {before_feature_cols}\n")
        f.write(f"PCA features after: {after_feature_cols}\n")

    return {
        "sampled_path": sampled_path,
        "report_path": os.path.join(out_dir, "sampling_report.txt"),
        "pca_before_path": os.path.join(out_dir, "pca_before.png"),
        "pca_after_path": os.path.join(out_dir, "pca_after.png"),
        "run_info_path": os.path.join(out_dir, "run_info.txt")
    }


def run_sampling_analysis(
    df,
    out_dir="sampling_results",
    target_cols=("Domestic", "Arrest"),
    methods=("oversampling", "undersampling"),
    seed=42,
    pca_sample_max_rows=50000
):
    os.makedirs(out_dir, exist_ok=True)

    results = {}

    for target_col in target_cols:
        results[target_col] = {}

        for method in methods:
            case_out_dir = os.path.join(out_dir, target_col.lower(), method)
            results[target_col][method] = _run_single_sampling_case(
                df=df,
                target_col=target_col,
                method=method,
                out_dir=case_out_dir,
                seed=seed,
                pca_sample_max_rows=pca_sample_max_rows
            )

    return results