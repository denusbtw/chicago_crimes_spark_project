import os
import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.linalg import DenseVector, SparseVector
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, BooleanType


def _prepare_numeric_df(df):
    numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]
    bool_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, BooleanType)]

    selected_cols = numeric_cols + bool_cols

    work_df = df
    for col_name in bool_cols:
        work_df = work_df.withColumn(col_name, F.col(col_name).cast("double"))

    return work_df.select(*selected_cols), selected_cols


def _sample_df(df, fraction=0.02, seed=42, max_rows=100000):
    sampled = df.sample(withReplacement=False, fraction=fraction, seed=seed)
    count_sampled = sampled.count()

    if count_sampled > max_rows:
        ratio = max_rows / count_sampled
        sampled = sampled.sample(withReplacement=False, fraction=ratio, seed=seed)

    return sampled


def _vector_to_list(v):
    if isinstance(v, DenseVector):
        return list(v)
    if isinstance(v, SparseVector):
        return v.toArray().tolist()
    return list(v)


def _save_summary(explained, out_path):
    cumulative = []
    running = 0.0
    for value in explained:
        running += float(value)
        cumulative.append(running)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"===== PCA with {len(explained)} components =====\n\n")

        f.write("Explained variance by components:\n")
        f.write(
            f"{'Component':<12}"
            f"{'Explained variance ratio':<28}"
            f"{'Explained variance (%)':<24}\n"
        )

        for i, ev in enumerate(explained, start=1):
            f.write(
                f"{f'PC{i}':<12}"
                f"{ev:<28.4f}"
                f"{ev * 100:<24.4f}\n"
            )

        f.write("\n")
        f.write("Cumulative explained variance (%):\n")
        f.write(f"{'Component':<12}{'Cumulative explained variance (%)':<32}\n")

        for i, cum in enumerate(cumulative, start=1):
            f.write(
                f"{f'PC{i}':<12}"
                f"{cum * 100:<32.4f}\n"
            )

        f.write("\n")
        f.write(
            f"Total explained variance by first {len(explained)} components: "
            f"{cumulative[-1] * 100:.2f}%\n"
        )

def _save_loadings(pc_matrix, feature_names, out_path):
    arr = pc_matrix.toArray()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("PCA LOADINGS\n")
        f.write("=" * 60 + "\n\n")

        for comp_idx in range(arr.shape[1]):
            f.write(f"PC{comp_idx + 1}\n")
            f.write("-" * 40 + "\n")

            pairs = []
            for feat_idx, feat_name in enumerate(feature_names):
                weight = float(arr[feat_idx, comp_idx])
                pairs.append((feat_name, weight, abs(weight)))

            pairs.sort(key=lambda x: x[2], reverse=True)

            for feat_name, weight, _ in pairs:
                f.write(f"{feat_name}: {weight:.6f}\n")

            f.write("\n")


def _plot_cumulative_variance(explained, out_path):
    cumulative = []
    running = 0.0
    for value in explained:
        running += float(value)
        cumulative.append(running * 100)

    labels = [f"PC{i}" for i in range(1, len(cumulative) + 1)]

    plt.figure(figsize=(10, 6))
    plt.plot(labels, cumulative, marker="o")
    plt.xlabel("Principal Components")
    plt.ylabel("Cumulative Explained Variance (%)")
    plt.title(f"Cumulative Explained Variance ({len(labels)} Components)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_2d(scores_pdf, out_path):
    if not {"PC1", "PC2"}.issubset(scores_pdf.columns):
        return

    plt.figure(figsize=(10, 7))
    plt.scatter(scores_pdf["PC1"], scores_pdf["PC2"], s=8, alpha=0.5)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA 2D Projection")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_3d(scores_pdf, out_path):
    if not {"PC1", "PC2", "PC3"}.issubset(scores_pdf.columns):
        return

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(scores_pdf["PC1"], scores_pdf["PC2"], scores_pdf["PC3"], s=8, alpha=0.45)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA 3D Projection")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def run_pca_analysis(
    df,
    out_dir="pca_results",
    n_components=6,
    sample_fraction=0.02,
    sample_max_rows=100000,
    seed=42
):
    os.makedirs(out_dir, exist_ok=True)

    numeric_df, feature_names = _prepare_numeric_df(df)

    sampled_df = _sample_df(
        numeric_df,
        fraction=sample_fraction,
        seed=seed,
        max_rows=sample_max_rows
    ).dropna()

    feature_count = len(feature_names)
    if feature_count < 2:
        raise ValueError("Not enough numeric features for PCA.")

    effective_k = min(n_components, feature_count)

    assembler = VectorAssembler(
        inputCols=feature_names,
        outputCol="features",
        handleInvalid="skip"
    )

    assembled_df = assembler.transform(sampled_df).select("features")

    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaled_features",
        withStd=True,
        withMean=True
    )

    scaler_model = scaler.fit(assembled_df)
    scaled_df = scaler_model.transform(assembled_df)

    pca = PCA(
        k=effective_k,
        inputCol="scaled_features",
        outputCol="pca_features"
    )

    pca_model = pca.fit(scaled_df)
    pca_df = pca_model.transform(scaled_df)

    explained = [float(x) for x in pca_model.explainedVariance]
    _save_summary(explained, os.path.join(out_dir, "pca_summary.txt"))
    _save_loadings(pca_model.pc, feature_names, os.path.join(out_dir, "pca_loadings.txt"))
    _plot_cumulative_variance(explained, os.path.join(out_dir, "pca_cumulative_variance.png"))

    scores_rows = pca_df.select("pca_features").limit(sample_max_rows).collect()
    scores = [_vector_to_list(row["pca_features"]) for row in scores_rows]

    score_cols = [f"PC{i}" for i in range(1, effective_k + 1)]
    scores_pdf = pd.DataFrame(scores, columns=score_cols)
    scores_pdf.to_csv(os.path.join(out_dir, "pca_scores_sample.csv"), index=False)

    if effective_k >= 2:
        _plot_2d(scores_pdf, os.path.join(out_dir, "pca_2d_scatter.png"))

    if effective_k >= 3:
        _plot_3d(scores_pdf, os.path.join(out_dir, "pca_3d_scatter.png"))

    with open(os.path.join(out_dir, "pca_run_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"Input numeric features: {feature_names}\n")
        f.write(f"Number of numeric features: {feature_count}\n")
        f.write(f"Requested components: {n_components}\n")
        f.write(f"Effective components: {effective_k}\n")
        f.write(f"Sample fraction: {sample_fraction}\n")
        f.write(f"Sample max rows: {sample_max_rows}\n")
        f.write(f"Used rows after sampling and dropna: {scores_pdf.shape[0]}\n")

    return {
        "feature_names": feature_names,
        "explained_variance": explained,
        "out_dir": out_dir
    }