#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.decomposition
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing


parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    default="data/METABRIC_RNA_Mutation.csv"
)

parser.add_argument(
    "--output",
    default="results"
)

parser.add_argument(
    "--seed",
    default=0
)

args = parser.parse_args()
args.seed = int(args.seed)


def main():
    np.random.seed(args.seed)

    # Read data
    df = pd.read_csv(args.input, low_memory=False)

    # Keep patients whose 5-year outcome is known:
    # - survived at least 60 months
    # - OR died before 60 months
    # Exclude living patients followed for less than 5 years.
    known_5yr = (
        (df["overall_survival_months"] >= 60)
        |
        (
            (df["overall_survival"] == 0)
            &
            (df["overall_survival_months"] < 60)
        )
    )

    df = df.loc[known_5yr].copy()

    # Target:
    # 1 = survived at least 5 years
    # 0 = died before 5 years
    y = (
        df["overall_survival_months"] >= 60
    ).astype(int).values

    # Find RNA-expression columns
    gene_start = df.columns.get_loc("brca1")

    mutation_start = next(
        i
        for i, col in enumerate(df.columns)
        if col.endswith("_mut")
    )

    x_rna = df.iloc[
        :,
        gene_start:mutation_start
    ]

    # Encode ER status
    # 1 = ER positive
    # 0 = ER negative
    er = (
        df["er_status"] == "Positive"
    ).astype(int).values.reshape(-1, 1)

    # Model containing ER status + transcriptomic profile
    x_full = np.column_stack(
        [er, x_rna.values]
    )

    # Figure 1: 5-year survival counts
    counts = pd.Series(y).map(
        {
            0: "Died <5 years",
            1: "Survived ≥5 years"
        }
    ).value_counts()

    plt.figure(figsize=(5, 4))

    plt.bar(
        counts.index,
        counts.values
    )

    plt.ylabel("Patients")
    plt.title("Five-year survival groups")
    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        f"{args.output}/survival_counts.png",
        dpi=200
    )

    plt.close()

    # Figure 2: PCA
    scaler = sklearn.preprocessing.StandardScaler()

    x_scaled = scaler.fit_transform(
        x_rna
    )

    pca = sklearn.decomposition.PCA(
        n_components=2
    )

    pcs = pca.fit_transform(
        x_scaled
    )

    plt.figure(figsize=(5, 4))

    for value, label in [
        (0, "Died <5 years"),
        (1, "Survived ≥5 years")
    ]:
        mask = y == value

        plt.scatter(
            pcs[mask, 0],
            pcs[mask, 1],
            alpha=0.4,
            s=15,
            label=label
        )

    plt.xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)"
    )

    plt.ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)"
    )

    plt.title(
        "Tumor transcriptomic profiles"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"{args.output}/survival_pca.png",
        dpi=200
    )

    plt.close()

    # Classification
    cv = sklearn.model_selection.StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=args.seed
    )

    # Model 1: ER status only
    model_er = sklearn.linear_model.LogisticRegression(
        max_iter=5000
    )

    # Model 2: ER status + RNA expression
    model_full = sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.StandardScaler(),
        sklearn.linear_model.LogisticRegression(
            max_iter=5000,
            C=0.1
        )
    )

    # Cross-validated probability predictions
    pred_er = sklearn.model_selection.cross_val_predict(
        model_er,
        er,
        y,
        cv=cv,
        method="predict_proba"
    )[:, 1]

    pred_full = sklearn.model_selection.cross_val_predict(
        model_full,
        x_full,
        y,
        cv=cv,
        method="predict_proba"
    )[:, 1]

    # ROC analysis
    er_fpr, er_tpr, _ = sklearn.metrics.roc_curve(
        y,
        pred_er
    )

    full_fpr, full_tpr, _ = sklearn.metrics.roc_curve(
        y,
        pred_full
    )

    er_auc = sklearn.metrics.roc_auc_score(
        y,
        pred_er
    )

    full_auc = sklearn.metrics.roc_auc_score(
        y,
        pred_full
    )

    # Figure 3: ROC comparison
    plt.figure(figsize=(5, 4))

    plt.plot(
        er_fpr,
        er_tpr,
        label=f"ER only (AUC={er_auc:.2f})"
    )

    plt.plot(
        full_fpr,
        full_tpr,
        label=f"ER + RNA (AUC={full_auc:.2f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")

    plt.title(
        "Prediction of 5-year survival"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"{args.output}/survival_roc.png",
        dpi=200
    )

    plt.close()

    # Print results
    print(f"Patients analyzed: {len(df)}")
    print(f"Died before 5 years: {(y == 0).sum()}")
    print(f"Survived at least 5 years: {(y == 1).sum()}")
    print(f"ER-only AUROC: {er_auc:.3f}")
    print(
        f"ER + transcriptome AUROC: {full_auc:.3f}"
    )


if __name__ == "__main__":
    main()
