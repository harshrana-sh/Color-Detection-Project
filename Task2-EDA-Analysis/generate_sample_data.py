"""
generate_sample_data.py
------------------------
Builds data/iris.csv from sklearn's bundled Iris dataset (no internet
required) and injects a handful of realistic data-quality issues -
missing values and a duplicate row - so the EDA notebook's cleaning
steps have something real to demonstrate. Seeded for reproducibility.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


def main():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df = df.rename(columns={
        "sepal length (cm)": "sepal_length_cm",
        "sepal width (cm)": "sepal_width_cm",
        "petal length (cm)": "petal_length_cm",
        "petal width (cm)": "petal_width_cm",
        "target": "species_code",
    })
    target_names = dict(enumerate(iris.target_names))
    df["species"] = df["species_code"].map(target_names)
    df = df.drop(columns=["species_code"])

    # Inject a few missing values (seeded for reproducibility)
    rng = np.random.default_rng(42)
    missing_idx = rng.choice(df.index, size=6, replace=False)
    missing_cols = rng.choice(["sepal_length_cm", "petal_width_cm"], size=6)
    for idx, col in zip(missing_idx, missing_cols):
        df.loc[idx, col] = np.nan

    # Inject one duplicate row
    dup_row = df.iloc[[10]].copy()
    df = pd.concat([df, dup_row], ignore_index=True)

    import os
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/iris.csv", index=False)
    print(f"Wrote data/iris.csv with shape {df.shape}")


if __name__ == "__main__":
    main()
