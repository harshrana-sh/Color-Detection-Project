# EDA Insights Summary — Iris Dataset

Based on `EDA_Analysis.ipynb` and `report.html`. See those files for full
detail, charts, and statistics behind each point.

- **Data quality was mostly clean but not perfect.** Of 151 rows, 6 values
  (~4% of `petal_width_cm`) were missing and 1 exact duplicate row was
  present. Missingness was handled with median imputation; the duplicate
  was dropped. No other columns had missing or malformed values.

- **Classes are perfectly balanced.** 50 setosa, 50 versicolor, 50
  virginica (pre-cleaning). This means no resampling or class-weighting
  would be needed if this dataset were used for classification.

- **Petal measurements are the most informative features.** `petal_length_cm`
  and `petal_width_cm` show clear multi-modal distributions and separate
  the three species far better than the sepal measurements do — visible
  in both the histograms and the pairplot.

- **`petal_length_cm` and `petal_width_cm` are highly redundant**
  (Pearson r ≈ 0.96). Both also correlate strongly with `sepal_length_cm`
  (r ≈ 0.87 and 0.82 respectively). A downstream model would likely lose
  little accuracy by dropping one of the two petal features.

- **`sepal_width_cm` behaves differently from the other three features** —
  it's the only one that is *weakly negatively* correlated with the rest,
  and it's the only column with any IQR-flagged outliers (2 mild ones).
  It's the least useful single feature for separating species.

- **`setosa` is linearly separable from the other two species** on nearly
  any single feature or pair of features. `versicolor` and `virginica`
  overlap more, especially on sepal measurements, and would need petal
  measurements (or a non-linear boundary) to separate reliably.

- **No aggressive outlier removal is warranted.** Only 2 mild outliers
  were found (in `sepal_width_cm`) out of 151 rows using the standard
  1.5×IQR rule — they look like natural variation, not data errors.

- **Recommendation for any future modeling task:** use `petal_length_cm`
  and `petal_width_cm` as the primary features (they carry most of the
  separating signal), treat `sepal_width_cm` as a secondary/weak feature,
  and apply median imputation for any missing `petal_width_cm` values
  consistent with what was done here.
