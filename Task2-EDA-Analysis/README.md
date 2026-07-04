# Exploratory Data Analysis (EDA) Project

A complete EDA workflow — structure inspection, cleaning, summary stats,
visualization, correlation/outlier analysis, and an automated profiling
report — applied to the Iris dataset.

## Project files

| File                          | Purpose                                                        |
|--------------------------------|------------------------------------------------------------------|
| `EDA_Analysis.ipynb`           | Main deliverable notebook — all EDA steps, already executed with outputs/plots visible |
| `data/iris.csv`                 | The dataset analyzed (see below for how it was built)          |
| `generate_sample_data.py`      | Reproducibly builds `data/iris.csv` from sklearn's bundled Iris dataset, with a few injected data-quality issues |
| `build_profile_report.py`      | Builds `report.html` — uses real `ydata-profiling` if installed, otherwise a lightweight fallback profiler |
| `execute_notebook_inplace.py`  | Utility used to execute the notebook and embed outputs (only needed if nbclient/jupyter isn't installed) |
| `report.html`                   | Automated HTML profiling report (open in any browser)          |
| `insights_summary.md`          | 8 bullet-point key findings                                     |
| `figures/`                      | Standalone PNG copies of every chart, for use outside the notebook |

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib seaborn ydata-profiling jupyter
```

`ydata-profiling` is optional but recommended — it's the tool named in the
project spec and produces a much richer interactive report (per-column
warnings, interactions, sample rows, etc.) than the fallback profiler
included here. Everything else works without it.

## Usage

1. **Build the dataset** (reproducible, no internet required — uses
   `sklearn.datasets.load_iris`, which ships with scikit-learn):

   ```bash
   python generate_sample_data.py
   ```

2. **Open and run the notebook:**

   ```bash
   jupyter notebook EDA_Analysis.ipynb
   ```

   The notebook is already executed and includes all outputs and charts,
   so you can also just read it directly (e.g. on GitHub) without running
   anything.

3. **Regenerate the profiling report** (also runnable as its own script,
   outside the notebook):

   ```bash
   python build_profile_report.py --csv data/iris.csv --out report.html
   ```

   Open `report.html` in any browser.

## Why Iris instead of Titanic?

The project spec suggests Titanic, Iris, or any CSV from Kaggle/UCI. This
build uses **Iris** because it ships directly with scikit-learn
(`sklearn.datasets.load_iris`) and needs no external download — making the
whole project runnable offline and byte-for-byte reproducible. To use
Titanic instead, drop a `titanic.csv` into `data/` and point
`EDA_Analysis.ipynb` / `build_profile_report.py` at it; every step
(missing values, dtypes, outliers, correlations, profiling) generalizes
directly, though you'll want to swap the species-based groupby/boxplot
cells for something Titanic-specific (e.g. `survived` by `pclass`/`sex`).

## Data-quality issues (intentionally injected, for a realistic exercise)

`generate_sample_data.py` starts from the clean, canonical Iris data and
seeds in:
- 6 missing values (~4% of `petal_width_cm`)
- 1 exact duplicate row

This gives the "cleaning" section of the notebook something concrete to
detect and fix, rather than a dataset that's already spotless. The
injection is seeded (`np.random.default_rng(42)`) so re-running the script
always produces the identical CSV.

## Notebook sections

1. Load & inspect structure (shape, dtypes)
2. Data cleaning (missing values, duplicates)
3. Summary statistics (`describe()`, class balance)
4. Univariate distributions (histograms + KDE)
5. Relationships between features (boxplots + pairplot, by species)
6. Correlation analysis (heatmap)
7. Outlier detection (IQR rule)
8. Automated profiling report generation
9. Key insights (see also `insights_summary.md`)

## Troubleshooting

- **`ModuleNotFoundError: ydata_profiling`**: expected if you skipped that
  optional install — `build_profile_report.py` automatically falls back to
  a built-in lightweight HTML profiler, no action needed unless you want
  the full ydata-profiling experience.
- **Notebook cells show no output**: some Jupyter setups clear outputs on
  save; re-run all cells (`Kernel > Restart & Run All`) if that happens.
