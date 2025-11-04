# pm25 — PM2.5 Forecasting 

A collection of notebooks and utilities for exploring, modeling, and visualizing **PM2.5 (fine particulate matter)**. The project focuses on time-series forecasting and sequence-to-sequence modeling, with dedicated notebooks for **data exploration**, **modeling**, and **evaluation**.

---

## ✨ Highlights

* End-to-end workflow: data prep → exploration → modeling → evaluation
* Sequence-to-sequence modeling experiments (see `models_seq2seq/`)
* Comprehensive baseline & SOTA coverage: **StatsForecast**, **MLForecast**, **NeuralForecast**, **Prophet**, **NeuralProphet**
* Notebook-driven exploration and forecasting pipeline
* Reproducible environment (`requirements.txt`)

---

## 📁 Repository structure

```
pm25/
├─ code/                # Notebooks / scripts for data prep, modeling, evaluation
├─ data/                # Your datasets (raw/processed); ignored by git as needed
├─ maps/                # Data exploration notebooks and map visualizations
├─ models_seq2seq/      # Seq2seq experiments, checkpoints, helpers
├─ papers/              # Related papers/notes for context and citation
├─ output.txt           # Example/temporary run output
├─ statsforecast_wawrning.txt  # Notes from the StatsForecast package
└─ requirements.txt     # Reproducible Python environment
```

> Note: `maps/` is used for **data exploration** and visual analysis, not for final model results. Model results are typically presented as **plots in notebooks** under `code/` or `models_seq2seq/`.

---

## 🚀 Quickstart

### 1) Clone

```bash
git clone https://github.com/makataomu/pm25.git
cd pm25
```

### 2) Create an environment

```bash
# with uv (recommended)
uv venv
source .venv/bin/activate

# or with conda
conda create -n pm25 python=3.10 -y
conda activate pm25
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Open the notebooks

Use Jupyter Lab/Notebook to explore:

```bash
jupyter lab
```

Start with `maps/` notebooks for **data exploration**, then move to `code/` and `models_seq2seq/` for **model development** and **evaluation**.

---

## 🧰 Data

This project does **not** include large datasets. Typical inputs include:

* Time-stamped PM2.5 observations from monitoring stations or open datasets
* Meteorological covariates (temperature, humidity, wind, etc.)
* Optional satellite or reanalysis features

Place your files in the `data/` folder (you can add subfolders like `raw/` and `processed/`). Update notebook paths if your layout differs.

> **Privacy & licensing:** Ensure you have the right to use and redistribute any data you add.

---

## 🧠 Modeling (What we used)

The project explores a wide range of forecasting models, from classical to deep learning approaches, implemented across **StatsForecast**, **MLForecast**, and **NeuralForecast**, plus **Prophet** and **NeuralProphet** baselines.

### Classical / Statistical

* **ARIMA, SARIMA, ETS, AutoARIMA, AutoETS** (via *StatsForecast*)
* Additional StatsForecast families where applicable (automatic seasonality & frequency handling)

### Machine Learning (via *MLForecast*)

* **RandomForestRegressor**, **XGBRegressor**
* **Linear baselines**: Ridge, Lasso, **SGDRegressor** (with optional Optuna tuning)
* Rich **lag features**, **date features**, and **lag transforms** (e.g., `rolling_mean`, `expanding_mean`)

### Deep Learning

* **Seq2Seq (LSTM/GRU encoder–decoder)** — notebooks in `models_seq2seq/`
* **Temporal Convolutional Networks (TCN)** — causal convolutions for long receptive fields
* **Transformer-based** variants (as extensions)

### Prophet & NeuralProphet baselines

* **Prophet** variants: base, +additional seasonality (≈30.5‑day), +holidays (e.g., Kazakhstan), and combined
* **NeuralProphet**: additive components with optional auto-regressive terms

All results are visualized **inside notebooks** as plots (not in separate artifacts).

---

## 🧩 MLForecastPipeline (overview)

A reusable pipeline (see associated file/notebook) that:

* **Formats data** to `MLForecast` schema (`unique_id`, `ds`, `y`) and supports **multi-series**.
* **Selects lags dynamically**:

  * Computes a base `max_lags` from train length and explores fractions (¼, ½, full) with upper bounds.
  * Uses a **RandomForest feature-importance** pass to select **top‑K lags**; supports multiple K values and names each lag set.
* **Applies target transforms** via `mlforecast.target_transforms`:

  * `AutoDifferences`, `AutoSeasonalDifferences`, `AutoSeasonalityAndDifferences`
  * Scaling: `LocalStandardScaler`, `LocalMinMaxScaler`, `LocalBoxCox`
  * Safeguards against conflicting transform combos.
* **Adds lag transforms** per-lag (e.g., `rolling_mean`, `expanding_mean`).
* **Fits a grid of models** (RF, XGB, Ridge, Lasso, SGD, etc.).
* **Optionally reweights seasons** (e.g., higher weights in winter months) during training.
* **Evaluates** with progressive horizons (30→180 days, then 240, 300, …, full test) using **MAPE**, plus optional **per‑month** breakdowns.
* **Persists results** to CSV and returns tidy DataFrames for downstream analysis.

> There is also a **multi‑series evaluator** that predicts and scores each `unique_id` separately, and an **SGD tuning** routine (Optuna) when enabled.

---

## 🔧 TimeSeriesPreprocessor (imputation & transforms)

Utility class for robust preprocessing of univariate series before/after modeling:

* **Stationarity checks** (KPSS), **trend** removal (diff or decomposition)
* **Seasonality handling** (STL‑style decomposition or seasonal differencing)
* **Stabilization**: Box‑Cox (auto‑lambda), log; **standardization**
* **`fit_transform` / `inverse_transform_predictions`** to round‑trip predictions back to original scale
* **`create_pipeline`** to generate multiple transformation combinations for ablation

This module plays nicely with Prophet/MLForecast by transforming `y` during training and **inverting** predictions for fair evaluation.

---

## 🗺️ Data Exploration

`maps/` contains notebooks for **data exploration**, including:

* Visualizing spatial and temporal PM2.5 distributions
* Mapping pollutant levels or monitoring station coverage
* Inspecting correlations between variables

These notebooks help understand spatial patterns before moving to modeling.

---

## 🧭 Suggested workflow

1. **EDA** in `maps/`  →  2) **Feature prep & lag selection** via *MLForecastPipeline*  →  3) **Modeling** with (StatsForecast / MLForecast / Prophet / NeuralProphet / Seq2Seq)  →  4) **Evaluation & plots in notebooks**  →  5) **Compare CSV results**.

---

## 📈 Future Improvements

* Publish a small anonymized sample dataset and a **quickstart notebook**
* Add **GitHub Actions** to execute smoke tests on notebooks
* Export **interactive dashboards** for metric comparison

---

## FAQ

**Q: Where are the datasets?**
A: Add them under `data/`; large files should be excluded from Git (consider Git LFS).

**Q: Which notebooks should I start with?**
A: Begin with `maps/` for exploratory data analysis, then proceed to `code/` and `models_seq2seq/` for forecasting and evaluation.
