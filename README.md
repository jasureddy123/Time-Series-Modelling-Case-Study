# Electricity Demand Forecasting — Notebook

A single, self-contained Jupyter notebook that reproduces the whole case study:
data **download** and preparation, exploratory analysis and stationarity
testing, benchmark forecasts, SARIMA, SARIMAX + temperature, a gradient-boosting
feature model and an **LSTM** (with hyperparameter tuning and a genuine two-year
forecast), followed by a horizon-aware comparison.

## Files

```
Electricity_Demand_Forecasting.ipynb   the full analysis (run top to bottom)
Report-2.docx                            written report answering the 6 questions
requirements.txt                       pinned Python dependencies
data/opsd_de_load_hourly.csv           cached German hourly load (MW)
data/berlin_temperature_daily.csv      cached Berlin daily temperature (°C)
data/sarima_grid.csv                   cached AIC grid-search results (147 models)
data/lstm_tuning.csv                   cached LSTM hyperparameter sweep
data/lstm.pt                           pre-trained LSTM weights
```

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.10. `holidays` is optional — the calendar features degrade
gracefully if it is not installed.

## Running it

Open the notebook and **Run All**. It executes in well under a minute because the
heavy steps reuse the cached artefacts in `data/`. Every cached step now also has
a **missing-cache fallback**: if a cache file is absent the cell recomputes it
from scratch and re-writes the cache, so the notebook always runs end-to-end even
on a fresh checkout. Each step can also be forced to recompute with a flag:

| Step | Flag | Default behaviour |
|------|------|-------------------|
| Download the official 130 MB OPSD file | automatic if cache missing | uses cache |
| Download Berlin temperature (Open-Meteo) | automatic if cache missing | uses cache |
| SARIMA AIC grid search (147 models, minutes) | `RUN_FULL_GRID` | loads / rebuilds `sarima_grid.csv` |
| LSTM hyperparameter sweep | `RUN_LSTM_TUNING` | loads / rebuilds `lstm_tuning.csv` |
| Train the LSTM (~1 min) | `TRAIN_LSTM` | loads / rebuilds `lstm.pt` |

Benchmarks, SARIMA, SARIMAX and gradient boosting are always fitted live.

## What the notebook covers

* **Part 1** downloads the official OPSD data and aggregates to weekly/daily GW.
* **Part 2** tests stationarity systematically across `d∈{0,1} × D∈{0,1}`
  (including the seasonal-only difference) with ADF + KPSS.
* **Part 4** selects SARIMA by AIC over the full `p,d,q` grid and shows the top
  models; residual diagnostics and 95% intervals included.
* **Part 5** adds Open-Meteo temperature as an exogenous (conditional) regressor
  and plots the conditional forecast with a 95% interval.
* **Part 6** builds leakage-safe features for a tuned Gradient Boosting model.
* **Part 7** tunes an LSTM, produces a **genuine recursive two-year forecast**
  (no future actuals) *and* a clearly-labelled one-hour diagnostic.
* **Part 8** compares every model with a **horizon** column so short- and
  long-horizon forecasts are never confused, then discusses the six required
  questions.

## A note on the LSTM configuration

The tuning sweep runs on a short, illustrative budget (2 epochs on sub-sampled
windows), so its ranking is indicative only and favours a larger two-layer
network on one-step validation error. The **deployed** recursive forecaster is a
compact **single-layer LSTM (48 hidden units, lookback 168 h)**, chosen for
stability over a long recursive rollout and cheaper retraining; this is the
architecture saved in `data/lstm.pt`.

## Key result

At the genuine two-year horizon **no model beats the seasonal-naive benchmark**
(RMSE 3.01 GW). SARIMA (3.93), SARIMAX + temperature (3.78), the recursive
gradient-boosting model (3.70) and especially the **recursive LSTM (6.11)** all
trail it, because the 2019–2020 demand decline (efficiency + COVID-19) lies
outside the training period. Models only excel when they may condition on recent
data — a horizon effect, not a sign of model quality.
