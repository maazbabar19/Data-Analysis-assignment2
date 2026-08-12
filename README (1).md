# Forecasting Household Appliance Energy Use

Hourly forecasting of domestic appliance electricity consumption using the UCI
*Appliances Energy Prediction* dataset, comparing benchmark, statistical,
machine-learning and pretrained foundation models over a rolling-origin
backtest.

## Problem

Forecast `Appliances` (Wh per hour) **24 hours ahead**. The final 14 days are
held out and split into **14 consecutive 24-hour forecast origins**,
giving 336 evaluated hours. Every model is scored on the same origins
with the same metrics.

A single 24-point test set cannot separate 11 models: the
differences between them are smaller than the sampling error of a sample that
size. The rolling-origin design reports a spread as well as a point estimate.

## Data

| Property | Value |
|---|---|
| Source | UCI ML Repository, dataset 374 |
| Raw resolution | 10 minutes, 19,735 rows |
| Modelled resolution | hourly, 3,289 rows |
| Period | 11 Jan 2016 to 27 May 2016 |
| Target | `Appliances`, mean 586.1 Wh/h, max 3650 Wh/h |

Aggregation uses two rules. `Appliances` and `lights` record energy accumulated
within each ten-minute interval and are **summed**; every temperature, humidity
and weather column is an instantaneous reading and is **averaged**. Applying
`mean` uniformly would rescale the target by a factor of six.

`rv1` and `rv2` are random values injected by the dataset authors. They are not
used as predictors. They are kept as a noise floor on the feature-importance
chart, so that any feature ranking below pure randomness is visibly worthless.

## Models

| Family | Models |
|---|---|
| Benchmarks | mean, naive, seasonal naive at 24h and 168h, drift |
| Statistical | SARIMA, SARIMAX with Fourier terms, SARIMAX with Fourier terms and weather |
| Machine learning | XGBoost, operational and conditional variants |
| Foundation model | Chronos-Bolt, zero-shot |

The SARIMA order came from an exhaustive AIC search over p in [0,6], d in [0,2],
q in [0,6] (147 combinations), followed by a seasonal-order refinement.
Selected specification: **SARIMA(2, 0, 4)(1, 1, 1, 24)**.

A SARIMA model carries one seasonal period. The daily cycle is taken by the
seasonal component at s=24; the weekly cycle is supplied through
6 Fourier regressors at period 168, which are deterministic and
therefore known at any forecast origin.

## Results

Strongest benchmark: **Seasonal naive (168h)**, MASE 0.804.
Best model overall: **Chronos (zero-shot)**, MASE 0.630.
Best genuine (non-conditional) forecast: **Chronos (zero-shot)**,
MASE 0.630.

```
                                   model           family     MAE     RMSE  MASE     Bias
                     Chronos (zero-shot) Foundation model 201.542  405.301 0.630 -108.435
                       SARIMAX (Fourier)      Statistical 204.348  394.456 0.638  -80.600
SARIMAX (Fourier + weather, conditional)      Statistical 204.648  395.100 0.639  -79.181
                   XGBoost (operational) Machine learning 205.350  388.588 0.641  -66.335
                                  SARIMA      Statistical 208.007  398.133 0.650  -82.851
                   XGBoost (conditional) Machine learning 209.679  387.574 0.655  -58.320
                   Seasonal naive (168h)        Benchmark 257.530  478.010 0.804  -75.744
                    Seasonal naive (24h)        Benchmark 290.357  514.742 0.907   10.000
                                    Mean        Benchmark 300.061  443.916 0.937  -18.368
                                   Naive        Benchmark 860.298 1023.735 2.687  688.036
                                   Drift        Benchmark 863.652 1027.559 2.698  691.886
```

MASE is scaled by the in-sample 24-hour seasonal naive error
(320.15 Wh), so a value below 1 beats that benchmark.

## Avoiding data leakage

* No target lag shorter than the 24-hour horizon, and every rolling statistic
  shifted by 24 hours, so each row is computable at its forecast origin. This is
  the *direct* multi-horizon design; it avoids the error accumulation of
  recursive forecasting and is far easier to audit.
* Strictly chronological splitting. Hyperparameter search uses
  `TimeSeriesSplit`, never `KFold`, so no validation fold precedes its own
  training data.
* Indoor sensor and outdoor weather readings inside the forecast window are not
  known at the origin. Models using them are labelled **conditional** and
  reported separately from the operational variants; the gap between each pair
  quantifies what that information is worth.
* No model was selected on test-set performance. The SARIMA order comes from
  in-sample AIC and the XGBoost hyperparameters from time-series
  cross-validation inside the training period.
* Assertions in the notebook and `tests/test_features.py` enforce these
  properties, because a leakage bug produces excellent metrics and no error.

## Repository layout

```
appliance-energy-forecasting/
├── README.md
├── requirements.txt
├── src/appliance_energy/       reusable package
│   ├── config.py               paths, horizons and column groups
│   ├── evaluation.py           MAE, RMSE, MASE, Bias, sMAPE, coverage
│   ├── features.py             calendar, lag, rolling and Fourier features
│   └── models/benchmarks.py    the five benchmark forecasters
├── scripts/run_pipeline.py     end-to-end runnable pipeline
├── notebooks/                  the full study
├── outputs/                    figures, forecasts and metrics
└── tests/                      pytest suite
```

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_pipeline.py
PYTHONPATH=src pytest
```

The notebook in `notebooks/` carries the complete study including SARIMA,
XGBoost and the foundation model. It downloads the data itself and runs top to
bottom in roughly 12 minutes on CPU.

## References

1. Candanedo, L.M., Feldheim, V. and Deramaix, D. (2017) Data driven prediction
   models of energy use of appliances in a low-energy house. *Energy and
   Buildings*, 140, pp. 81-97.
2. Hyndman, R.J. and Athanasopoulos, G. (2021) *Forecasting: Principles and
   Practice*, 3rd edn. OTexts.
3. Hyndman, R.J. and Koehler, A.B. (2006) Another look at measures of forecast
   accuracy. *International Journal of Forecasting*, 22(4), pp. 679-688.
4. Chen, T. and Guestrin, C. (2016) XGBoost: a scalable tree boosting system.
   *Proceedings of KDD 2016*, pp. 785-794.
5. Ansari, A.F. et al. (2024) Chronos: learning the language of time series.
   arXiv:2403.07815.
6. Bergmeir, C. and Benitez, J.M. (2012) On the use of cross-validation for time
   series predictor evaluation. *Information Sciences*, 191, pp. 192-213.
