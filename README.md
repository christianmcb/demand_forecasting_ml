# Demand Forecasting ML System

Production-style machine learning pipeline for retail demand forecasting using time-series feature engineering and gradient boosting models.

This project demonstrates an end-to-end ML workflow including feature engineering, model training, experiment tracking, batch inference, API serving, and containerised deployment.

---

# Project Overview

This system predicts **daily retail store sales** using historical demand patterns, calendar features, and business metadata.

The project was designed to demonstrate **production-ready machine learning engineering practices**, including:

- Modular ML pipeline
- Time-series feature engineering
- Experiment tracking with MLflow
- Model evaluation and comparison
- Batch prediction workflows
- REST API model serving with FastAPI
- Dockerised deployment
- Automated testing
- CI integration

---

# System Architecture

```
raw data
   ↓
data preprocessing
   ↓
feature engineering
   ↓
model training
   ↓
experiment tracking (MLflow)
   ↓
model evaluation
   ↓
saved model artifact
   ↓
batch prediction / API serving
```

---

# Dataset

This project uses the **Rossmann Store Sales dataset**, a real-world retail forecasting dataset.

The objective is to predict daily store sales using:

- promotional activity
- school holidays
- store type
- assortment category
- competition proximity
- historical sales patterns

---

# Feature Engineering

The forecasting system uses standard time-series tabular modelling techniques.

### Calendar Features

- month
- day
- week_of_year
- day_of_week
- is_weekend
- is_month_start
- is_month_end

### Lag Features

Historical demand signals:

- lag_1
- lag_7
- lag_14
- lag_28

### Rolling Features

Short-term demand trends:

- rolling_mean_7
- rolling_mean_14
- rolling_std_7

All rolling statistics are computed using **shifted historical data to prevent leakage**.

---

# Model

The model used is **LightGBM**, a gradient boosting framework well suited to tabular data.

The pipeline includes:

- chronological train / validation / test split
- baseline comparison using lag features
- feature importance analysis
- experiment tracking with MLflow

---

# Evaluation

Models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Example results:

| Model | MAE | RMSE |
|------|------|------|
| Baseline (lag_7) | ... | ... |
| LightGBM | ... | ... |

---

# Running the Project

## Train model

```
make train
```

---

## Evaluate model

```
make evaluate
```

---

## Run MLflow experiment tracker

```
make mlflow-ui
```

---

## Run batch forecasting

```
make batch-predict
```

Outputs are written to:

```
data/predictions/
```

---

# API Inference

Start the FastAPI service:

```
make run-api
```

Example request:

```
POST /predict
```

Example payload:

```json
{
  "Store": 1,
  "Promo": 1,
  "SchoolHoliday": 0,
  "StateHoliday": "0",
  "StoreType": "a",
  "Assortment": "a",
  "CompetitionDistance": 500.0,
  "Promo2": 1,
  "month": 5,
  "day": 12,
  "week_of_year": 20,
  "day_of_week": 2,
  "is_weekend": 0,
  "is_month_start": 0,
  "is_month_end": 0,
  "lag_1": 5200,
  "lag_7": 5300,
  "lag_14": 5100,
  "lag_28": 5050,
  "rolling_mean_7": 5150,
  "rolling_mean_14": 5200,
  "rolling_std_7": 180
}
```

---

# Docker Deployment

Build container

```
make docker-build
```

Run container

```
make docker-run
```

---

# Project Structure

```
src/
    data_loader.py
    preprocessing.py
    feature_engineering.py
    train_model.py
    evaluate_model.py
    predict.py
    config.py

scripts/
    train.py
    evaluate.py
    batch_predict.py

api/
    main.py

tests/

models/

data/
    raw/
    processed/
    predictions/
```

---

# Experiment Tracking

Experiments are tracked using **MLflow**, logging:

- model parameters
- training metrics
- evaluation metrics
- feature importance artifacts
- trained model artifacts

Run locally with:

```
mlflow ui
```

---

# Technologies

Python  
LightGBM  
MLflow  
FastAPI  
Docker  
Pandas / NumPy  
Scikit-learn  

---

# Author

Christian McBride