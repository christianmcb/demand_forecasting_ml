# Demand Forecasting ML System

A production-style machine learning project demonstrating how time-series models can be **trained, evaluated, tracked, and deployed for retail demand forecasting**.

The system predicts **daily retail store sales** using historical demand patterns, calendar features, and store metadata. The trained model can be used for **batch forecasting workflows or deployed as an API for real-time inference**.

---

## Core Components

- **Time-Series Data Processing**
  - Date parsing and temporal feature extraction (year, month, week, etc.)
  - Handling missing values and irregular time intervals
  - Lag and rolling window feature generation

- **Feature Engineering**
  - Creation of autoregressive features (lags, rolling means)
  - Calendar-based signals (seasonality, promotions, holidays)
  - Structured pipeline for reproducible transformations

- **Model Training & Evaluation**
  - Tree-based models (e.g. XGBoost, LightGBM) for forecasting
  - Time-aware validation (train/validation splits respecting temporal order)
  - Evaluation using appropriate forecasting metrics (e.g. RMSE, MAE)

- **Forecasting Strategy**
  - Supervised learning formulation of time-series problem
  - Multi-step / recursive forecasting approach

- **API Deployment**
  - FastAPI service for generating forecasts
  - Dockerised for reproducibility and portability

- **Project Structure**
  - Modular design separating data, features, models, and serving
  - Designed to resemble production ML systems

---

## Current Limitations

- No dedicated **time-series cross-validation framework** (e.g. backtesting loops)
- Limited **forecast horizon strategy optimisation** (recursive vs direct vs hybrid)
- No **probabilistic forecasting** (prediction intervals / uncertainty)
- No **model versioning or experiment tracking**
- No **data drift or concept drift monitoring**
- No automated **retraining pipeline**
- Feature engineering is not yet abstracted into a reusable pipeline

---

## Next Steps

- Implement **backtesting framework** for robust time-series validation
- Add **probabilistic forecasting** (e.g. quantile regression / prediction intervals)
- Compare with dedicated time-series models (e.g. ARIMA, Prophet, deep learning)
- Introduce **model versioning + experiment tracking**
- Build **automated retraining pipeline** (scheduled updates)
- Add **data & concept drift detection**
- Optimise **multi-step forecasting strategies**
- Deploy to cloud (AWS/GCP) with scalable inference

---

## Live API Demo

The trained model can be deployed as a **FastAPI inference service** and queried directly.

Interactive API documentation:

```
https://demandforecastingml-production.up.railway.app/docs
```

---

# Dataset

This project uses the **Rossmann Store Sales dataset**, a real-world retail forecasting dataset containing daily sales records for over 1,000 stores.

The objective is to predict **daily store sales** using historical demand signals and store-level metadata.

Dataset source:

https://www.kaggle.com/competitions/rossmann-store-sales

Download the dataset and store the files in a directory named `data/`.

Required files:

```
train.csv
test.csv
store.csv
sample_submission.csv
```

Reference:

> Rossmann Store Sales Forecasting Dataset. Kaggle.

Target variable:

```
Sales
```

Where:

```
Sales → daily revenue for a store
```

This dataset contains **over 1 million observations**, making it well suited for training robust tabular ML models.

---

## Example Results

Model performance using chronological train/validation/test splits to prevent time-series leakage.

| Model | Validation MAE | Test MAE | Notes |
|------|------|------|------|
| Lag Baseline | 2433.9 | 3063.0 | Uses lag_7 demand |
<!-- | RandomForest | 793.4 | 801.0 | Captures nonlinear demand | -->
| LightGBM | 561.4 | 805.4 | Best performance |

**Selected model for deployment:** LightGBM

---

# Project Structure

```
demand_forecasting_ml/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── predictions/
│
├── models/
│   ├── lightgbm_model.pkl
│   └── metrics.json
│
├── notebooks/
│   └── main.ipynb
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── batch_predict.py
│   └── check_data_drift.py
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
│   ├── metadata.py
│   ├── schema.py
│   └── logger.py
│
├── api/
│   └── main.py
│
├── tests/
│
├── configs/
│   └── model_config.yaml
│
├── Dockerfile
├── Makefile
├── pytest.ini
├── requirements.txt
└── README.md
```

---

# Quickstart

Clone the repository, install dependencies, and train the model.

```bash
git clone https://github.com/christianmcb/demand_forecasting_ml.git
cd demand_forecasting_ml

pip install -r requirements.txt

make train
make evaluate
```

Generate batch forecasts:

```bash
make batch-predict
```

Start the API service:

```bash
make run-api
```

API documentation:

```
http://localhost:8000/docs
```

---

# Training Pipeline

Training executes a full ML workflow:

1. Load dataset
2. Validate input schema
3. Preprocess raw data
4. Build time-series features
5. Train LightGBM model
6. Evaluate model performance
7. Log experiments with MLflow
8. Save model artifacts

Key feature engineering techniques include:

**Calendar Features**

```
month
day_of_week
week_of_year
is_weekend
is_month_start
is_month_end
```

**Lag Features**

```
lag_1
lag_7
lag_14
lag_28
```

**Rolling Statistics**

```
rolling_mean_7
rolling_mean_14
rolling_std_7
```

Rolling statistics are computed using **shifted historical windows to avoid data leakage**.

---

# Experiment Tracking

Training runs are tracked using **MLflow**, logging:

- model parameters
- evaluation metrics
- feature importance
- model artifacts

Launch the MLflow dashboard:

```bash
make mlflow-ui
```

The dashboard will be available at:

```
http://localhost:5000
```

---

# Batch Forecasting

Generate demand forecasts for the latest available store data.

```bash
make batch-predict
```

Predictions are saved to:

```
data/predictions/forecast.csv
```

This mirrors how many real-world forecasting systems produce **daily demand forecasts via scheduled batch jobs**.

---

# API

Start the FastAPI inference service:

```bash
make run-api
```

Interactive API documentation:

```
http://localhost:8000/docs
```

Example request:

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

Example response:

```json
{
  "predicted_sales": 5284.2
}
```

---

# Docker

Build and run the API inside a container.

```bash
make docker-build
make docker-run
```

The API will be available at:

```
http://localhost:8000/docs
```

---

# Typical Workflow

```
train → evaluate → forecast → serve
```

Commands:

```bash
make train
make evaluate
make batch-predict
make run-api
```

---

## Author

Christian McBride  
Manchester, UK

GitHub: https://github.com/christianmcb  
LinkedIn: https://linkedin.com/in/christianmcb8

---

## License

This project is licensed under the MIT License.