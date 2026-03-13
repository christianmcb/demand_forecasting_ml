.PHONY: help train evaluate batch-predict run-api mlflow-ui test drift docker-build docker-run clean

PYTHON := python
MODULE := -m
PORT := 8000

help:
	@echo "Available commands:"
	@echo ""
	@echo "make train          Train forecasting model"
	@echo "make evaluate       Evaluate model and print metrics"
	@echo "make batch-predict  Run batch forecasting"
	@echo "make run-api        Start FastAPI inference service"
	@echo "make mlflow-ui      Launch MLflow experiment dashboard"
	@echo "make test           Run unit tests"
	@echo "make drift          Run data drift check"
	@echo "make docker-build   Build Docker image"
	@echo "make docker-run     Run Docker container"
	@echo "make clean          Remove temporary files"
	@echo ""

train:
	$(PYTHON) $(MODULE) scripts.train

evaluate:
	$(PYTHON) $(MODULE) scripts.evaluate

batch-predict:
	$(PYTHON) $(MODULE) scripts.batch_predict

run-api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port $(PORT)

mlflow-ui:
	mlflow ui

test:
	pytest

drift:
	$(PYTHON) $(MODULE) scripts.check_data_drift

docker-build:
	docker build -t demand-forecasting .

docker-run:
	docker run -p $(PORT):$(PORT) demand-forecasting

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf mlruns
	find . -name "*.pyc" -delete