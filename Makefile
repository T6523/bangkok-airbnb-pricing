PY ?= .venv/bin/python

.PHONY: setup data sql split tune evaluate bootstrap explain errors all

setup:
	python3 -m venv .venv
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m scripts.download_data

sql:
	$(PY) -m scripts.run_sql sql/01_load.sql sql/02_clean.sql sql/03_features.sql

split:
	$(PY) -m src.airbnb_pricing.split

# Optuna, capped at 20 minutes per model
tune:
	for m in lgbm xgb cat; do $(PY) -m src.airbnb_pricing.tune $$m || exit 1; done

evaluate:
	$(PY) -m src.airbnb_pricing.evaluate

bootstrap:
	$(PY) -m src.airbnb_pricing.bootstrap

explain:
	$(PY) -m src.airbnb_pricing.explain

errors:
	$(PY) -m src.airbnb_pricing.error_analysis

all: data sql split tune evaluate bootstrap explain errors
