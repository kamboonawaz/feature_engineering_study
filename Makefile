# Simple workflow

PY=python

.PHONY: setup data baseline engineered evaluate all clean

setup:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) src/data_download.py

baseline:
	$(PY) src/train.py --config configs/baseline.yaml

engineered:
	$(PY) src/train.py --config configs/engineered.yaml

evaluate:
	$(PY) src/evaluate.py

all: data baseline engineered evaluate

clean:
	rm -f models/*.joblib reports/metrics.json
