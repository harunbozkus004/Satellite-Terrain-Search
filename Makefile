PY := .venv/bin/python
CLI := $(PY) -m satterrain.cli

.PHONY: setup data train evaluate index serve test all

setup:
	python3 -m venv .venv
	$(PY) -m pip install -r requirements.txt

data:
	$(CLI) download-eurosat
	$(CLI) download-oscd
	$(CLI) fetch --locations 3000 --dates 5 --workers 40
	$(CLI) harmonize

train:
	$(CLI) train --epochs 100

evaluate:
	$(CLI) evaluate
	$(CLI) evaluate-change

index:
	$(CLI) build-index

serve:
	$(CLI) serve --port 8000

test:
	$(PY) -m pytest -q tests

all: data train evaluate index test
