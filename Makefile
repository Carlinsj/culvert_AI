.PHONY: setup setup-dev demo web-data serve-web test clean-demo

setup:
	python3 -m venv .venv
	.venv/bin/pip install -e .

setup-dev:
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev]"

demo:
	.venv/bin/culvert-ai run-demo --output-dir data/ulster_demo

web-data:
	.venv/bin/culvert-ai export-web --predictions data/ulster_demo/processed/unlabeled_predictions.gpkg --output-dir web/data

serve-web:
	scripts/serve_web.sh

test:
	.venv/bin/python -m pytest tests

clean-demo:
	rm -rf data/ulster_demo
