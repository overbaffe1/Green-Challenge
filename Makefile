PYTHON ?= .venv/bin/python
BASE_IMAGE ?= aij-green-challenge.cr.cloud.ru/green-vla:latest

.PHONY: test smoke docker-build report help

help:
	@echo "make test          run the CPU test suite"
	@echo "make smoke         two-hop pipeline with the scripted policy"
	@echo "make docker-build  build the submission image (BASE_IMAGE=...)"
	@echo "make report        summarise a results.json (RESULTS=path)"

venv:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install numpy msgpack websockets pytest

test:
	$(PYTHON) -m pytest tests -q

smoke:
	$(PYTHON) tools/smoke_test.py

docker-build:
	docker build -f docker/Dockerfile --build-arg BASE_IMAGE="$(BASE_IMAGE)" -t green-challenge:local .

report:
	$(PYTHON) eval/report.py $(RESULTS) --sweep
