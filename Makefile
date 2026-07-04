# Reproduction package: reading the data back (model-data misspecification at repository scale).
#
# Every committed artifact reproduce.py reads is BUNDLED, so the paper's numbers
# reproduce offline with no API key and no network:
#
#   make setup       create .venv and install requirements (one time)
#   make reproduce   every paper number, from the committed artifacts
#   make corpus      OPTIONAL: rebuild the raw census pool (~20 GB) from public DOIs
#   make help        show this list
#
# PY points at the venv interpreter made by `make setup`. Override to reuse your
# own environment, e.g.  make reproduce PY=python3

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: help setup reproduce corpus

help:
	@echo "The artifacts are already bundled; reproduction needs no API key and no network."
	@echo "Targets:"
	@echo "  setup       create .venv and install requirements.txt (one time)"
	@echo "  reproduce   all paper numbers   <- the main one"
	@echo "  corpus      rebuild the raw ~20 GB census pool into data/dv (network; optional)"
	@echo "  help        show this message"

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "[setup] done. Just run: make reproduce  (no .env needed; artifacts are bundled)."

reproduce:
	$(PY) reproduce.py

corpus:
	@echo "Rebuilding the census pool from Harvard Dataverse (public DOIs; ~20 GB, hours)."
	@echo "An API key is optional (example.env). Ctrl-C and re-run to resume."
	$(PY) src/census_harvest.py
