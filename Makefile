# Makefile for ML Project

# Python executable
PYTHON := python3

# Directory for virtual environment
VENV_DIR := venv

# Activate virtual environment
ACTIVATE := . $(VENV_DIR)/bin/activate

# Default target
.DEFAULT_GOAL := help

# Help target
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  setup      Set up the virtual environment and install dependencies"
	@echo "  run        Run the Flask application"
	@echo "  train      Train the model"
	@echo "  report     Generate the model comparison report"
	@echo "  clean      Clean up the virtual environment"
	@echo "  all        Run setup, train, report, and run"

# Set up the virtual environment and install dependencies
setup:
	@echo "Setting up virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)
	$(ACTIVATE)
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "Environment setup complete."

# Run the Flask application
run:
	@echo "Running the Flask application..."
	$(ACTIVATE)
	$(PYTHON) run.py run_app # Changed Here

# Train the model
train:
	@echo "Training the model..."
	$(ACTIVATE)
	$(PYTHON) run.py train

# Generate the model comparison report
report:
	@echo "Generating the model comparison report..."
	$(ACTIVATE)
	$(PYTHON) run.py report

# Clean up the virtual environment
clean:
	@echo "Cleaning up the virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "Environment cleaned."

# Run all
all: setup train report run
