.PHONY: install run clean

install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt

run:
	@echo "Starting app on PORT=$${PORT:-5000}..."
	PORT=$${PORT:-5000} python -m grounded_theory_agent.app

clean:
	@echo "Cleaning generated outputs..."
	rm -rf generated/*
