.PHONY: setup run clean all

# Default target
all: setup run

# Install dependencies
setup:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm
	python -m spacy download en_core_web_lg
	python -m spacy download en_core_web_trf

# Run the Streamlit app
run:
	streamlit run app.py

# Clean up cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".streamlit" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete