all:
	cat Makefile

test:
	pytest tests -v

benchmark:
	pytest tests/test_benchmark.py -v -s

install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev]"

clean:
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
