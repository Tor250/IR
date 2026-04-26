.PHONY: test benchmark clean

test:
	PYTHONPATH=.. python3 -m pytest -v test_positional_index.py

benchmark:
	PYTHONPATH=.. python3 benchmark_positional_index.py

clean:
	rm -rf __pycache__ .pytest_cache
