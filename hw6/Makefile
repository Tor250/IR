.PHONY: test benchmark clean

test:
	PYTHONPATH=.. python3 -m pytest -v test_ranked_index.py

benchmark:
	PYTHONPATH=.. python3 benchmark_ranked_index.py

clean:
	rm -rf __pycache__ .pytest_cache
