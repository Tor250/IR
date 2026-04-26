.PHONY: install test benchmark test-v1 test-v2 test-v3 benchmark-v1 benchmark-v2 benchmark-v3 clean

clean-data:
	rm -rf data/lsm_index data/lsm_index_v3

HW4_VERSION ?= v3

test: clean-data
	HW4_VERSION=$(HW4_VERSION) PYTHONPATH=.. python3 -m pytest -v test_hw4_index.py

benchmark: clean-data
	HW4_VERSION=$(HW4_VERSION) PYTHONPATH=.. python3 benchmark_hw4_index.py

test-v1:
	$(MAKE) test HW4_VERSION=v1

test-v2:
	$(MAKE) test HW4_VERSION=v2

test-v3:
	$(MAKE) test HW4_VERSION=v3

benchmark-v1:
	$(MAKE) benchmark HW4_VERSION=v1

benchmark-v2:
	$(MAKE) benchmark HW4_VERSION=v2

benchmark-v3:
	$(MAKE) benchmark HW4_VERSION=v3

clean:
	rm -rf lsm_index __pycache__
