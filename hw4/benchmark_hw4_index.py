import time
import random
import importlib
import os


def _load_index_class():
    version = os.getenv("HW4_VERSION", "v3").strip().lower()
    module_name = {
        "v1": "hw4.hw4_index_v1",
        "v2": "hw4.hw4_index_v2",
        "v3": "hw4.hw4_index_v3",
    }.get(version)
    if module_name is None:
        raise ValueError(f"Unknown HW4_VERSION: {version}")
    module = importlib.import_module(module_name)
    return module.HW4InvertedIndex


HW4InvertedIndex = _load_index_class()

def benchmark_inserts(N=10000):
    try:
        idx = HW4InvertedIndex(clear_on_init=True)
    except TypeError:
        idx = HW4InvertedIndex()
    start = time.time()
    for i in range(N):
        doc = f"doc{i} " + " ".join(f"word{random.randint(0,1000)}" for _ in range(20))
        start_date = f"2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
        end_date = None
        if random.random() < 0.3:
            end_date = f"2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
        idx.add_document(doc, start_date, end_date)
    print(f"Inserting {N} docs took {time.time()-start:.2f}s")
    return idx

def benchmark_queries(idx, N=1000):
    start = time.time()
    for _ in range(N):
        term = f"word{random.randint(0,1000)}"
        idx.query_and(term)
    print(f"{N} random term queries took {time.time()-start:.2f}s")

    start = time.time()
    for _ in range(N):
        start_date = f"2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
        end_date = f"2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
        idx.query_and_date(terms=None, start=start_date, end=end_date)
    print(f"{N} random date queries took {time.time()-start:.2f}s")

    start = time.time()
    for _ in range(N):
        term = f"word{random.randint(0,1000)}"
        start_date = f"2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
        end_date = f"2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
        idx.query_and_date(terms=[term], start=start_date, end=end_date)
    print(f"{N} random term+date queries took {time.time()-start:.2f}s")

if __name__ == "__main__":
    idx = benchmark_inserts()
    benchmark_queries(idx)
