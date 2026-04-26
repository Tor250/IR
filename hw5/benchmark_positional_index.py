import random
import time

from hw5.positional_index import HW5PositionalIndex


def benchmark_build(doc_count=10000, doc_len=60):
    idx = HW5PositionalIndex(
        use_lsm=True,
        lsm_path="data/hw5_benchmark_lsm",
        clear_on_init=True,
        lsm_auto_flush_docs=200,
    )
    start = time.time()
    for doc_id in range(doc_count):
        tokens = [f"word{random.randint(0, 2000)}" for _ in range(doc_len)]
        if doc_id % 50 == 0:
            insert_at = random.randint(0, doc_len - 4)
            tokens[insert_at:insert_at + 4] = ["alpha", "beta", "gamma", "delta"]
        idx.add_document(" ".join(tokens))
    elapsed = time.time() - start
    print(f"Build {doc_count} docs took {elapsed:.2f}s")
    return idx


def benchmark_phrase_queries(idx, query_count=1000):
    queries = []
    for i in range(query_count):
        if i % 3 == 0:
            queries.append("alpha beta gamma delta")
        else:
            left = random.randint(0, 1997)
            queries.append(f"word{left} word{left + 1}")

    start = time.time()
    total_hits = 0
    for query in queries:
        total_hits += len(idx.search_phrase(query))
    elapsed = time.time() - start
    avg_ms = elapsed * 1000.0 / max(query_count, 1)
    print(f"{query_count} phrase queries took {elapsed:.2f}s ({avg_ms:.2f} ms/query), total hits={total_hits}")


if __name__ == "__main__":
    random.seed(42)
    index = benchmark_build()
    benchmark_phrase_queries(index)
    index.close()
