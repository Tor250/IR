import random
import time

from hw6.ranked_index import HW6RankedIndex


def benchmark_build(doc_count=12000, doc_len=70):
    idx = HW6RankedIndex(
        champion_size=16,
        tier_size=16,
        use_lsm=True,
        lsm_path="data/hw6_benchmark_lsm",
        clear_on_init=True,
        lsm_auto_flush_docs=200,
    )
    start = time.time()
    for doc_id in range(doc_count):
        tokens = [f"word{random.randint(0, 3000)}" for _ in range(doc_len)]
        if doc_id % 31 == 0:
            tokens.extend(["alpha", "beta", "gamma", "ranking"])
        if doc_id % 43 == 0:
            tokens.extend(["python", "search", "engine"])
        idx.add_document(" ".join(tokens))
    elapsed = time.time() - start
    print(f"Build {doc_count} docs took {elapsed:.2f}s")
    return idx


def benchmark_queries(idx, query_count=1500):
    queries = []
    for i in range(query_count):
        if i % 4 == 0:
            queries.append("alpha beta gamma ranking")
        elif i % 4 == 1:
            queries.append("python search engine")
        else:
            base = random.randint(0, 2997)
            queries.append(f"word{base} word{base + 1} word{base + 2}")

    start = time.time()
    for query in queries:
        idx.search_vector(query, top_k=10)
    exact_elapsed = time.time() - start

    start = time.time()
    for query in queries:
        idx.search_inexact_top_k(query, top_k=10)
    approx_elapsed = time.time() - start

    print(f"{query_count} exact cosine queries took {exact_elapsed:.2f}s ({exact_elapsed * 1000.0 / query_count:.2f} ms/query)")
    print(f"{query_count} inexact top-k queries took {approx_elapsed:.2f}s ({approx_elapsed * 1000.0 / query_count:.2f} ms/query)")
    if approx_elapsed > 0:
        print(f"Speedup: {exact_elapsed / approx_elapsed:.2f}x")


if __name__ == "__main__":
    random.seed(42)
    index = benchmark_build()
    benchmark_queries(index)
    index.close()
