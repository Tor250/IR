import shutil
import random
import importlib
import os
from pyroaring import BitMap


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

def cleanup_test_data():
    shutil.rmtree("hw4_test_data", ignore_errors=True)


def make_index():
    try:
        return HW4InvertedIndex(clear_on_init=True)
    except TypeError:
        return HW4InvertedIndex()

HW4InvertedIndex.query_boolean = HW4InvertedIndex.query_complex


def test_basic_term_and_date():
    idx = make_index()
    idx.add_document("cat dog", "2023-01-01")
    idx.add_document("dog elephant", "2023-02-01")
    idx.add_document("cat elephant", "2023-03-01")

    assert idx.query_and("cat", "dog") == BitMap([0])
    assert idx.query_or("cat", "elephant") == BitMap([0, 1, 2])

    res = idx.query_and_date(terms=["cat"], start="2023-01-01", end="2023-01-31")
    assert list(res) == [0]

    res = idx.query_and_date(terms=None, start="2023-02-01", end="2023-02-28", mode="appeared")
    assert list(res) == [1]

    print("✓ test_basic_term_and_date passed")


def test_date_ranges_and_overlap():
    idx = make_index()
    idx.add_document("a", "2023-01-01", "2023-01-31")
    idx.add_document("b", "2023-01-15", "2023-02-15")
    idx.add_document("c", "2023-02-01")

    res = idx.query_and_date(terms=None, start="2023-01-20", end="2023-02-01", mode="valid")
    assert sorted(res) == [0, 1, 2]

    res = idx.query_and_date(terms=None, start="2023-01-20", end="2023-02-01", mode="appeared")
    assert list(res) == [2]

    res = idx.query_and_date(terms=None, start="2023-02-01", end=None, mode="valid")
    assert sorted(res) == [1, 2]

    print("✓ test_date_ranges_and_overlap passed")


def test_terms_and_date_combined():
    idx = make_index()
    idx.add_document("foo bar", "2023-01-01")
    idx.add_document("foo baz", "2023-02-01")
    idx.add_document("baz qux", "2023-03-01")

    res = idx.query_and_date(terms=["foo"], start="2023-02-01", end="2023-02-28", mode="appeared")
    assert list(res) == [1]

    res = idx.query_and_date(terms=["baz"], start="2023-01-15", end="2023-03-15", mode="appeared")
    assert sorted(res) == [1, 2]

    res = idx.query_and_date(terms=["foo", "bar"], start="2023-01-01", end="2023-12-31")
    assert list(res) == [0]

    print("✓ test_terms_and_date_combined passed")


def test_edge_cases():
    idx = make_index()
    idx.add_document("", "2023-01-01")
    idx.add_document("term", "2023-01-02", "2023-01-02")
    idx.add_document("term", "2023-01-03")

    res = idx.query_and_date(terms=None, start="2023-01-02", end="2023-01-03")
    assert sorted(res) == [0, 1, 2]

    res = idx.query_and_date(terms=["foo"], start="2023-01-01", end="2023-01-03")
    assert len(res) == 0

    res = idx.query_and_date(terms=["term"], start="2023-01-02", end="2023-01-02", mode="valid")
    assert list(res) == [1]

    print("✓ test_edge_cases passed")



def test_tokenization_and_stemming():
    idx = make_index()
    idx.add_document("Running runs runner", "2023-01-01")
    idx.add_document("Jogging slow", "2023-02-01")

    for term in ["running", "runs", "run"]:
        res = idx.query_and(term)
        assert 0 in res
        assert 1 not in res

    idx.add_document("the cat and dog", "2023-03-01")
    res = idx.query_and("the")
    assert len(res) == 0

    res = idx.query_and("cat", "dog")
    assert 2 in res

    print("✓ test_tokenization_and_stemming passed")


def test_composite_terms():
    idx = make_index()
    idx.add_document("common word_25 unique_term", "2023-01-01")
    assert 0 in idx.query_and("word_25")
    assert 0 in idx.query_and("unique_term")
    assert 0 not in idx.query_and("word")

    print("✓ test_composite_terms passed")



def test_complex_boolean_with_dates():
    idx = make_index()
    idx.add_document("python tutorial", "2023-01-01")
    idx.add_document("java tutorial", "2023-02-01")
    idx.add_document("python advanced", "2023-03-01")
    idx.add_document("javascript basics", "2023-01-15")

    res = idx.query_boolean(
        expr=("OR", ("TERM", "python"), ("TERM", "java")),
        start="2023-02-01", end="2023-02-28", mode="appeared"
    )
    assert list(res) == [1]

    res = idx.query_boolean(
        expr=("AND_NOT", ("TERM", "python"), ("TERM", "tutorial")),
        start="2023-01-01", end="2023-03-31", mode="appeared"
    )
    assert list(res) == [2]

    print("✓ test_complex_boolean_with_dates passed")


def test_nested_boolean_expressions():
    idx = make_index()
    idx.add_document("alpha charlie", "2023-01-01")
    idx.add_document("bravo charlie", "2023-02-01")
    idx.add_document("delta", "2023-03-01")
    idx.add_document("alpha delta", "2023-01-15")

    res = idx.query_complex(
        ("OR",
            ("AND",
                ("OR", ("TERM", "alpha"), ("TERM", "bravo")),
                ("TERM", "charlie")
            ),
            ("AND_DATE", "delta", "2023-01-01", "2023-01-31", "appeared")
        )
    )
    assert sorted(res) == [0, 1, 3], f"Expected [0,1,3], got {sorted(res)}"

    print("✓ test_nested_boolean_expressions passed")


def cleanup_lsm():
    shutil.rmtree("data/lsm_index", ignore_errors=True)
    shutil.rmtree("data/lsm_index_v3", ignore_errors=True)

def test_large_scale_with_dates():
    cleanup_lsm()
    idx = make_index()
    N = 5000

    for i in range(N):
        tokens = [f"word{random.randint(0, 500)}" for _ in range(20)]
        text = " ".join(tokens)

        start_month = random.randint(1, 12)
        start_day = random.randint(1, 28)
        start_date = f"2023-{start_month:02d}-{start_day:02d}"

        end_date = None
        if random.random() < 0.5:
            end_month = random.randint(start_month, 12)
            end_day = random.randint(1, 28)
            end_date = f"2023-{end_month:02d}-{end_day:02d}"

        idx.add_document(text, start_date, end_date)

    res = idx.query_and("word0")
    assert 20 <= len(res) <= 300, f"Неверное распределение: {len(res)}"

    res = idx.query_and_date(terms=None, start="2023-06-01", end="2023-06-30", mode="appeared")
    assert 200 <= len(res) <= 600, f"Неверное распределение по датам: {len(res)}"

    res = idx.query_and_date(terms=["word0"], start="2023-01-01", end="2023-12-31")
    assert len(res) <= len(idx.query_and("word0"))

    print("✓ test_large_scale_with_dates passed")


def test_persistence_across_operations():
    idx = make_index()

    for i in range(100):
        idx.add_document(f"doc{i} term_a term_b", f"2023-{(i%12)+1:02d}-01")
        if i % 10 == 0:
            res = idx.query_and_date(terms=["term_a"], start="2023-01-01", end="2023-06-30")
            assert all(doc_id <= i for doc_id in res)

    res = idx.query_and("term_a")
    assert len(res) == 100

    print("✓ test_persistence_across_operations passed")

def test_date_boundary_conditions():
    idx = make_index()
    idx.add_document("doc", "2023-01-15", "2023-01-15")
    
    res = idx.query_and_date(start="2023-01-15", end="2023-01-15", mode="valid")
    assert 0 in res
    
    res = idx.query_and_date(start="2023-01-14", end="2023-01-15", mode="valid")
    assert 0 in res

def test_invalid_date_format():
    idx = make_index()
    try:
        idx.add_document("doc", "invalid-date")
        assert False, "Должна быть ошибка"
    except ValueError:
        pass




if __name__ == "__main__":
    test_basic_term_and_date()
    test_date_ranges_and_overlap()
    test_terms_and_date_combined()
    test_edge_cases()
    test_tokenization_and_stemming()
    test_composite_terms()
    test_complex_boolean_with_dates()
    test_nested_boolean_expressions()
    test_large_scale_with_dates()
    test_persistence_across_operations()
    print("\nAll tests passed!")
