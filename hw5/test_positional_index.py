import random
import shutil

from pyroaring import BitMap

from hw5.positional_index import HW5PositionalIndex, tokenize


def _cleanup_lsm(path):
    shutil.rmtree(path, ignore_errors=True)


def _make_index():
    return HW5PositionalIndex(
        use_lsm=True,
        lsm_path="data/hw5_test_lsm",
        clear_on_init=True,
        lsm_auto_flush_docs=50,
    )


def _naive_phrase_positions(tokens, phrase_terms):
    if not phrase_terms:
        return []
    window = len(phrase_terms)
    matches = []
    for start in range(len(tokens) - window + 1):
        if tokens[start:start + window] == phrase_terms:
            matches.append(start)
    return matches


def _naive_phrase_search(docs, phrase):
    phrase_terms = tokenize(phrase, remove_stopwords=False)
    positions = {}
    for doc_id, text in enumerate(docs):
        doc_tokens = tokenize(text, remove_stopwords=False)
        starts = _naive_phrase_positions(doc_tokens, phrase_terms)
        if starts:
            positions[doc_id] = starts
    bitmap = BitMap(positions.keys())
    counts = {doc_id: len(starts) for doc_id, starts in positions.items()}
    return bitmap, positions, counts


def test_basic_phrase_search():
    idx = _make_index()
    idx.add_document("new york city is busy")
    idx.add_document("new big york city")
    idx.add_document("york new city")

    assert idx.search_phrase("new york") == BitMap([0])
    assert idx.search_phrase("york city") == BitMap([0, 1])
    assert idx.phrase_positions("new york") == {0: [0]}


def test_phrase_without_post_filtering():
    idx = _make_index()
    idx.add_document("cat sat on mat")
    idx.add_document("cat on sat mat")
    idx.add_document("cat sat sat mat")

    assert idx.search_phrase("cat sat") == BitMap([0, 2])
    assert idx.search_phrase("sat mat") == BitMap([1, 2])
    assert idx.search_phrase("cat mat") == BitMap()


def test_repeated_term_phrase():
    idx = _make_index()
    idx.add_document("go go go stop")
    idx.add_document("go stop go")

    assert idx.search_phrase("go go") == BitMap([0])
    assert idx.phrase_positions("go go") == {0: [0, 1]}
    assert idx.count_phrase_occurrences("go go") == {0: 2}


def test_alternating_repeated_phrase_positions():
    idx = _make_index()
    idx.add_document("alpha beta alpha beta alpha")
    idx.add_document("alpha alpha beta alpha")
    idx.add_document("beta alpha beta")

    assert idx.search_phrase("alpha beta alpha") == BitMap([0, 1])
    assert idx.phrase_positions("alpha beta alpha") == {0: [0, 2], 1: [1]}
    assert idx.count_phrase_occurrences("alpha beta alpha") == {0: 2, 1: 1}


def test_stopwords_are_preserved_inside_phrase():
    idx = _make_index()
    idx.add_document("the cat in the hat")
    idx.add_document("cat hat")

    assert idx.search_phrase("in the") == BitMap([0])
    assert idx.search_phrase("the hat") == BitMap([0])
    assert idx.search_phrase("cat in the") == BitMap([0])


def test_single_term_phrase_equals_term_query():
    idx = _make_index()
    idx.add_document("alpha beta gamma")
    idx.add_document("beta beta alpha")
    idx.add_document("delta")

    assert idx.search_phrase("beta") == idx.query_term("beta")
    assert idx.phrase_positions("beta") == {0: [1], 1: [0, 1]}


def test_stemming_and_positions():
    idx = _make_index()
    idx.add_document("running fast and runs fast")
    idx.add_document("runner fast")

    assert idx.search_phrase("run fast") == BitMap([0])
    assert idx.phrase_positions("run fast") == {0: [0, 3]}
    assert idx.query_term("running") == BitMap([0])
    assert idx.query_term("runs") == BitMap([0])
    assert idx.query_term("runner") == BitMap([1])


def test_incremental_build_consistency():
    idx = _make_index()
    docs = [
        "alpha beta gamma",
        "beta alpha gamma",
        "alpha beta alpha beta",
        "gamma alpha beta",
        "alpha gamma beta",
    ]
    expected_after_each_insert = [
        BitMap([0]),
        BitMap([0]),
        BitMap([0, 2]),
        BitMap([0, 2, 3]),
        BitMap([0, 2, 3]),
    ]

    for doc, expected in zip(docs, expected_after_each_insert):
        idx.add_document(doc)
        assert idx.search_phrase("alpha beta") == expected


def test_phrase_matches_naive_reference_on_random_queries():
    rng = random.Random(42)
    idx = _make_index()
    docs = []
    vocab = [f"word{i}" for i in range(18)] + ["alpha", "beta", "gamma", "delta"]

    for doc_id in range(220):
        tokens = [rng.choice(vocab) for _ in range(30)]
        if doc_id % 19 == 0:
            insert_at = rng.randint(0, len(tokens) - 3)
            tokens[insert_at:insert_at + 3] = ["alpha", "beta", "gamma"]
        if doc_id % 23 == 0:
            insert_at = rng.randint(0, len(tokens) - 4)
            tokens[insert_at:insert_at + 4] = ["beta", "gamma", "delta", "alpha"]
        text = " ".join(tokens)
        docs.append(text)
        idx.add_document(text)

    queries = [
        "alpha beta gamma",
        "beta gamma delta",
        "missing term sequence",
    ]

    for _ in range(60):
        doc_tokens = tokenize(rng.choice(docs), remove_stopwords=False)
        start = rng.randint(0, len(doc_tokens) - 4)
        width = rng.randint(1, 4)
        queries.append(" ".join(doc_tokens[start:start + width]))

    for query in queries:
        expected_bitmap, expected_positions, expected_counts = _naive_phrase_search(docs, query)
        assert idx.search_phrase(query) == expected_bitmap, f"Bitmap mismatch for query: {query}"
        assert idx.phrase_positions(query) == expected_positions, f"Positions mismatch for query: {query}"
        assert idx.count_phrase_occurrences(query) == expected_counts, f"Counts mismatch for query: {query}"


def test_large_phrase_corpus():
    idx = _make_index()
    rng = random.Random(123)
    docs = []

    for doc_id in range(500):
        tokens = [f"word{rng.randint(0, 60)}" for _ in range(45)]
        if doc_id % 17 == 0:
            insert_at = rng.randint(0, len(tokens) - 3)
            tokens[insert_at:insert_at + 3] = ["alpha", "beta", "gamma"]
        docs.append(" ".join(tokens))
        idx.add_document(docs[-1])

    expected_bitmap, expected_positions, expected_counts = _naive_phrase_search(docs, "alpha beta gamma")
    assert idx.search_phrase("alpha beta gamma") == expected_bitmap
    assert idx.phrase_positions("alpha beta gamma") == expected_positions
    assert idx.count_phrase_occurrences("alpha beta gamma") == expected_counts


def test_lsm_persistence_roundtrip():
    lsm_path = "data/hw5_persistence_lsm"
    _cleanup_lsm(lsm_path)

    idx = HW5PositionalIndex(
        use_lsm=True,
        lsm_path=lsm_path,
        clear_on_init=True,
        lsm_auto_flush_docs=1000,
    )
    idx.add_document("new york city and new york state")
    idx.add_document("alpha beta alpha beta alpha")
    idx.add_document("running fast and runs fast")
    idx.close()

    reopened = HW5PositionalIndex(
        use_lsm=True,
        lsm_path=lsm_path,
        clear_on_init=False,
    )

    assert reopened.search_phrase("new york") == BitMap([0])
    assert reopened.phrase_positions("alpha beta alpha") == {1: [0, 2]}
    assert reopened.search_phrase("run fast") == BitMap([2])
    reopened.close()
    _cleanup_lsm(lsm_path)
