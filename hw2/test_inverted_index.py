import shutil
import random
from pyroaring import BitMap
from hw2.inverted_index_v2 import InMemoryInvertedIndex, LSMInvertedIndex

def cleanup_lsm():
    shutil.rmtree("lsm_index", ignore_errors=True)

def test_basic_inmemory():
    idx = InMemoryInvertedIndex()
    idx.add_document("cat dog mouse")
    idx.add_document("cat elephant")
    idx.add_document("dog tiger")
    assert idx.query_and("cat", "dog") == BitMap([0])
    assert idx.query_or("elephant", "tiger") == BitMap([1,2])
    print("InMemory basic test OK")

def test_basic_lsm():
    cleanup_lsm()
    idx = LSMInvertedIndex(mem_limit=5)
    docs = ["cat dog mouse","cat elephant","dog tiger","elephant tiger cat","mouse tiger"]
    for d in docs:
        idx.add_document(d)
    idx._flush()
    idx.close()
    assert idx.query_and("cat","dog") == BitMap([0])
    assert idx.query_or("elephant","tiger") == BitMap([1,2,3,4])
    cleanup_lsm()
    print("LSM basic test OK")

def test_stress_lsm():
    cleanup_lsm()
    N = 5000
    idx = LSMInvertedIndex(mem_limit=100)
    data = {}
    for i in range(N):
        doc = f"doc{i} " + " ".join(f"word{random.randint(0,1000)}" for _ in range(20))
        data[i] = doc
        idx.add_document(doc)
    idx._flush()
    idx.close()

    sample = random.sample(range(N), 50)
    for i in sample:
        bm = idx.query_or(f"doc{i}")
        assert i in bm

    for _ in range(20):
        i, j = random.sample(range(N), 2)
        words_i = data[i].split()[1:]
        words_j = data[j].split()[1:]
        if words_i and words_j:
            bm = idx.query_and(words_i[0], words_j[0])
            assert isinstance(bm, BitMap)
    cleanup_lsm()
    print("LSM stress test OK")

def test_query_not_operator():
    idx = InMemoryInvertedIndex()
    idx.add_document("python tutorial")
    idx.add_document("python advanced")
    idx.add_document("java tutorial")
    result = idx.query_and_not("python", "tutorial")
    assert result == BitMap([1])

def test_nonexistent_term():
    idx = InMemoryInvertedIndex()
    idx.add_document("cat dog")
    
    result = idx.query_and("unicorn")
    assert len(result) == 0

def test_double_stemming_consistency():
    idx = InMemoryInvertedIndex()
    idx.add_document("running runs runner")
    assert 0 in idx.query_and("running")
    assert 0 in idx.query_and("run")
    assert 0 in idx.query_and("runs")

def test_lsm_flush_merge_correctness():
    cleanup_lsm()
    idx = LSMInvertedIndex(mem_limit=10, lsm_path="lsm_index")
    for batch in range(5):
        for i in range(10):
            doc_id = batch * 10 + i
            idx.add_document(f"common word_{doc_id}")
    
    idx.close()
    result = idx.query_and("common")
    assert len(result) == 50, f"Ожидалось 50 документов с 'common', найдено {len(result)}"
    result = idx.query_and("word_25")
    assert len(result) == 1 and 25 in result, "Уникальный терм найден не в том документе"
    
    cleanup_lsm()
    print("✓ test_lsm_flush_merge_correctness passed")


def test_stemming_collision_handling():
    cleanup_lsm()
    idx = LSMInvertedIndex(mem_limit=20, lsm_path="lsm_index")
    idx.add_document("running fast")
    idx.add_document("runs quickly")
    idx.add_document("run won")
    idx.add_document("jogging slow")
    
    idx.close()
    for term in ["running", "runs", "run"]:
        result = idx.query_and(term)
        assert 0 in result and 1 in result and 2 in result, f"Стем-коллизия для '{term}': {list(result)}"
        assert 3 not in result, f"Ложное срабатывание для '{term}'"
    result = idx.query_and("jogging")
    assert 3 in result and 0 not in result, "Некорректное разделение стемов"
    
    cleanup_lsm()
    print("✓ test_stemming_collision_handling passed")


def test_large_bitmap_operations():
    cleanup_lsm()
    idx = LSMInvertedIndex(mem_limit=100, lsm_path="lsm_index")
    
    N = 2000
    for i in range(N):
        tokens = []
        if i % 2 == 0:
            tokens.append("frequent")
        if i % 100 == 0:
            tokens.append("rare")
        tokens.append(f"unique_{i}")
        idx.add_document(" ".join(tokens))
    
    idx.close()
    result = idx.query_and("frequent", "rare")
    expected_count = N // 100
    assert abs(len(result) - expected_count) <= 2, f"Неверное количество для AND: {len(result)} vs {expected_count}"
    result = idx.query_or("rare", "unique_500")
    assert 500 in result, "Уникальный терм пропал в OR"
    assert len(result) >= 20, "OR вернул слишком мало документов"
    for i in [0, 100, 500, 1999]:
        result = idx.query_and(f"unique_{i}")
        assert list(result) == [i], f"Уникальный терм unique_{i} не изолирован"
    
    cleanup_lsm()
    print("✓ test_large_bitmap_operations passed")


def test_persistence_across_restarts():
    cleanup_lsm()
    idx1 = LSMInvertedIndex(mem_limit=50, lsm_path="lsm_index")
    docs = [
        "python programming tutorial",
        "java programming guide", 
        "python advanced course",
        "javascript basics"
    ]
    for i, doc in enumerate(docs):
        idx1.add_document(doc)
    idx1.close()
    idx2 = LSMInvertedIndex(mem_limit=50, lsm_path="lsm_index")
    assert 0 in idx2.query_and("python") and 2 in idx2.query_and("python")
    assert 1 in idx2.query_and("java")
    assert 0 in idx2.query_and("python", "tutorial")
    assert 1 not in idx2.query_and("python")
    cleanup_lsm()
    print("✓ test_persistence_across_restarts passed")


def test_concurrent_term_updates():
    cleanup_lsm()
    idx = LSMInvertedIndex(mem_limit=5, lsm_path="lsm_index")
    for wave in range(10):
        for i in range(5):
            doc_id = wave * 5 + i
            idx.add_document(f"shared wave_{wave}_doc_{i}")
    
    idx.close()
    result = idx.query_and("shared")
    assert len(result) == 50, f"Потеряны документы при обновлении терма: {len(result)}/50"
    for i in range(50):
        assert i in result, f"Документ {i} пропал из постинг-листа"
    
    cleanup_lsm()
    print("✓ test_concurrent_term_updates passed")


def test_stopwords_and_empty_tokens():
    idx = InMemoryInvertedIndex()
    idx.add_document("the a and is in")
    idx.add_document("python the programming and")
    idx.add_document("")
    idx.add_document("!!! @@@ ###")
    assert 1 in idx.query_and("python")
    assert 1 in idx.query_and("programming")
    result = idx.query_and("the")
    assert len(result) == 0, "Стоп-слово попало в индекс"
    
    print("✓ test_stopwords_and_empty_tokens passed")



if __name__ == "__main__":
    test_basic_inmemory()
    test_double_stemming_consistency()