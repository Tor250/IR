import json
import re
import shutil
from collections import Counter, defaultdict

from nltk.stem import PorterStemmer
from pyroaring import BitMap

from hw1.lsm_v3 import LSMTree

stemmer = PorterStemmer()
STOP_WORDS = {
    "the", "a", "and", "is", "in", "on", "of", "to", "for", "with", "by",
    "и", "в", "во", "на", "с", "со", "к", "ко", "по", "из", "у", "за", "под", "о", "об",
}
_WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё_-]+")


def _normalize_token(token: str) -> str:
    lowered = token.lower()
    if lowered.isascii() and any("a" <= ch <= "z" for ch in lowered):
        return stemmer.stem(lowered)
    return lowered


def tokenize(text: str, remove_stopwords: bool = False):
    if not text or not isinstance(text, str):
        return []
    tokens = []
    for raw in _WORD_RE.findall(text):
        normalized = _normalize_token(raw)
        if remove_stopwords and normalized in STOP_WORDS:
            continue
        tokens.append(normalized)
    return tokens


class HW5PositionalIndex:
    def __init__(
        self,
        use_lsm: bool = True,
        lsm_path: str = "data/hw5_positional_lsm",
        clear_on_init: bool = False,
        lsm_mem_limit: int = 256,
        lsm_auto_flush_docs: int = 200,
    ):
        self.docs = {}
        self.doc_tokens = {}
        self.index = defaultdict(lambda: defaultdict(list))
        self.term_docs = defaultdict(BitMap)
        self.term_frequencies = defaultdict(Counter)
        self.doc_count = 0
        self.use_lsm = use_lsm
        self.lsm_path = lsm_path
        self.lsm_auto_flush_docs = lsm_auto_flush_docs
        self._dirty_terms = set()
        self._dirty_docs = set()
        self._docs_since_flush = 0
        self.lsm = None

        if self.use_lsm:
            if clear_on_init:
                shutil.rmtree(self.lsm_path, ignore_errors=True)
            self.lsm = LSMTree(path=self.lsm_path, mem_limit=lsm_mem_limit)
            self._load_from_lsm()

    @staticmethod
    def _term_key(term: str):
        return "t:" + term

    @staticmethod
    def _doc_key(doc_id: int):
        return "__doc__:" + str(doc_id)

    @staticmethod
    def _meta_key():
        return "__meta__:doc_count"

    def _serialize_positions_map(self, postings):
        encoded = {}
        for doc_id, positions in postings.items():
            encoded[str(doc_id)] = list(positions)
        return json.dumps(encoded, separators=(",", ":")).encode("utf-8")

    def _deserialize_positions_map(self, blob):
        if blob is None:
            return {}
        try:
            raw = json.loads(blob.decode("utf-8"))
        except Exception:
            return {}

        decoded = {}
        for doc_id_raw, positions_raw in raw.items():
            try:
                doc_id = int(doc_id_raw)
            except Exception:
                continue

            if not isinstance(positions_raw, list):
                continue
            positions = []
            for item in positions_raw:
                if isinstance(item, int):
                    positions.append(item)
            decoded[doc_id] = positions
        return decoded

    def _load_from_lsm(self):
        if self.lsm is None:
            return

        self.docs.clear()
        self.doc_tokens.clear()
        self.index.clear()
        self.term_docs.clear()
        self.term_frequencies.clear()
        self.doc_count = 0

        items = self.lsm.range("", "\uffff")
        max_doc_id = -1
        saved_doc_count = None

        for key, value in items:
            if value is None:
                continue
            if key == self._meta_key():
                try:
                    saved_doc_count = int(value.decode("utf-8"))
                except Exception:
                    saved_doc_count = None
                continue
            if key.startswith("__doc__:"):
                doc_id_raw = key.split(":", 1)[1]
                try:
                    doc_id = int(doc_id_raw)
                except Exception:
                    continue
                try:
                    text = value.decode("utf-8")
                except Exception:
                    text = ""
                self.docs[doc_id] = text
                self.doc_tokens[doc_id] = tokenize(text, remove_stopwords=False)
                if doc_id > max_doc_id:
                    max_doc_id = doc_id
                continue
            if key.startswith("t:"):
                term = key[2:]
                postings = self._deserialize_positions_map(value)
                if not postings:
                    continue
                for doc_id, positions in postings.items():
                    self.index[term][doc_id] = positions
                    self.term_docs[term].add(doc_id)
                    self.term_frequencies[term][doc_id] = len(positions)
                    if doc_id > max_doc_id:
                        max_doc_id = doc_id

        if saved_doc_count is not None:
            self.doc_count = saved_doc_count
        else:
            if max_doc_id >= 0:
                self.doc_count = max_doc_id + 1

    def flush(self):
        if not self.use_lsm:
            return
        if self.lsm is None:
            return

        for doc_id in sorted(self._dirty_docs):
            text = self.docs.get(doc_id, "")
            self.lsm.put(self._doc_key(doc_id), text.encode("utf-8"))

        for term in sorted(self._dirty_terms):
            postings = self.index.get(term, {})
            blob = self._serialize_positions_map(postings)
            self.lsm.put(self._term_key(term), blob)

        self.lsm.put(self._meta_key(), str(self.doc_count).encode("utf-8"))
        self._dirty_terms.clear()
        self._dirty_docs.clear()
        self._docs_since_flush = 0

    def close(self):
        if self.use_lsm and self.lsm is not None:
            self.flush()
            self.lsm.close()

    def add_document(self, text: str):
        doc_id = self.doc_count
        self.doc_count += 1
        tokens = tokenize(text, remove_stopwords=False)

        self.docs[doc_id] = text
        self.doc_tokens[doc_id] = tokens

        for position, term in enumerate(tokens):
            self.index[term][doc_id].append(position)
            self.term_docs[term].add(doc_id)
            self.term_frequencies[term][doc_id] += 1

            if self.use_lsm:
                self._dirty_terms.add(term)

        if self.use_lsm:
            self._dirty_docs.add(doc_id)
            self._docs_since_flush += 1
            if self._docs_since_flush >= self.lsm_auto_flush_docs:
                self.flush()
        return doc_id

    def _normalize_query_terms(self, query):
        if isinstance(query, str):
            return tokenize(query, remove_stopwords=False)
        if not query:
            return []
        return [_normalize_token(term) for term in query if term]

    def get_positions(self, term: str, doc_id: int):
        normalized = _normalize_token(term)
        return list(self.index.get(normalized, {}).get(doc_id, []))

    def query_term(self, term: str):
        normalized = _normalize_token(term)
        return self.term_docs.get(normalized, BitMap()).copy()

    def query_and(self, *terms):
        if not terms:
            return BitMap()
        normalized_terms = [_normalize_token(term) for term in terms if term]
        if not normalized_terms:
            return BitMap()

        result = self.term_docs.get(normalized_terms[0], BitMap()).copy()
        for term in normalized_terms[1:]:
            result &= self.term_docs.get(term, BitMap())
            if not result:
                break
        return result

    def _phrase_end_positions(self, doc_id: int, terms):
        if not terms:
            return []
        if len(terms) == 1:
            return list(self.index.get(terms[0], {}).get(doc_id, []))

        current_positions = list(self.index.get(terms[0], {}).get(doc_id, []))
        if not current_positions:
            return []

        for term in terms[1:]:
            next_positions = list(self.index.get(term, {}).get(doc_id, []))
            if not next_positions:
                return []

            aligned = []
            left = 0
            right = 0
            while left < len(current_positions) and right < len(next_positions):
                expected = current_positions[left] + 1
                candidate = next_positions[right]
                if candidate == expected:
                    aligned.append(candidate)
                    left += 1
                    right += 1
                elif candidate < expected:
                    right += 1
                else:
                    left += 1
            if not aligned:
                return []
            current_positions = aligned
        return current_positions

    def phrase_positions(self, phrase: str):
        terms = self._normalize_query_terms(phrase)
        if not terms:
            return {}

        candidate_docs = self.query_and(*terms)
        start_shift = len(terms) - 1
        result = {}
        for doc_id in candidate_docs:
            end_positions = self._phrase_end_positions(doc_id, terms)
            if end_positions:
                result[doc_id] = [pos - start_shift for pos in end_positions]
        return result

    def search_phrase(self, phrase: str):
        matches = BitMap()
        for doc_id in sorted(self.phrase_positions(phrase)):
            matches.add(doc_id)
        return matches

    def count_phrase_occurrences(self, phrase: str):
        return {doc_id: len(positions) for doc_id, positions in self.phrase_positions(phrase).items()}
