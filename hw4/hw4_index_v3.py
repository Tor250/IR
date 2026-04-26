import re
import shutil
from datetime import datetime

from pyroaring import BitMap
from nltk.stem import PorterStemmer

from hw1.lsm_v3 import LSMTree

stemmer = PorterStemmer()
STOP_WORDS = {"the", "a", "and", "is", "in", "on", "of", "to", "for", "with", "by"}
_WORD_RE = re.compile(r"[A-Za-z0-9_-]+")


def tokenize(text: str):
    if not text or not isinstance(text, str):
        return []
    words = _WORD_RE.findall(text.lower())
    words = [w for w in words if w not in STOP_WORDS]
    return [stemmer.stem(w) for w in words]


class HW4InvertedIndex:
    DATE_BITS = datetime.max.toordinal().bit_length()

    def __init__(self, lsm_path="data/lsm_index_v3", clear_on_init=False):
        if clear_on_init:
            shutil.rmtree(lsm_path, ignore_errors=True)

        self.lsm_path = lsm_path
        self.lsm = LSMTree(path=lsm_path)
        self.docs = {}
        self.doc_dates = {}
        self.doc_count = 0
        self._all_docs = BitMap()
        self._start_slices = [BitMap() for _ in range(self.DATE_BITS)]
        self._end_slices = [BitMap() for _ in range(self.DATE_BITS)]

    @staticmethod
    def _parse_datetime(value, field_name):
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(value)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid {field_name}: {value}")

    def _add_to_bit_slices(self, slices, ordinal, doc_id):
        for bit in range(self.DATE_BITS):
            if ordinal & (1 << bit):
                slices[bit].add(doc_id)

    def _bitmap_lt(self, slices, ordinal):
        lt = BitMap()
        eq = self._all_docs.copy()

        for bit in range(self.DATE_BITS - 1, -1, -1):
            bit_set = slices[bit]
            if ordinal & (1 << bit):
                lt |= eq - bit_set
                eq &= bit_set
            else:
                eq -= bit_set
            if not eq:
                break
        return lt

    def _bitmap_le(self, slices, ordinal):
        lt = BitMap()
        eq = self._all_docs.copy()

        for bit in range(self.DATE_BITS - 1, -1, -1):
            bit_set = slices[bit]
            if ordinal & (1 << bit):
                lt |= eq - bit_set
                eq &= bit_set
            else:
                eq -= bit_set
            if not eq:
                break
        return lt | eq

    def _bitmap_ge(self, slices, ordinal):
        return self._all_docs - self._bitmap_lt(slices, ordinal)

    def _bitmap_between(self, slices, start_ordinal, end_ordinal):
        if start_ordinal > end_ordinal:
            return BitMap()
        return self._bitmap_ge(slices, start_ordinal) & self._bitmap_le(slices, end_ordinal)

    def add_document(self, text: str, start_date: str, end_date: str = None):
        doc_id = self.doc_count
        self.doc_count += 1
        self.docs[doc_id] = text
        self._all_docs.add(doc_id)

        start_dt = self._parse_datetime(start_date, "start_date")
        end_dt = self._parse_datetime(end_date, "end_date")
        if start_dt is None:
            raise ValueError(f"Invalid start_date: {start_date}")

        start_ord = start_dt.toordinal()
        end_ord = end_dt.toordinal() if end_dt is not None else datetime.max.toordinal()

        self.doc_dates[doc_id] = (start_dt, end_dt)
        self._add_to_bit_slices(self._start_slices, start_ord, doc_id)
        self._add_to_bit_slices(self._end_slices, end_ord, doc_id)

        tokens = tokenize(text)
        for t in tokens:
            current = self._get_term_bitmap(t)
            current.add(doc_id)
            self.lsm.put(t, current.serialize())
        return doc_id

    def _get_term_bitmap(self, term: str) -> BitMap:
        stemmed = stemmer.stem(term)
        val = self.lsm.get(stemmed)
        if val is None:
            return BitMap()
        try:
            return BitMap.deserialize(val)
        except Exception:
            return BitMap()

    def query_and(self, *terms):
        if not terms:
            return BitMap()
        result = self._get_term_bitmap(terms[0]).copy()
        for t in terms[1:]:
            result &= self._get_term_bitmap(t)
        return result

    def query_or(self, *terms):
        result = BitMap()
        for t in terms:
            result |= self._get_term_bitmap(t)
        return result

    def query_and_not(self, include_term: str, exclude_term: str):
        return self._get_term_bitmap(include_term) - self._get_term_bitmap(exclude_term)

    def _filter_by_date(self, doc_ids: BitMap, start: datetime, end: datetime, mode: str) -> BitMap:
        start_ord = start.toordinal()
        end_ord = end.toordinal()

        if mode == "appeared":
            candidates = self._bitmap_between(self._start_slices, start_ord, end_ord)
        elif mode == "valid":
            starts_before_query_end = self._bitmap_le(self._start_slices, end_ord)
            ends_after_query_start = self._bitmap_ge(self._end_slices, start_ord)
            candidates = starts_before_query_end & ends_after_query_start
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return doc_ids & candidates

    def query_and_date(self, terms=None, start: str = None, end: str = None, mode: str = "valid"):
        dt_start = self._parse_datetime(start, "start") if start else None
        dt_end = self._parse_datetime(end, "end") if end else None

        if terms:
            doc_ids = self.query_and(*terms)
        else:
            doc_ids = self._all_docs.copy()

        if dt_start is not None or dt_end is not None:
            effective_start = dt_start or datetime.min
            effective_end = dt_end or datetime.max
            return self._filter_by_date(doc_ids, effective_start, effective_end, mode)
        return doc_ids

    def query_complex(self, expr, start=None, end=None, mode="valid"):
        dt_start = self._parse_datetime(start, "start") if start else None
        dt_end = self._parse_datetime(end, "end") if end else None
        result = self._eval_expr(expr)
        if dt_start or dt_end:
            effective_start = dt_start or datetime.min
            effective_end = dt_end or datetime.max
            result = self._filter_by_date(result, effective_start, effective_end, mode)
        return result

    def query_boolean(self, expr, start=None, end=None, mode="valid"):
        return self.query_complex(expr, start=start, end=end, mode=mode)

    def _eval_expr(self, expr):
        if isinstance(expr, str):
            return self._get_term_bitmap(expr).copy()
        if not isinstance(expr, tuple) or len(expr) == 0:
            return BitMap()

        op = expr[0]

        if op == "TERM":
            return self._get_term_bitmap(expr[1]).copy()
        if op == "AND":
            if len(expr) < 2:
                return BitMap(range(self.doc_count))
            result = self._eval_expr(expr[1]).copy()
            for subexpr in expr[2:]:
                result &= self._eval_expr(subexpr)
            return result
        if op == "OR":
            result = BitMap()
            for subexpr in expr[1:]:
                result |= self._eval_expr(subexpr)
            return result
        if op == "NOT":
            if len(expr) != 2:
                raise ValueError("NOT requires exactly one argument")
            return self._all_docs - self._eval_expr(expr[1])
        if op == "AND_NOT":
            if len(expr) != 3:
                raise ValueError("AND_NOT requires two arguments")
            return self._eval_expr(expr[1]) - self._eval_expr(expr[2])
        if op == "DATE":
            if len(expr) != 4:
                raise ValueError("DATE requires start, end, mode")
            _, start_str, end_str, mode = expr
            dt_start = self._parse_datetime(start_str, "start") if start_str else None
            dt_end = self._parse_datetime(end_str, "end") if end_str else None
            effective_start = dt_start or datetime.min
            effective_end = dt_end or datetime.max
            return self._filter_by_date(self._all_docs.copy(), effective_start, effective_end, mode)
        if op == "AND_DATE":
            if len(expr) < 5:
                raise ValueError("AND_DATE requires term_expr, start, end, mode")
            term_expr = expr[1]
            start_str, end_str, mode = expr[2], expr[3], expr[4]
            if isinstance(term_expr, str):
                doc_ids = self._get_term_bitmap(term_expr).copy()
            else:
                doc_ids = self._eval_expr(term_expr)
            dt_start = self._parse_datetime(start_str, "start") if start_str else None
            dt_end = self._parse_datetime(end_str, "end") if end_str else None
            effective_start = dt_start or datetime.min
            effective_end = dt_end or datetime.max
            return self._filter_by_date(doc_ids, effective_start, effective_end, mode)

        raise ValueError(f"Unknown operator: {op}")

    def get_document_count(self):
        return self.doc_count

    def get_term_frequency(self, term: str):
        return len(self._get_term_bitmap(term))

    def close(self):
        if hasattr(self.lsm, "close"):
            self.lsm.close()

    def cleanup(self):
        shutil.rmtree(self.lsm_path, ignore_errors=True)
