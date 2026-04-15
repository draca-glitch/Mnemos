"""
Unit tests for pure functions in Mnemos.

No LLM, no network, no database. These exercise the algorithmic building
blocks (query cleaning, RRF merge, CML subject extraction, embedding-text
preparation, sigmoid) so regressions get caught before they reach a store.
"""

import math

import pytest

from mnemos.query import clean_fts_query, STOP_WORDS
from mnemos.rerank import rrf_merge
from mnemos.core import _sigmoid, _extract_cml_subject
from mnemos.embed import prep_memory_text


class TestCleanFtsQuery:
    def test_empty_query_returns_empty(self):
        assert clean_fts_query("") == ""
        assert clean_fts_query("   ") == ""

    def test_and_mode_joins_with_and(self):
        out = clean_fts_query("redis postgres mariadb", mode="AND")
        assert out == '"redis" AND "postgres" AND "mariadb"'

    def test_or_mode_joins_with_or(self):
        out = clean_fts_query("redis postgres", mode="OR")
        assert out == '"redis" OR "postgres"'

    def test_stop_words_are_removed(self):
        out = clean_fts_query("the redis is on the server", mode="AND")
        # "the", "is", "on" are stop words; "redis" and "server" survive
        assert '"redis"' in out
        assert '"server"' in out
        assert '"the"' not in out
        assert '"is"' not in out

    def test_single_char_tokens_dropped(self):
        # len >= 2 filter
        out = clean_fts_query("a bb c dd", mode="AND")
        # "a" and "c" dropped, "bb" and "dd" kept (not stop words)
        assert '"bb"' in out
        assert '"dd"' in out
        assert '"a"' not in out
        assert '"c"' not in out

    def test_case_is_lowered(self):
        out = clean_fts_query("REDIS Postgres", mode="AND")
        assert '"redis"' in out
        assert '"postgres"' in out

    def test_all_stopwords_returns_empty(self):
        out = clean_fts_query("the a an is of to")
        assert out == ""

    def test_stopwords_list_is_non_empty(self):
        assert len(STOP_WORDS) > 10
        assert "the" in STOP_WORDS


class TestRrfMerge:
    def test_empty_input_returns_empty(self):
        assert rrf_merge() == []
        assert rrf_merge([]) == []

    def test_single_list_preserves_order(self):
        assert rrf_merge([1, 2, 3]) == [1, 2, 3]

    def test_merge_gives_higher_rank_to_items_in_both(self):
        fts = [10, 20, 30]
        vec = [20, 40, 10]
        out = rrf_merge(fts, vec)
        # 20 is rank 2 in fts and rank 1 in vec, should beat 10 (rank 1 + rank 3) and 30 (only fts)
        assert out[0] == 20

    def test_unique_items_preserved(self):
        out = rrf_merge([1, 2, 3], [4, 5, 6])
        assert set(out) == {1, 2, 3, 4, 5, 6}

    def test_k_parameter_affects_ranking(self):
        # Higher k dampens rank-position weight, so ties tighten. Just verify
        # the function accepts the parameter and returns ranked ids.
        out = rrf_merge([1, 2], [2, 1], k=10)
        assert set(out) == {1, 2}


class TestSigmoid:
    def test_zero_is_half(self):
        assert _sigmoid(0) == pytest.approx(0.5)

    def test_large_positive_approaches_one(self):
        assert _sigmoid(10) > 0.999

    def test_large_negative_approaches_zero(self):
        assert _sigmoid(-10) < 0.001

    def test_monotonic_increasing(self):
        assert _sigmoid(-1) < _sigmoid(0) < _sigmoid(1) < _sigmoid(2)


class TestExtractCmlSubject:
    def test_decision_prefix(self):
        letter, subject = _extract_cml_subject("D:use redis for caching")
        assert letter == "D"
        assert subject == "use"

    def test_fact_prefix_with_id(self):
        letter, subject = _extract_cml_subject("F:Ethereum wallet 0x1234")
        assert letter == "F"
        assert subject == "Ethereum"

    def test_all_prefix_letters_recognized(self):
        for ch in "DCFLPW":
            letter, subject = _extract_cml_subject(f"{ch}:hello world")
            assert letter == ch
            assert subject == "hello"

    def test_non_cml_returns_none_none(self):
        assert _extract_cml_subject("plain prose, no prefix") == (None, None)
        assert _extract_cml_subject("") == (None, None)

    def test_stops_at_relation_symbol(self):
        # Subject should end at the first relation symbol or whitespace
        letter, subject = _extract_cml_subject("D:redis→fast")
        assert letter == "D"
        assert subject == "redis"

    def test_multiline_only_reads_first_line(self):
        letter, subject = _extract_cml_subject("D:decision\nsecond line")
        assert letter == "D"
        assert subject == "decision"

    def test_whitespace_stripped(self):
        letter, subject = _extract_cml_subject("D: spaced")
        assert letter == "D"
        assert subject == "spaced"


class TestPrepMemoryText:
    def test_minimal_just_project_and_content(self):
        out = prep_memory_text("dev", "F:redis on 6379")
        assert "dev" in out
        assert "redis" in out

    def test_tags_appended(self):
        out = prep_memory_text("dev", "content", tags="tag1,tag2")
        assert "tag1" in out
        assert "tag2" in out

    def test_non_default_type_included(self):
        out = prep_memory_text("dev", "content", mem_type="decision")
        assert "[decision]" in out

    def test_default_type_fact_not_included(self):
        # mem_type='fact' is the default, shouldn't be added as a tag
        out = prep_memory_text("dev", "content", mem_type="fact")
        assert "[fact]" not in out

    def test_non_default_layer_included(self):
        out = prep_memory_text("dev", "content", layer="episodic")
        assert "[episodic]" in out

    def test_default_layer_semantic_not_included(self):
        out = prep_memory_text("dev", "content", layer="semantic")
        assert "[semantic]" not in out

    def test_content_is_always_preserved(self):
        content = "F:verbatim content that must not be transformed"
        out = prep_memory_text("p", content, tags="t", mem_type="decision",
                               layer="episodic")
        assert content in out
