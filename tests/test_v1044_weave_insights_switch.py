"""10.44.0: phase 3 keeps the link and stores no bridge memory unless asked.

MNEMOS_WEAVE_STORE_INSIGHTS (default off) decides whether a weave insight
becomes a memory; MNEMOS_WEAVE_MAX_INSIGHTS caps how many one run may store
when it is on. Background: a newer weave model accepted 15 of 16 candidate
pairs, and one bridge memory per accepted pair grew to 9 % of one store and
68 % of another within three weeks.
"""
import importlib

import mnemos.consolidation.phases as phases
from mnemos.constants import FASTEMBED_DIMS as DIMS

from . import test_v1017_zero_llm as zero

LINK = "LINK_TYPE: reflects\nSTRENGTH: 0.8\nINSIGHT: both show the same habit"


def _fixture(tmp_path, monkeypatch):
    store, conn, mem_by_id, embeddings = zero.TestWeaveHygiene()._weave_fixture(tmp_path)
    monkeypatch.setattr(phases, "opus_chat", lambda *a, **k: LINK)
    monkeypatch.setattr(phases, "fastembed_embed",
                        lambda texts: [[0.001] * DIMS for _ in texts])
    return conn, mem_by_id, embeddings


def _bridges(conn):
    return conn.execute(
        "SELECT COUNT(*) FROM memories WHERE tags LIKE '%bridge%'").fetchone()[0]


def test_default_keeps_the_link_and_stores_no_memory(tmp_path, monkeypatch):
    conn, mem_by_id, embeddings = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        phases.nli, "p_entailment",
        lambda p, h: (_ for _ in ()).throw(AssertionError(
            "the novelty gate has nothing to judge when insights are off")))
    monkeypatch.setattr(phases.nli, "is_available", lambda: True)
    stats = phases.phase_weave(conn, embeddings, mem_by_id,
                               is_surge=False, execute=True)
    assert stats["links_created"] >= 1
    assert stats["insights_stored"] == 0
    assert stats["insights_skipped_disabled"] >= 1
    assert _bridges(conn) == 0


def test_switch_on_stores_the_insight(tmp_path, monkeypatch):
    conn, mem_by_id, embeddings = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(phases, "WEAVE_STORE_INSIGHTS", True)
    monkeypatch.setattr(phases.nli, "is_available", lambda: False)
    stats = phases.phase_weave(conn, embeddings, mem_by_id,
                               is_surge=False, execute=True)
    assert stats["insights_stored"] >= 1
    assert _bridges(conn) == stats["insights_stored"]


def test_cap_bounds_what_one_run_stores(tmp_path, monkeypatch):
    conn, mem_by_id, embeddings = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(phases, "WEAVE_STORE_INSIGHTS", True)
    monkeypatch.setattr(phases, "WEAVE_MAX_INSIGHTS", 0)
    monkeypatch.setattr(phases.nli, "is_available", lambda: False)
    stats = phases.phase_weave(conn, embeddings, mem_by_id,
                               is_surge=False, execute=True)
    assert stats["links_created"] >= 1
    assert stats["insights_stored"] == 0
    assert stats["insights_skipped_cap"] >= 1
    assert _bridges(conn) == 0


def test_env_parsing(monkeypatch):
    import mnemos.constants as constants
    try:
        monkeypatch.delenv("MNEMOS_WEAVE_STORE_INSIGHTS", raising=False)
        monkeypatch.delenv("MNEMOS_WEAVE_MAX_INSIGHTS", raising=False)
        c = importlib.reload(constants)
        assert c.WEAVE_STORE_INSIGHTS is False and c.WEAVE_MAX_INSIGHTS == 5
        monkeypatch.setenv("MNEMOS_WEAVE_STORE_INSIGHTS", "1")
        monkeypatch.setenv("MNEMOS_WEAVE_MAX_INSIGHTS", "2")
        c = importlib.reload(constants)
        assert c.WEAVE_STORE_INSIGHTS is True and c.WEAVE_MAX_INSIGHTS == 2
    finally:
        monkeypatch.undo()
        importlib.reload(constants)
