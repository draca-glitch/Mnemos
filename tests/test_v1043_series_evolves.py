"""v10.43.0: dated-series evolution at store time.

A newer entry of the same dated series ("D:TMG SWEEP 2026-10-21 ..." after
"D:TMG SWEEP 2026-10-14 ...") gets an `evolves` link from the older entry
and the older entry gets valid_until=today, the same effect as a Nyx phase 4
EVOLVED verdict, without an LLM. The gate is the series key plus the dates;
the vector distance only keeps strangers out, because on the live store
consecutive sweeps sit in the same distance band as unrelated same-topic
pairs.
"""

from datetime import date

import mnemos.core as core_mod
from mnemos.core import Mnemos, _series_key
from mnemos.storage.sqlite_store import SQLiteStore
from mnemos.constants import FASTEMBED_DIMS as DIMS


def _store(tmp_path):
    return SQLiteStore(db_path=str(tmp_path / "m.db"), namespace="t")


def _mnemos(tmp_path):
    return Mnemos(store=_store(tmp_path), namespace="t",
                  enable_contradiction_detection=False, enable_rerank=False)


def _unit(axis):
    v = [0.0] * DIMS
    v[axis] = 1.0
    return v


def _embed_by_marker(texts, prefix="passage"):
    """Texts containing FAR get an orthogonal vector; the rest share one."""
    return [_unit(1) if "FAR" in t else _unit(0) for t in texts]


class TestSeriesKey:
    def test_subject_then_date(self):
        assert _series_key("D:TMG SWEEP 2026-10-14 23:10 → done") == ("D", "tmg sweep", "2026-10-14")

    def test_numbered_items_stay_distinct(self):
        a = _series_key("D:HIBANA D-16 (Mikael 2026-10-27 08:41) ...")
        b = _series_key("D:HIBANA D-22 (Mikael 2026-10-28 13:25) ...")
        assert a[1] == "hibana d 16" and b[1] == "hibana d 22"

    def test_no_date_is_not_a_series(self):
        assert _series_key("F:Model Y Standard wheel spec for winter") is None

    def test_date_before_subject_is_not_a_series(self):
        assert _series_key("D:2026-09-07 Mnemos v10.40.0 shipped") is None

    def test_one_token_subject_is_not_a_series(self):
        assert _series_key("D:Mnemos 2026-09-07 shipped") is None

    def test_prose_is_not_a_series(self):
        assert _series_key("The sweep on 2026-10-14 went fine") is None


class TestStoreTimeEvolution:
    def test_newer_entry_expires_older_and_links(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_mod, "embed", _embed_by_marker)
        m = _mnemos(tmp_path)
        old = m.store_memory("dev", "D:TMG SWEEP 2026-10-14 23:10 → board 12/12 done",
                             skip_dedup=True)["id"]
        res = m.store_memory("dev", "D:TMG SWEEP 2026-10-21 17:55 → board 13/13 done",
                             skip_dedup=True)
        assert [e["id"] for e in res["evolved"]] == [old]
        assert res["evolved"][0]["valid_until"] == date.today().isoformat()
        assert "evolved_note" in res
        older = m.store.get_memory(old)
        assert older.valid_until == date.today().isoformat()
        links = m.store._get_conn().execute(
            "SELECT source_id, target_id, relation_type FROM memory_links").fetchall()
        assert [tuple(r) for r in links] == [(old, res["id"], "evolves")]

    def test_older_date_stored_later_does_not_expire_newer(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_mod, "embed", _embed_by_marker)
        m = _mnemos(tmp_path)
        newer = m.store_memory("dev", "D:TMG SWEEP 2026-10-21 → done", skip_dedup=True)["id"]
        res = m.store_memory("dev", "D:TMG SWEEP 2026-10-14 → done", skip_dedup=True)
        assert "evolved" not in res
        assert m.store.get_memory(newer).valid_until is None

    def test_same_date_is_left_alone(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_mod, "embed", _embed_by_marker)
        m = _mnemos(tmp_path)
        m.store_memory("dev", "D:HIBANA D-16 (Mikael 2026-10-27 08:41) sticker", skip_dedup=True)
        res = m.store_memory("dev", "D:HIBANA D-16 (Mikael 2026-10-27 09:00) revised", skip_dedup=True)
        assert "evolved" not in res

    def test_different_key_same_topic_is_left_alone(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_mod, "embed", _embed_by_marker)
        m = _mnemos(tmp_path)
        a = m.store_memory("dev", "D:HIBANA D-16 (Mikael 2026-10-27) sticker as image", skip_dedup=True)["id"]
        res = m.store_memory("dev", "D:HIBANA D-22 (Mikael 2026-10-28) open for reading", skip_dedup=True)
        assert "evolved" not in res
        assert m.store.get_memory(a).valid_until is None

    def test_other_project_is_left_alone(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_mod, "embed", _embed_by_marker)
        m = _mnemos(tmp_path)
        a = m.store_memory("work", "D:TMG SWEEP 2026-10-14 → done", skip_dedup=True)["id"]
        res = m.store_memory("dev", "D:TMG SWEEP 2026-10-21 → done", skip_dedup=True)
        assert "evolved" not in res
        assert m.store.get_memory(a).valid_until is None

    def test_far_vector_is_a_stranger(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_mod, "embed", _embed_by_marker)
        m = _mnemos(tmp_path)
        a = m.store_memory("dev", "D:TMG SWEEP 2026-10-14 FAR away content", skip_dedup=True)["id"]
        res = m.store_memory("dev", "D:TMG SWEEP 2026-10-21 → done", skip_dedup=True)
        assert "evolved" not in res
        assert m.store.get_memory(a).valid_until is None

    def test_verified_older_entry_gets_link_only(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_mod, "embed", _embed_by_marker)
        m = _mnemos(tmp_path)
        old = m.store_memory("dev", "D:TMG SWEEP 2026-10-14 → done", skip_dedup=True,
                             verified=True)["id"]
        res = m.store_memory("dev", "D:TMG SWEEP 2026-10-21 → done", skip_dedup=True)
        assert res["evolved"][0]["valid_until"] is None
        assert "guard" in res["evolved"][0]
        assert m.store.get_memory(old).valid_until is None
        links = m.store._get_conn().execute(
            "SELECT source_id, target_id, relation_type FROM memory_links").fetchall()
        assert [tuple(r) for r in links] == [(old, res["id"], "evolves")]

    def test_expired_older_entry_hidden_from_default_search_but_not_history(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core_mod, "embed", _embed_by_marker)
        m = _mnemos(tmp_path)
        old = m.store_memory("dev", "D:TMG SWEEP 2026-10-14 → board done", skip_dedup=True)["id"]
        new = m.store_memory("dev", "D:TMG SWEEP 2026-10-21 → board done", skip_dedup=True)["id"]
        ids = [r["id"] for r in m.search("TMG SWEEP board", search_mode="fts")["results"]]
        assert new in ids and old not in ids
        ids_all = [r["id"] for r in m.search("TMG SWEEP board", search_mode="fts",
                                             valid_only=False)["results"]]
        assert old in ids_all
