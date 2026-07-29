"""Tests for embed cache healing: orphaned .incomplete cleanup and empty
refs/main detection.

huggingface_hub leaves .incomplete temp files when a download is interrupted
and never garbage-collects them. A failed initial download can also leave
refs/main empty, which causes snapshot_download(local_files_only=True) to
return the snapshots/ parent dir instead of the hash subdir, producing a
NoSuchFile error from onnxruntime. _clean_broken_cache() runs before the
model is loaded and removes both conditions so the next load triggers a
clean re-download.

The cleanup is safe with concurrent downloads: only .incomplete files older
than _INCOMPLETE_STALE_SECONDS are removed, and empty refs/main is deleted
as a file (not a dir nuke) so already-downloaded files survive.
"""

import os
import time

from mnemos import embed


def _make_model_dir(cache, repo_slug, *, refs_content=None, incomplete=True,
                    incomplete_age=7200):
    """Create a fake HF cache layout for one model.

    repo_slug: e.g. "qdrant/multilingual-e5-large-onnx"
    refs_content: bytes to write into refs/main. None = don't create the file.
    incomplete_age: age in seconds for .incomplete files (default 2h = stale).
    """
    model_dir_name = f"models--{repo_slug.replace('/', '--')}"
    model_dir = cache / model_dir_name
    blobs = model_dir / "blobs"
    snapshots = model_dir / "snapshots"
    refs = model_dir / "refs"
    blobs.mkdir(parents=True)
    snapshots.mkdir(parents=True)
    refs.mkdir(parents=True)

    # A complete blob + symlink in the snapshot
    (blobs / "aaa").write_bytes(b"model")
    hash_dir = snapshots / "abc123"
    hash_dir.mkdir()
    (hash_dir / "model.onnx").symlink_to("../../blobs/aaa")

    if refs_content is not None:
        (refs / "main").write_bytes(refs_content)

    if incomplete:
        for name in ("bbb.incomplete", "ccc.incomplete"):
            p = blobs / name
            p.write_bytes(b"\x00" * 100)
            old = time.time() - incomplete_age
            os.utime(p, (old, old))

    return model_dir


class TestCleanIncompleteFiles:
    def test_removes_stale_incomplete_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(embed, "FASTEMBED_CACHE", str(tmp_path))
        _make_model_dir(tmp_path, "qdrant/multilingual-e5-large-onnx")
        embed._clean_broken_cache()
        assert not list(tmp_path.glob("models--*/blobs/*.incomplete"))

    def test_preserves_recent_incomplete_files(self, tmp_path, monkeypatch):
        """Active downloads (recent .incomplete) must not be touched."""
        monkeypatch.setattr(embed, "FASTEMBED_CACHE", str(tmp_path))
        _make_model_dir(tmp_path, "qdrant/multilingual-e5-large-onnx",
                        incomplete_age=60)  # 1 min ago = active
        embed._clean_broken_cache()
        assert list(tmp_path.glob("models--*/blobs/*.incomplete"))

    def test_preserves_complete_blobs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(embed, "FASTEMBED_CACHE", str(tmp_path))
        _make_model_dir(tmp_path, "qdrant/multilingual-e5-large-onnx")
        embed._clean_broken_cache()
        assert (tmp_path / "models--qdrant--multilingual-e5-large-onnx/blobs/aaa").exists()

    def test_noop_when_cache_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(embed, "FASTEMBED_CACHE", str(tmp_path / "nonexistent"))
        embed._clean_broken_cache()


class TestCleanEmptyRefsMain:
    def test_deletes_empty_refs_file_only(self, tmp_path, monkeypatch):
        """Empty refs/main is deleted, but the model dir and its files survive."""
        monkeypatch.setattr(embed, "FASTEMBED_CACHE", str(tmp_path))
        model_dir = _make_model_dir(
            tmp_path, "qdrant/multilingual-e5-large-onnx",
            refs_content=b"", incomplete=False,
        )
        embed._clean_broken_cache()
        assert model_dir.exists()
        assert not (model_dir / "refs/main").exists()
        # Already-downloaded files are preserved
        assert (model_dir / "blobs/aaa").exists()
        assert (model_dir / "snapshots/abc123/model.onnx").exists()

    def test_preserves_model_dir_with_valid_refs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(embed, "FASTEMBED_CACHE", str(tmp_path))
        model_dir = _make_model_dir(
            tmp_path, "qdrant/multilingual-e5-large-onnx",
            refs_content=b"abc123def456", incomplete=False,
        )
        embed._clean_broken_cache()
        assert model_dir.exists()
        assert (model_dir / "refs/main").exists()

    def test_preserves_model_dir_without_refs_file(self, tmp_path, monkeypatch):
        # A model downloaded via the deprecated tar.gz URL has no refs/ at all.
        monkeypatch.setattr(embed, "FASTEMBED_CACHE", str(tmp_path))
        model_dir = _make_model_dir(
            tmp_path, "qdrant/multilingual-e5-large-onnx",
            refs_content=None, incomplete=False,
        )
        embed._clean_broken_cache()
        assert model_dir.exists()


class TestCleanMultipleModels:
    def test_cleans_all_models_selectively(self, tmp_path, monkeypatch):
        monkeypatch.setattr(embed, "FASTEMBED_CACHE", str(tmp_path))
        broken = _make_model_dir(
            tmp_path, "qdrant/broken-model",
            refs_content=b"", incomplete=True,
        )
        healthy = _make_model_dir(
            tmp_path, "qdrant/healthy-model",
            refs_content=b"validhash", incomplete=True,
        )
        embed._clean_broken_cache()
        assert broken.exists()
        assert not (broken / "refs/main").exists()
        assert healthy.exists()
        assert (healthy / "refs/main").exists()
        assert not list(healthy.glob("blobs/*.incomplete"))


class TestGetModelCallsCleanup:
    def test_cleanup_runs_before_model_load(self, tmp_path, monkeypatch):
        monkeypatch.setattr(embed, "FASTEMBED_CACHE", str(tmp_path))
        _make_model_dir(
            tmp_path, "qdrant/multilingual-e5-large-onnx",
            refs_content=b"", incomplete=True,
        )
        monkeypatch.setattr(embed, "_instance", None)

        calls = []

        class FakeTextEmbedding:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        monkeypatch.setattr(embed, "_resource", type("R", (), {"guard_memory": staticmethod(lambda: None)})())
        import sys
        monkeypatch.setitem(sys.modules, "fastembed",
                            type("M", (), {"TextEmbedding": FakeTextEmbedding}))

        embed._get_model()
        assert calls, "TextEmbedding was never constructed"
        assert not list(tmp_path.glob("models--*/blobs/*.incomplete"))
        # refs/main deleted but model dir preserved
        model_dir = tmp_path / "models--qdrant--multilingual-e5-large-onnx"
        assert model_dir.exists()
        assert not (model_dir / "refs/main").exists()
