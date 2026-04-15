"""
File ingestion for Mnemos.

Walks a directory or single file, extracts text from supported formats,
chunks if necessary, and stores each chunk as a memory with provenance
metadata (source path, ingestion timestamp). Designed to be the foundation
for indexing personal documents, code, notes, or any text-based content
without leaving the Mnemos surface.

Built-in extractors handle plain-text formats only (txt, md, code files,
config files, html-as-text). PDF, EPUB, eml, docx, and other binary or
structured formats are not included in v10. The architecture supports
adding extractors via `register_extractor(extension, callable)`, so
community plugins can add format support without modifying core.

Usage from CLI:
    mnemos ingest /path/to/file.md
    mnemos ingest /path/to/folder --pattern "*.md" --recursive
    mnemos ingest /path/to/notes --project notes --chunk 2000

Usage from Python:
    from mnemos import Mnemos
    from mnemos.ingest import ingest_path

    m = Mnemos()
    stats = ingest_path(m, "/path/to/notes", pattern="*.md", recursive=True)
"""

import fnmatch
import os
from pathlib import Path
from typing import Callable, Optional


# Built-in text extensions. Treated as plain text and read directly.
TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".log",
    ".py", ".js", ".jsx", ".ts", ".tsx", ".rb", ".go", ".rs", ".java",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".php", ".sh", ".bash", ".zsh",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".conf", ".cfg",
    ".html", ".htm", ".xml", ".css", ".scss", ".sass",
    ".sql", ".csv", ".tsv",
    ".env", ".dockerfile", ".gitignore",
}

DEFAULT_CHUNK_CHARS = 2000
DEFAULT_PROJECT = "ingested"

# Pluggable extractor registry. Maps file extension to a function that
# takes a Path and returns the extracted text (or None to skip).
_extractors: dict = {}


def register_extractor(extension: str, fn: Callable[[Path], Optional[str]]) -> None:
    """Register a custom text extractor for a file extension.

    Example:
        from mnemos.ingest import register_extractor
        import pypdf

        def extract_pdf(path):
            reader = pypdf.PdfReader(str(path))
            return "\\n".join(p.extract_text() or "" for p in reader.pages)

        register_extractor(".pdf", extract_pdf)
    """
    if not extension.startswith("."):
        extension = "." + extension
    _extractors[extension.lower()] = fn


def _read_text_file(path: Path) -> Optional[str]:
    """Default extractor for plain-text files."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _extract(path: Path) -> Optional[str]:
    """Extract text from a file using the appropriate extractor."""
    ext = path.suffix.lower()
    if ext in _extractors:
        return _extractors[ext](path)
    if ext in TEXT_EXTENSIONS:
        return _read_text_file(path)
    return None


def _chunk(text: str, chunk_size: int) -> list:
    """Split text into chunks of approximately `chunk_size` characters.

    Tries to break at paragraph boundaries first, then sentences, then
    falls back to hard splits. Returns a list of chunk strings.
    """
    if chunk_size <= 0 or len(text) <= chunk_size:
        return [text]

    chunks = []
    paragraphs = text.split("\n\n")
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 2 <= chunk_size:
            current = current + "\n\n" + p if current else p
        else:
            if current:
                chunks.append(current)
            if len(p) <= chunk_size:
                current = p
            else:
                # Paragraph itself too big - hard split
                for i in range(0, len(p), chunk_size):
                    chunks.append(p[i:i + chunk_size])
                current = ""
    if current:
        chunks.append(current)
    return chunks


def _walk(path: Path, pattern: str, recursive: bool):
    """Yield files under `path` matching the glob `pattern`."""
    if path.is_file():
        if fnmatch.fnmatch(path.name, pattern):
            yield path
        return
    if not path.is_dir():
        return
    if recursive:
        for f in path.rglob(pattern):
            if f.is_file():
                yield f
    else:
        for f in path.glob(pattern):
            if f.is_file():
                yield f


def ingest_path(
    mnemos,
    path,
    project: str = DEFAULT_PROJECT,
    subcategory: Optional[str] = None,
    pattern: str = "*",
    recursive: bool = True,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    importance: int = 4,
    layer: str = "semantic",
    skip_dedup: bool = True,
    dry_run: bool = False,
) -> dict:
    """Ingest a file or directory into a Mnemos instance.

    Args:
        mnemos: A Mnemos instance to store memories into
        path: File or directory to ingest
        project: Mnemos project to store under (default "ingested")
        subcategory: Optional subcategory (defaults to source folder name)
        pattern: Glob pattern to match (default "*")
        recursive: Walk subdirectories
        chunk_chars: Max characters per chunk (0 = no chunking)
        importance: Default importance for ingested memories (1-10)
        layer: episodic or semantic (default semantic)
        skip_dedup: Skip dedup check on store (faster, recommended for bulk)
        dry_run: Don't actually store, just report what would happen

    Returns: dict with stats {files, chunks, stored, skipped, errors}
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return {"error": f"Path not found: {path}"}

    stats = {
        "files": 0,
        "chunks": 0,
        "stored": 0,
        "skipped": 0,
        "errors": 0,
        "by_extension": {},
    }

    sub = subcategory or (path.name if path.is_dir() else path.parent.name)

    for file_path in _walk(path, pattern, recursive):
        stats["files"] += 1
        ext = file_path.suffix.lower() or "(none)"
        stats["by_extension"][ext] = stats["by_extension"].get(ext, 0) + 1

        text = _extract(file_path)
        if not text or not text.strip():
            stats["skipped"] += 1
            continue

        rel_path = str(file_path)
        try:
            rel_path = str(file_path.relative_to(path)) if path.is_dir() else file_path.name
        except ValueError:
            rel_path = file_path.name

        chunks = _chunk(text, chunk_chars)
        for i, chunk in enumerate(chunks):
            stats["chunks"] += 1
            tags = f"ingested,file,{file_path.suffix.lstrip('.') or 'plain'}"
            if len(chunks) > 1:
                tags += f",chunk-{i+1}-of-{len(chunks)}"

            content = (
                f"F: {rel_path}{f' (chunk {i+1}/{len(chunks)})' if len(chunks) > 1 else ''} @{file_path}\n"
                f"{chunk.strip()}"
            )

            if dry_run:
                stats["stored"] += 1
                continue

            try:
                result = mnemos.store_memory(
                    project=project,
                    content=content,
                    tags=tags,
                    importance=importance,
                    mem_type="fact",
                    layer=layer,
                    subcategory=sub,
                    skip_dedup=skip_dedup,
                )
                if result.get("status") == "stored":
                    stats["stored"] += 1
                else:
                    stats["skipped"] += 1
            except Exception:
                stats["errors"] += 1

    return stats
