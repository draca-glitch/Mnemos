# English-primary stores

Since v10.15.0 the recommended convention for Mnemos stores is English-primary
content: the storing agent writes memories in English regardless of the
conversation language, keeping verbatim quotes and canonical terms in their
original language.

## Why

The NLI decision layer's strongest model (dedup confirmation, contradiction
detection) is English-only; the multilingual fallback runs a tier below
(contradiction AUC 0.84 vs 0.94 on the project bench). An English store gets
the strong model on every decision. It also avoids the alternative designs,
which all manage bilingual state: a translation column needs sync invariants
across every write path, and per-call translation puts an LLM in the store
hot path.

What you accept in exchange:

- one-time nuance loss when migrating existing non-English rows (keep the
  pre-migration backup; the originals are the fidelity record)
- keyword (FTS) search in the original language stops matching migrated
  content; vector search is multilingual and compensates in hybrid retrieval

## The convention

Give the storing agent an instruction like this (adapt the examples to your
language):

> Store all new memories in English CML. Verbatim quotes stay in the
> original language; never translate quoted speech. Terms of art, legal
> identifiers, and institution names stay untranslated when the original
> word is the canonical retrieval key or the English equivalent is lossy;
> an optional one-time gloss in parentheses helps both search and NLI.
> Generic prose words: always English.

The "canonical retrieval key" test matters more than translatability. If
your documents, emails, and search queries use the original-language term,
translating it in the store breaks keyword retrieval against your own
vocabulary.

## Migrating an existing store

Order of operations per machine:

1. Update Mnemos to >= 10.15.0 and restart any MCP server processes.
2. Back up the store: `sqlite3 "$MNEMOS_DB" ".backup '$MNEMOS_DB.bak-pre-english'"`
3. Configure the translation LLM (standard `MNEMOS_LLM_*` env; optionally
   pin a stronger model just for this job via `MNEMOS_LLM_MODEL_TRANSLATE`).
   Translation quality becomes the store, so use the strongest model you
   are willing to pay for, and prefer per-record calls over batch prose.
4. Dry-run and eyeball the report before writing anything:
   `python scripts/translate_store_english.py --dry-run`
5. Real run: `python scripts/translate_store_english.py --execute`
6. Enable the NLI decision layer (`pip install mnemos[nli]`, then
   `MNEMOS_DEDUP_CONFIRM=nli`, `MNEMOS_CONTRADICT_MODE=nli`,
   `MNEMOS_NYX_CONTRADICT_FINDER=nli`).
7. Add the storing convention (above) to the agent instructions on that
   machine so new memories arrive in English.

Notes from the reference migration (a ~670-active-row store, Swedish/English
mixed): about a third of rows needed translation, cost a few dollars on a
mid-tier model, took minutes. A few giant multi-topic rows failed the
line-structure guard twice and kept their originals; that is the guard
working, not a failure. Rows that still classify as non-English afterwards
are usually quote-dense rows where the preserved verbatim quotes carry the
signal while the prose is English.

What NOT to migrate:

- locked rows (`consolidation_lock=1`): document-shaped content must not be
  machine-rewritten (the script skips them)
- archived rows: they are the immutable lineage layer; the NLI decision
  path never reads them, tier-2 recall is vector-based and multilingual,
  and keeping them verbatim preserves the original phrasing forever
