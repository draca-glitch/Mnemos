# Session Hooks

Mnemos is designed to inject memory context into an AI session automatically.
Most MCP clients (Claude Code, Cursor, etc.) support hook-style commands
that fire on session lifecycle events. Wiring Mnemos into those hooks turns
`briefing` and `prime` into no-effort background behavior.

The repository ships a reference implementation at
[`scripts/mnemos-session-hook.sh`](../scripts/mnemos-session-hook.sh).

## The three-hook pattern

| Event | Subcommand | Purpose |
| --- | --- | --- |
| `SessionStart` | `start` | Inject the standard briefing + first-pass priming from the working directory. Briefing is sentence-aware truncated (v10.1.2) so cut-off lines end with ` …` rather than dangling mid-sentence. |
| `UserPromptSubmit` | `prompt` | Fire on first user message per session. Run a second priming pass using the user's actual question as the vec-search context. This is where meaningful topical memories surface — CWD alone is usually too weak a signal. Subsequent prompts in the same session are no-ops via a marker file. |
| `Stop` | `stop` | No-op by default. If you want a "long session without any stores" nag, wire it here via transcript parsing (no LLM needed). |

## Why two priming passes (start + prompt)?

At `SessionStart` the only context signal is the working directory. That's
weak — a bare `/root` or `/home/user` path matches random memories across
all projects. The `UserPromptSubmit` hook has access to what the user just
typed, which is a much stronger vec-search anchor.

Without the prompt-time pass, priming surfaces semi-random memories. With
it, priming actually surfaces what you need.

## Why not an LLM-assisted stop hook?

It would duplicate work. The in-session LLM already has full context and
is the authoritative decider of what's worth storing via `memory_store`.
A stop-time LLM pass would:

- Double the API cost per session
- Split responsibility (which memories came from which pipeline?)
- Create a lag between "decision made" and "memory persisted"

If sessions routinely end without stored memories, the fix is in-session
prompting (CLAUDE.md rules, tool-use instructions), not a second pipeline.
The reference stop hook therefore does nothing unless you customize it.

## Wiring into Claude Code

Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/mnemos/scripts/mnemos-session-hook.sh start",
            "timeout": 30000
          }
        ]
      }
    ],
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/mnemos/scripts/mnemos-session-hook.sh prompt",
            "timeout": 5000
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/mnemos/scripts/mnemos-session-hook.sh stop",
            "timeout": 5000
          }
        ]
      }
    ]
  }
}
```

Dependencies: `mnemos` CLI on `$PATH` (installed via `pip install mnemos`
or `uv pip install mnemos`), and `jq` for parsing the `UserPromptSubmit`
JSON payload.

## Wiring the CWD → project heuristic

By default the `prime` CLI takes a free-text context string and vec-searches
across all projects. For sharper results, applications can override
`Mnemos.CWD_PROJECT_MAP` in Python:

```python
from mnemos import Mnemos

class MyMnemos(Mnemos):
    CWD_PROJECT_MAP = [
        # (path_prefix, project, subcategory)
        ("/home/me/code/web",     "dev",     "web"),
        ("/home/me/code/infra",   "dev",     "infra"),
        ("/home/me/personal",     "personal", None),
    ]

m = MyMnemos()
# When cwd matches /home/me/code/web/foo, results are filtered to
# project=dev and the query is augmented with "dev web" tokens
results = m.prime(context="authentication flow", cwd="/home/me/code/web/foo")
```

Without the map, `cwd` is ignored and priming falls back to pure text-
signal search. First-matching prefix wins, so order most-specific paths
first.

## Customizing the reference script

The reference `scripts/mnemos-session-hook.sh` is a starting point. Common
modifications:

- **Per-session state directory**: set `MNEMOS_SESSION_STATE_DIR`
  to move the `{session_id}.primed` markers somewhere other than
  `/tmp/mnemos-session/`.
- **Namespace isolation**: set `MNEMOS_NAMESPACE` in the hook command
  so each session uses a distinct memory namespace.
- **Enable stop-hook reminder**: uncomment the stop subcommand's echo
  line if you want a "session ended, consider storing learnings"
  nudge in your terminal.
