#!/bin/bash
# Mnemos session hook for Claude Code (or any MCP client with hooks support).
#
# Three subcommands: start / prompt / stop.
#
#   start   Fires on session start. Injects the briefing plus a first-pass
#           priming based on the working directory.
#   prompt  Fires on first user prompt per session. Runs a second priming
#           pass using the user's actual message text as the vec-search
#           query — this is when real topical memories surface. Subsequent
#           prompts in the same session are no-ops (marker file).
#   stop    Fires on session end. Deliberately dumb: nags if the session
#           was long but stored nothing. Does NOT call an LLM — the
#           in-session LLM already had full context, so a post-hoc
#           summarizer would duplicate effort.
#
# Usage in ~/.claude/settings.json:
#
#   "hooks": {
#     "SessionStart": [{
#       "hooks": [{
#         "type": "command",
#         "command": "/path/to/mnemos/scripts/mnemos-session-hook.sh start",
#         "timeout": 30000
#       }]
#     }],
#     "UserPromptSubmit": [{
#       "hooks": [{
#         "type": "command",
#         "command": "/path/to/mnemos/scripts/mnemos-session-hook.sh prompt",
#         "timeout": 5000
#       }]
#     }],
#     "Stop": [{
#       "hooks": [{
#         "type": "command",
#         "command": "/path/to/mnemos/scripts/mnemos-session-hook.sh stop",
#         "timeout": 5000
#       }]
#     }]
#   }
#
# Dependencies: mnemos CLI on $PATH (or MNEMOS_BIN env override), jq.

set -euo pipefail

MNEMOS_BIN="${MNEMOS_BIN:-mnemos}"
STATE_DIR="${MNEMOS_SESSION_STATE_DIR:-/tmp/mnemos-session}"
mkdir -p "$STATE_DIR" 2>/dev/null || true

case "${1:-start}" in
    start)
        # Clear any stale per-session marker (crash recovery)
        if [ -n "${CLAUDE_SESSION_ID:-}" ]; then
            rm -f "$STATE_DIR/${CLAUDE_SESSION_ID}.primed" 2>/dev/null || true
        fi
        # Sweep markers older than 7 days
        find "$STATE_DIR" -type f -mtime +7 -delete 2>/dev/null || true

        # Inject compact briefing
        "$MNEMOS_BIN" briefing 2>/dev/null || true

        # First-pass priming from working directory. The CWD signal alone is
        # weak — the meaningful priming fires on first user prompt via
        # the `prompt` subcommand.
        CWD_CONTEXT="${CLAUDE_PROJECT_DIR:-${PWD:-}}"
        if [ -n "$CWD_CONTEXT" ]; then
            echo ""
            echo "## Context-relevant memories"
            "$MNEMOS_BIN" prime "$CWD_CONTEXT" --limit 3 2>/dev/null || true
        fi
        ;;

    prompt)
        # First-prompt priming. Claude Code sends a JSON payload on stdin
        # with fields including `prompt` (the user's message) and
        # `session_id`. We run vec-search priming once per session using
        # the prompt text as context signal — the real value of this hook
        # is turning "random memories at session start" into "memories
        # relevant to what the user just asked".
        PAYLOAD=$(cat 2>/dev/null || true)
        SID="${CLAUDE_SESSION_ID:-$(echo "$PAYLOAD" | jq -r '.session_id // empty' 2>/dev/null)}"
        PROMPT_TEXT=$(echo "$PAYLOAD" | jq -r '.prompt // empty' 2>/dev/null)

        # Skip if no session id or trivial prompt
        if [ -z "$SID" ] || [ -z "$PROMPT_TEXT" ] || [ "${#PROMPT_TEXT}" -lt 5 ]; then
            exit 0
        fi

        # Once-per-session guard
        MARKER="$STATE_DIR/${SID}.primed"
        if [ -f "$MARKER" ]; then
            exit 0
        fi

        # Cap prompt length; first 500 chars carry intent
        PROMPT_TEXT="${PROMPT_TEXT:0:500}"

        "$MNEMOS_BIN" prime "$PROMPT_TEXT" --limit 5 2>/dev/null || true
        touch "$MARKER"
        ;;

    stop)
        # Deliberately simple: no LLM call at stop-time. The in-session
        # LLM is authoritative for memory decisions. If you want the
        # classic "you had a long session and stored nothing" nag, wire
        # that here using transcript parsing (jq over CLAUDE_TRANSCRIPT)
        # — no LLM required.
        :
        ;;

    *)
        echo "Usage: $0 {start|prompt|stop}" >&2
        exit 1
        ;;
esac
