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

        # Once-per-session guard — atomic mkdir is race-free (touch+test
        # is vulnerable to TOCTOU if two invocations share a session id).
        # mkdir returns 0 only for the winner; losers get non-zero and exit.
        MARKER="$STATE_DIR/${SID}.primed"
        if ! mkdir "$MARKER" 2>/dev/null; then
            exit 0
        fi

        # Cap prompt length; first 500 chars carry intent
        PROMPT_TEXT="${PROMPT_TEXT:0:500}"

        "$MNEMOS_BIN" prime "$PROMPT_TEXT" --limit 5 2>/dev/null || true
        ;;

    stop)
        # Two optional behaviours, both transcript-based, no LLM:
        #   1. Nag if session was long but stored nothing (opt-in:
        #      MNEMOS_STOP_NAG=1). Dumb signal, no content inspection.
        #   2. Mechanical session summary as an episodic memory
        #      (opt-in: MNEMOS_STOP_SUMMARY=1). Stores session_id,
        #      duration-ish, tool call breakdown, first/last user
        #      prompt snippets. Purely structural extraction — does
        #      NOT call an LLM. Next session's briefing picks up the
        #      summary by recency, giving cross-session continuity.
        #
        # Both default off to honor the guiding principle (in-session
        # LLM is authoritative for memory decisions). If a session
        # regularly ends without worthwhile stores, the correct fix is
        # in-session prompting — see CLAUDE.md rules. These hooks are
        # escape hatches, not primary pipelines.
        if [ -z "${CLAUDE_TRANSCRIPT:-}" ] || [ ! -f "${CLAUDE_TRANSCRIPT:-}" ]; then
            exit 0
        fi

        TURNS=$(jq -r 'select(.type == "assistant") | .type' "$CLAUDE_TRANSCRIPT" 2>/dev/null | wc -l)
        STORES=$(jq -r '
            select(.type == "assistant")
            | .message.content[]?
            | select(.type == "tool_use" and (.name | test("memory_store")))
            | .name
        ' "$CLAUDE_TRANSCRIPT" 2>/dev/null | wc -l)

        if [ "${MNEMOS_STOP_NAG:-0}" = "1" ] && [ "$TURNS" -ge 10 ] && [ "$STORES" -eq 0 ] 2>/dev/null; then
            echo "Reminder: session had ${TURNS} turns with no memory_store calls. Consider storing learnings."
        fi

        if [ "${MNEMOS_STOP_SUMMARY:-0}" != "1" ] || [ "$TURNS" -lt 3 ] 2>/dev/null; then
            exit 0
        fi

        SID="${CLAUDE_SESSION_ID:-unknown}"
        NOW=$(date '+%Y-%m-%d %H:%M')
        TOOL_BREAKDOWN=$(jq -r '
            select(.type == "assistant")
            | .message.content[]?
            | select(.type == "tool_use")
            | .name
        ' "$CLAUDE_TRANSCRIPT" 2>/dev/null | sort | uniq -c | sort -rn | awk '{printf "%s=%d ", $2, $1}')

        FIRST_USER=$(jq -r '
            select(.type == "user") | .message.content[]?
            | select(.type == "text" or type == "string")
            | if type == "object" then .text else . end
        ' "$CLAUDE_TRANSCRIPT" 2>/dev/null | head -1 | cut -c1-120)
        LAST_USER=$(jq -r '
            select(.type == "user") | .message.content[]?
            | select(.type == "text" or type == "string")
            | if type == "object" then .text else . end
        ' "$CLAUDE_TRANSCRIPT" 2>/dev/null | tail -1 | cut -c1-120)

        SUMMARY="F: Session ${SID:0:8} @ ${NOW} in ${PWD:-?}: ${TURNS} assistant turns, tools: ${TOOL_BREAKDOWN}| first: ${FIRST_USER} | last: ${LAST_USER}"

        "$MNEMOS_BIN" add --project general --type fact \
            --tags "session-summary,auto-hook,src:stop-hook" \
            --content "$SUMMARY" >/dev/null 2>&1 || true
        ;;

    *)
        echo "Usage: $0 {start|prompt|stop}" >&2
        exit 1
        ;;
esac
