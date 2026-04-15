#!/bin/bash
# Mnemos session hook for Claude Code (or any MCP client with hooks support).
#
# Usage in ~/.claude/settings.json:
#
#   "hooks": {
#     "SessionStart": [{
#       "type": "command",
#       "command": "/path/to/mnemos/scripts/mnemos-session-hook.sh start"
#     }],
#     "Stop": [{
#       "type": "command",
#       "command": "/path/to/mnemos/scripts/mnemos-session-hook.sh stop"
#     }]
#   }
#
# Reads optional context from $CLAUDE_PROJECT_DIR or $PWD for predictive priming.

set -euo pipefail

# Use system mnemos by default, override with MNEMOS_BIN if installed elsewhere
MNEMOS_BIN="${MNEMOS_BIN:-mnemos}"

case "${1:-start}" in
    start)
        # Inject compact briefing into session context
        "$MNEMOS_BIN" briefing 2>/dev/null || true

        # If we have project context (e.g., CWD), surface relevant memories
        CONTEXT="${CLAUDE_PROJECT_DIR:-${PWD:-}}"
        if [ -n "$CONTEXT" ]; then
            echo ""
            echo "## Context-relevant memories"
            "$MNEMOS_BIN" prime "$CONTEXT" --limit 3 2>/dev/null || true
        fi
        ;;
    stop)
        # Optional: print a reminder to store learnings
        # (No-op by default; uncomment to enable)
        # echo "Session ended. Consider storing important learnings via memory_store."
        :
        ;;
    *)
        echo "Usage: $0 {start|stop}" >&2
        exit 1
        ;;
esac
