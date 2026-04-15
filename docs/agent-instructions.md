# Agent instructions for Mnemos

This file is an **example** of what to put in your AI client's project-level system-prompt file (`CLAUDE.md` for Claude Code, `.cursorrules` for Cursor, custom-instructions field for Claude Desktop, etc.) to get proactive, high-quality use of the Mnemos MCP tools.

It's a template, not something Mnemos reads. Copy whichever block matches your configuration into your own project's agent-rules file.

The MCP tool descriptions already cover the *format* the agent should use when writing a memory. This file covers the *behaviors* the agent should have: when to search, when to store, when to update, what not to store. Those are decisions the tool descriptions alone can't enforce.

---

## For CML mode deployments (default)

Copy this into `~/.claude/CLAUDE.md`, your project's `CLAUDE.md`, `.cursorrules`, or your Claude Desktop custom instructions:

```markdown
## Memory (Mnemos)

You have access to Mnemos via four MCP tools: `memory_store`, `memory_search`,
`memory_get`, `memory_update`. Use them proactively.

### When to search
- At the start of a session, before you start working on a topic, run
  `memory_search` with a short query describing the topic to surface
  any prior context
- When the user mentions a name, service, project, fact pattern, or
  decision that might have been established before, search first
  rather than asking
- If the first search returns fewer than 3 hits, the search
  auto-widens across projects; consider whether the wider context helps

### When to store
- After a meaningful decision is made (by the user or in the
  conversation), store it with `D:` prefix
- After learning something non-obvious about the user's setup, style,
  or environment, store it with `F:` or `L:` prefix
- After the user states a preference (how they want replies, tone,
  tools they prefer or reject), store it with `P:` prefix
- After catching a gotcha or warning (something that broke, something
  that must not be repeated), store it with `W:` prefix
- Don't store routine status updates, temporary state, or content
  already captured in this CLAUDE.md file

### Format: CML (Compressed Memory Language)
- Type prefix on the first line: `D:` decision, `F:` fact/config,
  `L:` learning, `P:` preference, `C:` contact, `W:` warning
- Relation symbols: `→` leads to / produces, `∵` because, `∴` therefore,
  `@` at / in context of, `✓` confirmed, `✗` wrong, `⚠` caveat,
  `△` changed from, `↔` mutual relation, `#N` reference memory #N
- One fact per line, chain tightly related facts on one line with `;`
- Keep each memory short. Multi-paragraph prose goes into
  `consolidation_lock: true` entries (runbooks, long docs, code)

### When to update
- If you find a memory is wrong or outdated, call `memory_update` with
  the corrected fields rather than storing a new conflicting memory
- If you confirm a memory is still current, it is fine to leave it
  alone; the system tracks access patterns automatically

### What NOT to store
- Third-party claims as fact (attribute them: "user X said Y")
- Content the user asked you to keep private or ephemeral
- Secrets, API keys, passwords, or anything that looks like one
- Test content, scratch notes, "just trying this out" material
```

---

## For prose-mode deployments (`MNEMOS_CML_MODE=off`)

Copy this instead if you set `MNEMOS_CML_MODE=off`:

```markdown
## Memory (Mnemos)

You have access to Mnemos via four MCP tools: `memory_store`, `memory_search`,
`memory_get`, `memory_update`. Use them proactively.

### When to search
- At the start of a session, before you start working on a topic, run
  `memory_search` with a short query describing the topic to surface
  any prior context
- When the user mentions a name, service, project, fact pattern, or
  decision that might have been established before, search first
  rather than asking
- If the first search returns fewer than 3 hits, the search
  auto-widens across projects; consider whether the wider context helps

### When to store
- After a meaningful decision is made (by the user or in the
  conversation), store the decision and the reasoning behind it
- After learning something non-obvious about the user's setup, style,
  or environment, store it as a short factual note
- After the user states a preference (how they want replies, tone,
  tools they prefer or reject), store the preference with context
- After catching a gotcha or warning (something that broke, something
  that must not be repeated), store it as a warning
- Don't store routine status updates, temporary state, or content
  already captured in this CLAUDE.md file

### Format: clear natural prose
- One short paragraph per memory (one or two sentences for a single
  fact, a short paragraph for a cluster of related facts)
- No special prefixes or symbols required; Mnemos indexes plain text
- Include the concrete specifics: names, numbers, dates, paths,
  reasoning. Preserve every detail that will matter when the memory
  is retrieved months from now
- Multi-paragraph docs (runbooks, long code samples, creative writing)
  should be stored with `consolidation_lock: true` so the Nyx cycle
  leaves them alone

### When to update
- If you find a memory is wrong or outdated, call `memory_update` with
  the corrected fields rather than storing a new conflicting memory
- If you confirm a memory is still current, it is fine to leave it
  alone; the system tracks access patterns automatically

### What NOT to store
- Third-party claims as fact (attribute them: "user X said Y")
- Content the user asked you to keep private or ephemeral
- Secrets, API keys, passwords, or anything that looks like one
- Test content, scratch notes, "just trying this out" material
```

---

## A note on system-prompt scope

Mnemos can only instruct the AI through the MCP tool descriptions it exposes. `MNEMOS_CML_MODE` controls those and the Nyx cycle prompts, which is everything Mnemos *owns*. Your project's `CLAUDE.md` (or equivalent) is **yours to manage** — Mnemos has no way to rewrite it when you flip the mode. If you switch between CML and prose deployments, swap the corresponding block above into your agent-rules file.
