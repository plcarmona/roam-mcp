# mcp-roam

**An MCP server that gives AI agents a semantic brain over an org-roam knowledge graph.**

`mcp-roam` bridges any [Model Context Protocol](https://modelcontextprotocol.io/) client (OpenCode, Claude, etc.) to an [org-roam](https://www.orgroam.com/) Zettelkasten. Beyond plain graph traversal, it embeds the entire graph **inside the existing org-roam SQLite database** so an agent can search your notes by *meaning* — and ingest new knowledge directly from YouTube transcripts or research papers.

---

## Why

org-roam is a powerful plain-text knowledge graph, but its search is keyword-only and it lives inside Emacs. `mcp-roam` turns it into a queryable semantic memory that any AI agent can read, extend, and reason over — without leaving the editor or chat. It is a worked example of:

- Turning a personal tool into an **MCP-native, agent-accessible service**.
- Embedding a **vector store inside an existing SQLite database** (no separate vector DB to run).
- A **local, private RAG pipeline** — embeddings, reranking, and LLM summarization all run on-device via Ollama.
- **One vector index over notes *and* code** — the same SQLite store serves semantic search across your knowledge graph and your source tree.

---

## Features

- **Graph operations** — search, backlinks, forward-links, N-degree subgraphs, tags, recent notes, daily notes.
- **Capture & authoring** — create notes, append to them, and dump structured research notes (paper metadata + findings).
- **Semantic search** — search by meaning, not keywords. Two-stage retrieval (vector KNN → cross-encoder rerank).
- **Claim extraction** — decompose dense notes (lectures, papers) into atomic, self-contained, embeddable claims.
- **LLM summarization** — map-reduce summaries for long notes via a local model.
- **Code understanding** — index a Python codebase into the *same* vector store, search symbols by meaning, and expand a query into its callers / callees / imports graph. A file watcher keeps the index live on every save.
- **YouTube ingestion** — transcribe hour-long videos into searchable notes asynchronously, then index them.
- **Privacy-first** — all AI runs locally (Ollama); your notes never leave the machine.

---

## Architecture

```
┌──────────────┐     MCP / stdio      ┌────────────────────────┐
│  MCP client  │◄────────────────────►│       mcp-roam         │
│ (AI agent)   │                      │   (FastMCP, Python)     │
└──────────────┘                      └───────────┬────────────┘
                                                  │
                            ┌─────────────────────┼──────────────────────┐
                            │ sqlite3 (RO graph)  │ sqlite-vec (RW vec)  │ pathlib (RW files)
                            ▼                     ▼                      ▼
                     ┌──────────────────────────────────────────┐   ┌──────────┐
                     │            org-roam SQLite DB            │   │  *.org   │
                     │  nodes · links · tags · files · aliases  │   │  files   │
                     │        embed_vec · embed_units           │   └──────────┘
                     └──────────────────────────────────────────┘

   Local services (optional, for AI features):          External (optional):
   ┌──────────────────────┐   ┌──────────────────────┐  ┌──────────────────┐
   │       Ollama         │   │      yt-service      │  │     scite.ai     │
   │  embeddings · LLM ·  │   │  (YouTube → text)    │  │   (research,     │
   │       reranker       │   │                      │  │    via its MCP)  │
   └──────────────────────┘   └──────────────────────┘  └──────────────────┘
```

Key boundary: the org-roam graph tables are opened **read-only** (Emacs owns them); only the `embed_*` tables and `.org` files are written by this server.

---

## Semantic search: a vector store *inside* org-roam

The standout design choice is **co-location**. Rather than spinning up a separate vector database (Chroma, Qdrant, etc.), `mcp-roam` stores embeddings in two tables appended to the *same* SQLite database org-roam already uses:

```sql
CREATE VIRTUAL TABLE embed_vec   USING vec0(embedding float[1024]);  -- sqlite-vec
CREATE TABLE            embed_units(...);   -- metadata: node, heading path, unit type, text
```

Benefits: zero new infrastructure, atomic backups (one file = graph + vectors), and `embed_`-prefixed tables are untouched by org-roam's own `clear` operations.

The pipeline:

1. **Org-aware segmentation** — `segmenter.py` splits a note into semantic units by heading structure (not naive character chunks). It classifies each unit (`summary`, `concept`, `heading`, `claim`), skips noise (properties drawers, raw transcripts), and merges tiny siblings so "Key Concepts" sections don't explode into hundreds of vectors.
2. **Embedding** — each unit is vectorized by Ollama (`snowflake-arctic-embed2`, 1024-dim) and stored via sqlite-vec. A content hash avoids re-embedding unchanged notes.
3. **Two-stage retrieval** — `roam_semantic_search` runs sqlite-vec KNN for fast candidates, then a **cross-encoder reranker** (`Qwen3-Reranker-4B`) scores each candidate against the query via yes/no logprobs, reordering by true relevance.

This makes notes discoverable by the *idea they express*, even when the exact words differ.

---

## Tool reference

`mcp-roam` exposes 23 tools and 3 prompts.

| Tool | Description |
|------|-------------|
| `roam_search` | Keyword search by title / alias / tag |
| `roam_get_node` | Full node content by ID or title |
| `roam_backlinks` | Nodes linking *to* a node |
| `roam_context` | Rich context: content + backlinks + forward links + tags |
| `roam_subgraph` | N-degree neighborhood around a node |
| `roam_tags` | List tags, or nodes for a given tag |
| `roam_recent` | Recently modified notes |
| `roam_daily` | Get/create a daily note by date |
| `roam_capture` | Create a new note |
| `roam_append` | Append to an existing note (under a heading) |
| `roam_research_dump` | Structured research note (paper/web → graph) |
| `roam_index` | Embed one or all notes for semantic search |
| `roam_semantic_search` | Meaning-based search with reranking |
| `roam_extract_claims` | Decompose a note into atomic embeddable claims |
| `roam_enhance` | LLM-generated summary (map-reduce for long notes) |
| `roam_index_stats` | Embedding index statistics |
| `roam_index_code` | Index a code project (Python) for semantic search |
| `roam_code_search` | Semantic search across indexed code symbols |
| `roam_code_graph` | Symbol search + callers / callees / imports expansion |
| `roam_watch_code` | Watch a project and re-index incrementally on save |
| `roam_watch_status` | Show the file watcher status and recent events |
| `roam_youtube_note` | Start async YouTube → note transcription |
| `roam_youtube_note_status` | Poll a transcription job |

---

## Example 1 — Semantic search over Jordan Peterson lectures

Imagine several Jordan Peterson lecture transcripts in your graph. A note on *Personality 13* discusses how the Big Five trait **openness** predicts political liberalism — but it never uses the phrase "how personality shapes politics."

**Keyword search misses it:**

```
roam_search(query="how personality shapes politics")
→ No nodes found matching "how personality shapes politics".
```

**Semantic search finds it by meaning.** First index the relevant notes (once):

```
roam_index(title="Personality 13: Personality and Politics")
→ indexed 6/6 units (type: structural segmentation)
```

Then query:

```
roam_semantic_search(query="how personality shapes politics", k=5)
```

```
Semantic search: "how personality shapes politics" — 3 notes matched

## Personality 13: Personality and Politics (rerank: 0.97)
> [Key Concepts] People high in openness tend toward liberalism and
> creativity; high conscientiousness correlates with conservatism and
> orderliness. These trait distributions predict political orientation...
ID: 9f3a...
File: 20240312101500-personality_13.org
---
## Big Five and Ideology (rerank: 0.91)
> [Summary] Political belief is substantially heritable and maps onto
> personality dimensions...
---
```

The reranker surfaced the exact passage an agent needs — without an exact-word match. The agent can now call `roam_context` to pull the surrounding notes and synthesize an answer.

---

## Example 2 — YouTube transcript → searchable note

Turn a fresh lecture into searchable knowledge in three steps.

**1. Start the transcription** (non-blocking — it returns immediately, even for hour-long videos):

```
roam_youtube_note(
  url="https://youtu.be/ysQm6pF5nEo",
  topic="Jordan Peterson"
)
→ Started transcription job 7c2f1a for https://youtu.be/ysQm6pF5nEo.
  Check status with roam_youtube_note_status(job_id="7c2f1a").
```

**2. Poll until done** (the agent does this automatically):

```
roam_youtube_note_status(job_id="7c2f1a")
→ DONE - created roam note: $ROAM_DIR/20240620143022-jordan_peterson_lecture.org
  Title: Jordan Peterson — Personality and Politics
  Transcript: $ROAM_DIR/transcripts/20240620143022.txt
```

**3. Summarize, index, then query** — the new note is now part of the same pipeline:

```
roam_enhance(title="Jordan Peterson — Personality and Politics")
→ Enhanced summary (map-reduce over 11 chunks)

roam_index(title="Jordan Peterson — Personality and Politics")
→ indexed 8/8 units

roam_semantic_search(query="how personality shapes politics")
→ now also returns passages from this freshly-ingested lecture
```

From a raw YouTube URL to a semantically-queryable note — no copy-paste, no manual tagging.

---

## Example 3 — Semantic search over a codebase

Index any Python project into the *same* vector store as your notes, then ask for a concept in natural language and get the exact symbol plus its call graph.

**1. Index the project (once):**

```
roam_index_code(path="/home/pit/projects/webui")
→ Indexed webui: 42 files, 1180 symbols, 3402 edges.
  Parse: 3.1s  Embed: 12.4s
```

**2. Ask for a concept — it returns the symbol and who calls it / what it calls:**

```
roam_code_graph(query="load a MIDI file into the synth", k=3)
→ Code graph for "load a MIDI file into the synth" — 3 symbols

## load_midi (webui/server.py, function, d=0.21)
  reads a .mid and routes note-on events to the engine
  Callers (2):
    <- webui/server.py :: handle_upload
  Callees (4):
    -> webui/engine.py :: note_on
    -> webui/parser.py :: parse_smf
```

One call gives the agent the symbol, its callers, and its callees — enough to answer or refactor without grepping. Code symbols reuse the `embed_*` tables (with `code:`-prefixed IDs), so there is no separate index.

**3. Keep it live** — edits re-index on save:

```
roam_watch_code(path="/home/pit/projects/webui")
→ Watcher started: webui ... re-indexes on save (debounce=300ms).
```

---

## Dependencies

`mcp-roam` is intentionally lean on the Python side and relies on **local, private** services for AI.

**Python (pip / uv)** — Python ≥ 3.14

| Package | Role |
|---------|------|
| `mcp[cli]` | MCP SDK + CLI runner |
| `sqlite-vec` | In-DB vector storage and KNN search |
| `tree-sitter` + `tree-sitter-python` / `-typescript` | Source parsing → symbol extraction for code indexing |
| `watchfiles` | inotify-based incremental re-indexing on save |

Everything else is stdlib (`sqlite3`, `pathlib`, `re`, `uuid`, `urllib`, `dataclasses`, `concurrent.futures`).

**Local services (optional, enable AI features)**

| Service | Role | Models |
|---------|------|--------|
| [Ollama](https://ollama.com) | Embeddings, reranking, LLM | `snowflake-arctic-embed2` (embed), `Qwen3-Reranker-4B` (rerank), `granite3.3` (LLM) |

Core graph tools work **without** Ollama. Semantic search, claims, enhancement, and code indexing each degrade gracefully and report what's missing (Ollama, sqlite-vec, or tree-sitter).

**External service (optional)**

| Service | Role |
|---------|------|
| `yt-service` | HTTP microservice (`$YT_SERVICE_URL`) that downloads and transcribes YouTube videos. `mcp-roam` only calls it over HTTP — no Python dependency added. |
| scite.ai (via its own MCP) | Research literature, used together with `roam_research_dump`. |

---

## Configuration

All config is via environment variables.

```bash
ROAM_DIR=$HOME/roam                      # org-roam directory (the .org files)
ROAM_DB=$HOME/.emacs.d/org-roam.db        # org-roam SQLite database

OLLAMA_HOST=localhost:11434               # Ollama API
OLLAMA_EMBED_MODEL=snowflake-arctic-embed2
OLLAMA_RERANKER_MODEL=awenleven/Qwen3-Reranker-4B:Q4_K_M
OLLAMA_MODEL=granite3.3:latest            # for enhance / claim extraction

YT_SERVICE_URL=http://localhost:9000/yt   # YouTube transcription service
```

## Run

```bash
uv run mcp-roam          # starts the MCP server over stdio
```

Register it with an MCP client, e.g. OpenCode (`~/.config/opencode/opencode.json`):

```json
{
  "mcp": {
    "roam": {
      "type": "local",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/roam", "mcp-roam"],
      "enabled": true
    }
  }
}
```

---

## Project structure

```
mcp_roam/
├── server.py       ← FastMCP entry point + composition root (lifespan DI)
├── _tools.py       ← 16 graph/semantic MCP tool definitions
├── _code_tools.py  ← 5 code-indexing MCP tools (index/search/graph/watch)
├── youtube.py      ← 2 YouTube transcription tools (HTTP, stdlib-only)
├── prompts.py      ← 3 MCP prompts (assistant, research, analyze)
├── embeddings.py   ← sqlite-vec store + Ollama embed/rerank
├── segmenter.py    ← org-aware semantic unit segmentation
├── code.py         ← code symbol graph: tree-sitter parse + embed + callers/callees + watcher
├── llm.py          ← Ollama LLM: map-reduce summary + claim extraction
├── domain.py       ← frozen dataclasses + org parsing (zero deps)
├── interfaces.py   ← Protocol definitions (DIP contracts)
├── repo.py         ← read-only SQLite repository (org-roam schema)
├── files.py        ← atomic file I/O + daily-note paths
├── capture.py      ← note creation / append
├── context.py      ← graph context + subgraph assembly
└── research.py     ← structured research note builder
```

## Design decisions

- **SOLID throughout** — `interfaces.py` defines `RoamReader`/`RoamWriter`/`FileAccess` Protocols; `repo.py` and `files.py` implement them; tools depend only on interfaces. One module = one responsibility.
- **Dependency injection via FastMCP lifespan** — the server hands each tool its deps (`reader`, `file_access`, `embed_repo`, `code_graph`) from the lifespan context; no globals, trivial to test.
- **Read-only on the graph, read-write on our own tables** — Emacs owns org-roam's tables; we only append `embed_*` (vectors + code symbols) and `code_projects`. No locking risk, no schema conflicts.
- **stdlib-first** — HTTP, JSON, hashing, concurrency all use the standard library. Pip deps are limited to the MCP SDK, sqlite-vec, tree-sitter (code parsing), and watchfiles (re-indexing on save).
- **Graceful degradation** — no Ollama? Graph tools still work. No sqlite-vec? Semantic tools report it clearly instead of crashing.
- **Async without threads blocking the event loop** — Ollama calls and the rerank fan-out run via `asyncio.to_thread` / `ThreadPoolExecutor`.
