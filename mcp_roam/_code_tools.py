"""MCP tool definitions for code indexing — registered on FastMCP instance."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from mcp.server.fastmcp import FastMCP


def _short(node_id: str) -> str:
    """code:Piano::webui/server.py -> webui/server.py"""
    return node_id.split('::', 1)[-1] if '::' in node_id else node_id


def _sym(heading_path: str) -> str:
    """webui/server.py > Session > load_midi -> load_midi"""
    return heading_path.split(' > ')[-1]


def _format_neighbors(nb: dict, node_id: str) -> list[str]:
    lines: list[str] = []
    if nb['callers']:
        lines.append(f'  Callers ({len(nb["callers"])}):')
        for nid, s in nb['callers'][:8]:
            lines.append(f'    <- {_short(nid)} :: {s}')
    if nb['callees']:
        lines.append(f'  Callees ({len(nb["callees"])}):')
        for nid, s in nb['callees'][:8]:
            lines.append(f'    -> {_short(nid)} :: {s}')
    if nb['imports']:
        lines.append(f'  Imports ({len(nb["imports"])}):')
        for nid, s in nb['imports'][:8]:
            lines.append(f'    ~> {_short(nid)} :: {s}')
    if not (nb['callers'] or nb['callees'] or nb['imports']):
        lines.append('  (no resolved edges)')
    return lines


def register_code_tools(
    mcp: FastMCP,
    get_code_graph: Callable[[], Any],
) -> None:
    """Register code indexing tools on the FastMCP instance."""

    @mcp.tool(name='roam_index_code')
    async def roam_index_code(
        path: str, project: str | None = None,
    ) -> str:
        """Index a code project (directory of source files) for semantic search.
        Parses all .py/.ts/.tsx files, embeds symbols, and builds a relationship
        graph (calls/imports/contains). Pass the project root directory path.
        project defaults to the directory name. Idempotent: re-indexing a
        project fully replaces its prior index (no duplicates, no stale state)."""
        graph = get_code_graph()
        if graph is None:
            return ('Code indexing disabled: tree-sitter not installed.\n'
                    'Run: pip install tree-sitter tree-sitter-python')

        root = Path(path).expanduser()
        if not root.is_dir():
            return f'Error: {root} is not a directory.'

        project = project or root.name
        stats = graph.index_project(root, project, replace=True)
        return (
            f'Indexed {project}: {stats["files"]} files, '
            f'{stats["symbols"]} symbols, {stats["edges"]} edges.\n'
            f'Parse: {stats["parse_s"]}s  Embed: {stats["embed_s"]}s'
        )

    @mcp.tool(name='roam_list_code_projects')
    async def roam_list_code_projects() -> str:
        """List all indexed code projects with their root path and last-updated
        time. Use to see what's available for search before querying."""
        graph = get_code_graph()
        if graph is None:
            return 'Code indexing disabled: tree-sitter not installed.'
        projects = graph.list_projects()
        if not projects:
            return 'No projects indexed yet. Use roam_index_code to add one.'
        parts = [f'Indexed code projects ({len(projects)}):\n']
        for p in projects:
            parts.append(
                f'- {p["project"]}\n    root: {p["root_path"]}\n'
                f'    updated: {p["updated_at"]}'
            )
        return '\n'.join(parts)

    @mcp.tool(name='roam_remove_project')
    async def roam_remove_project(project: str) -> str:
        """Remove a project and all its symbols, edges, and embeddings.
        Frees space and clears stale indexes. Pass the project name (see
        roam_list_code_projects)."""
        graph = get_code_graph()
        if graph is None:
            return 'Code indexing disabled: tree-sitter not installed.'
        result = graph.remove_project(project)
        if result['files_removed'] == 0 and result['units_removed'] == 0:
            return f'Project {project!r} was not indexed (nothing to remove).'
        return (
            f'Removed {project!r}: {result["files_removed"]} files, '
            f'{result["units_removed"]} embedding units deleted.'
        )

    @mcp.tool(name='roam_code_search')
    async def roam_code_search(
        query: str, k: int = 8, project: str | None = None,
        kind: str | None = None,
    ) -> str:
        """Semantic search across indexed code symbols by meaning.
        Returns ranked functions/classes/methods with file, type, and preview.
        Results are reranked by default for precision. Pass project= to scope to
        one indexed project (avoids cross-project noise). Pass kind= to filter
        by node type ('function', 'class', 'method'). Index first with
        roam_index_code if not already done."""
        graph = get_code_graph()
        if graph is None:
            return 'Code indexing disabled: tree-sitter not installed.'

        results = graph.search(query, k=k, project=project, kind=kind, rerank=True)
        if not results:
            scope_parts = []
            if project:
                scope_parts.append(f'project={project!r}')
            if kind:
                scope_parts.append(f'kind={kind!r}')
            scope = f' [{", ".join(scope_parts)}]' if scope_parts else ''
            return (f'No code results for "{query}"{scope}.\n'
                    f'Index a project first with roam_index_code.')

        parts = [f'Code search: "{query}" — {len(results)} symbols\n']
        for i, r in enumerate(results, 1):
            file = _short(r['node_id'])
            sym = _sym(r['heading_path'])
            rkind = r['unit_type']
            score = r.get('rerank_score', r.get('distance'))
            preview = r['text'].split('\n')[0][:80]
            parts.append(
                f'{i}. [{rkind}] {file} :: {sym}  (score={score})\n'
                f'   {preview}\n'
            )
        return '\n'.join(parts)

    @mcp.tool(name='roam_code_graph')
    async def roam_code_graph(
        query: str, k: int = 3, project: str | None = None,
        depth: int = 1,
    ) -> str:
        """Semantic code search + relationship expansion.
        Finds the top-k symbols matching the query, then for each shows its
        callers (who depends on it), callees (what it calls), and imports
        (cross-file dependencies). Pass depth>1 for multi-hop traversal (e.g.
        depth=3 shows transitive callers/callees — who ultimately depends on X).
        Pass project= to scope to one indexed project. Index first with
        roam_index_code if not already done."""
        graph = get_code_graph()
        if graph is None:
            return 'Code indexing disabled: tree-sitter not installed.'

        results = graph.search(query, k=k, project=project, rerank=True)
        if not results:
            scope = f' in project {project!r}' if project else ''
            return (f'No code results for "{query}"{scope}.\n'
                    f'Index a project first with roam_index_code.')

        parts = [f'Code graph for "{query}" — {len(results)} symbols'
                 f' (depth={depth})\n']
        for r in results:
            file = _short(r['node_id'])
            sym = _sym(r['heading_path'])
            kind = r['unit_type']
            dist = r['distance']
            preview = r['text'].split('\n')[0][:80]
            nb = graph.neighbors(r['node_id'], sym, depth=depth)
            parts.append(f'## {sym} ({file}, {kind}, d={dist})')
            parts.append(f'  {preview}')
            parts.append('')
            parts.extend(_format_neighbors(nb, r['node_id']))
            parts.append('---\n')
        return '\n'.join(parts)

    @mcp.tool(name='roam_code_search_near')
    async def roam_code_search_near(
        query: str, anchor: str, depth: int = 2, k: int = 8,
        project: str | None = None,
    ) -> str:
        """Semantic search constrained to the graph neighborhood of a node.
        Finds code matching `query` but only within `depth` hops of the anchor
        symbol (by call + import edges, both directions). Answers questions like
        'find error-handling patterns in code that depends on the payment module'.

        anchor: 'file::symbol' (e.g. 'src/lib/payments.ts::processPayment') —
        get these from a prior roam_code_search or roam_code_graph result.
        depth: graph traversal depth (default 2).
        project: scope to one indexed project."""
        graph = get_code_graph()
        if graph is None:
            return 'Code indexing disabled: tree-sitter not installed.'

        # parse 'relpath::symbol' → (node_id, sym)
        parts = anchor.split('::', 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return ("Invalid anchor format. Use 'file::symbol' "
                    "(e.g. 'src/lib/payments.ts::processPayment').")
        file_raw, sym = parts
        # resolve to full node_id
        proj = project or ''
        if proj:
            node_id = f'code:{proj}::{file_raw}'
        else:
            # search across all projects for this relpath
            with graph.repo._get_connection() as conn:
                row = conn.execute(
                    'SELECT node_id FROM code_files WHERE relpath = ? LIMIT 1',
                    (file_raw,)).fetchone()
            if not row:
                return f'Anchor file {file_raw!r} not found in any indexed project.'
            node_id = row['node_id']

        results = graph.search_near(
            query, node_id, sym, depth=depth, k=k, project=project)
        if not results:
            return (f'No results for "{query}" within {depth} hops of '
                    f'{file_raw}::{sym}.')

        rparts = [f'Search near {file_raw}::{sym} (depth={depth}) — '
                  f'{len(results)} symbols\n']
        for i, r in enumerate(results, 1):
            file = _short(r['node_id'])
            rsym = _sym(r['heading_path'])
            kind = r['unit_type']
            score = r.get('rerank_score', r.get('distance'))
            preview = r['text'].split('\n')[0][:80]
            rparts.append(
                f'{i}. [{kind}] {file} :: {rsym}  (score={score})\n'
                f'   {preview}\n'
            )
        return '\n'.join(rparts)

    @mcp.tool(name='roam_watch_code')
    async def roam_watch_code(
        path: str, project: str | None = None,
    ) -> str:
        """Start a file watcher that incrementally re-indexes on save.
        Detects file modifications via inotify (zero polling cost), debounces
        300ms, skips unchanged files (hash check), and re-embeds in ~0.5s.
        The watcher runs in the MCP server background. Pass the project root.
        Index first with roam_index_code for initial bulk index."""
        graph = get_code_graph()
        if graph is None:
            return 'Code indexing disabled: tree-sitter not installed.'

        root = Path(path).expanduser()
        if not root.is_dir():
            return f'Error: {root} is not a directory.'

        project = project or root.name
        status = graph.start_watcher(root, project)
        if status['status'] == 'already_running':
            return f'Watcher already running on {status["root"]}.'
        return (
            f'Watcher started: {project} at {root}\n'
            f'Re-indexes on save (debounce=300ms, hash-skip for no-op writes).'
        )

    @mcp.tool(name='roam_watch_status')
    async def roam_watch_status() -> str:
        """Show the file watcher status: what's being watched, recent events,
        and any errors. Use to verify the watcher is alive and catching saves."""
        graph = get_code_graph()
        if graph is None:
            return 'Code indexing disabled: tree-sitter not installed.'

        status = graph.watcher_status()
        if not status['running']:
            return 'Watcher is not running. Start with roam_watch_code.'

        parts = [f'Watcher: {"running" if status["running"] else "stopped"}']
        if status['root']:
            parts.append(f'Root: {status["root"]}')
        if status['project']:
            parts.append(f'Project: {status["project"]}')

        events = status.get('recent_events', [])
        if events:
            parts.append(f'\nRecent events ({len(events)}):')
            for ev in events[-10:]:
                line = f'  {ev["time"]} {ev["status"]:12} {ev["file"]} ({ev["ms"]}ms)'
                if ev.get('error'):
                    line += f'  ERROR: {ev["error"]}'
                parts.append(line)
        else:
            parts.append('\nNo events yet (waiting for file changes...)')

        return '\n'.join(parts)
