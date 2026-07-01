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
        project defaults to the directory name."""
        graph = get_code_graph()
        if graph is None:
            return ('Code indexing disabled: tree-sitter not installed.\n'
                    'Run: pip install tree-sitter tree-sitter-python')

        root = Path(path).expanduser()
        if not root.is_dir():
            return f'Error: {root} is not a directory.'

        project = project or root.name
        stats = graph.index_project(root, project)
        return (
            f'Indexed {project}: {stats["files"]} files, '
            f'{stats["symbols"]} symbols, {stats["edges"]} edges.\n'
            f'Parse: {stats["parse_s"]}s  Embed: {stats["embed_s"]}s'
        )

    @mcp.tool(name='roam_code_search')
    async def roam_code_search(query: str, k: int = 8) -> str:
        """Semantic search across indexed code symbols by meaning.
        Returns ranked functions/classes/methods with file, type, and preview.
        Index first with roam_index_code if not already done."""
        graph = get_code_graph()
        if graph is None:
            return 'Code indexing disabled: tree-sitter not installed.'

        results = graph.search(query, k=k)
        if not results:
            return (
                f'No code results for "{query}".\n'
                f'Index a project first with roam_index_code.'
            )

        parts = [f'Code search: "{query}" — {len(results)} symbols\n']
        for i, r in enumerate(results, 1):
            file = _short(r['node_id'])
            sym = _sym(r['heading_path'])
            kind = r['unit_type']
            dist = r['distance']
            preview = r['text'].split('\n')[0][:80]
            parts.append(
                f'{i}. [{kind}] {file} :: {sym}  (d={dist})\n'
                f'   {preview}\n'
            )
        return '\n'.join(parts)

    @mcp.tool(name='roam_code_graph')
    async def roam_code_graph(query: str, k: int = 3) -> str:
        """Semantic code search + relationship expansion.
        Finds the top-k symbols matching the query, then for each shows its
        callers (who depends on it), callees (what it calls), and imports
        (cross-file dependencies). One call returns the complete context slice.
        Index first with roam_index_code if not already done."""
        graph = get_code_graph()
        if graph is None:
            return 'Code indexing disabled: tree-sitter not installed.'

        results = graph.search(query, k=k)
        if not results:
            return (
                f'No code results for "{query}".\n'
                f'Index a project first with roam_index_code.'
            )

        parts = [f'Code graph for "{query}" — {len(results)} symbols\n']
        for r in results:
            file = _short(r['node_id'])
            sym = _sym(r['heading_path'])
            kind = r['unit_type']
            dist = r['distance']
            preview = r['text'].split('\n')[0][:80]
            nb = graph.neighbors(r['node_id'], sym)
            parts.append(f'## {sym} ({file}, {kind}, d={dist})')
            parts.append(f'  {preview}')
            parts.append('')
            parts.extend(_format_neighbors(nb, r['node_id']))
            parts.append('---\n')
        return '\n'.join(parts)

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
