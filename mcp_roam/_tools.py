"""MCP tool definitions for org-roam — registered on FastMCP instance."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from mcp.server.fastmcp import FastMCP

from mcp_roam.capture import append_to_note, create_note
from mcp_roam.context import build_context, build_subgraph
from mcp_roam.domain import ResearchNoteData, get_excerpt
from mcp_roam.interfaces import FileAccess, RoamReader
from mcp_roam.research import research_dump

def _q(s: str | None) -> str | None:
    if s and s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    return s


def _node_to_text(node: Any, tags: list[str] | None = None) -> str:
    title = _q(node.title) or '(untitled)'
    file = _q(node.file) or ''
    tags_str = ', '.join(tags) if tags else ''
    lines = [
        f'## {title}',
        f'- ID: {_q(node.id)}',
        f'- File: {file}',
    ]
    if tags_str:
        lines.append(f'- Tags: {tags_str}')
    if node.todo:
        lines.append(f'- TODO: {_q(node.todo)}')
    lines.append('')
    return '\n'.join(lines)


async def _resolve_node(
    reader: RoamReader,
    args: dict,
) -> Any | None:
    node_id = args.get('id')
    title = args.get('title')
    if node_id:
        return await reader.get_node_by_id(node_id)
    if title:
        return await reader.get_node_by_title(title)
    return None


def register_all(
    mcp: FastMCP,
    get_deps: Callable[[], tuple[RoamReader, FileAccess, Any]],
) -> None:
    """Register all org-roam MCP tools on the FastMCP instance."""

    @mcp.tool(name='roam_search')
    async def roam_search(query: str, limit: int = 10) -> str:
        """Search org-roam nodes by title, alias, or tag. Returns matching notes with excerpts."""
        reader, file_access, _embed = get_deps()
        nodes = await reader.search_nodes(query, limit)

        if not nodes:
            return f'No nodes found matching "{query}".'

        parts: list[str] = [f'Found {len(nodes)} nodes matching "{query}":\n']
        for node in nodes:
            tags = [_q(t) or t for t in await reader.get_node_tags(node.id)]
            parts.append(_node_to_text(node, tags))
            file_path = _q(node.file) or ''
            if file_path and file_access.exists(file_path):
                content = file_access.read_file(file_path)
                parts.append(get_excerpt(content, 10))
                parts.append('---\n')

        return '\n'.join(parts)

    @mcp.tool(name='roam_get_node')
    async def roam_get_node(id: str | None = None, title: str | None = None) -> str:
        """Get the full content of an org-roam node by ID or title."""
        reader, file_access, _embed = get_deps()
        node = await _resolve_node(reader, {'id': id, 'title': title})
        if not node:
            return 'Node not found. Provide id or title.'

        file_path = _q(node.file) or ''
        content = ''
        if file_path and file_access.exists(file_path):
            content = file_access.read_file(file_path)

        tags = [_q(t) or t for t in await reader.get_node_tags(node.id)]

        text = _node_to_text(node, tags)
        if content:
            text += f'### Content\n\n{content}'

        return text

    @mcp.tool(name='roam_backlinks')
    async def roam_backlinks(id: str) -> str:
        """Get all nodes that link TO a given node (backlinks)."""
        reader, file_access, _embed = get_deps()
        backlinks = await reader.get_backlinks(id)

        if not backlinks:
            return f'No backlinks found for {id}.'

        parts = [f'Backlinks to {id} ({len(backlinks)}):\n']
        for bl in backlinks:
            file_path = _q(bl.file) or ''
            excerpt = ''
            if file_path and file_access.exists(file_path):
                content = file_access.read_file(file_path)
                excerpt = get_excerpt(content, 10)
            tags = [_q(t) or t for t in await reader.get_node_tags(bl.id)]
            parts.append(_node_to_text(bl, tags))
            if excerpt:
                parts.append(excerpt)
            parts.append('---\n')

        return '\n'.join(parts)

    @mcp.tool(name='roam_context')
    async def roam_context(
        id: str | None = None,
        title: str | None = None,
        depth: int = 1,
    ) -> str:
        """Build rich context for a node: content + backlinks + linked refs with excerpts. Use this to understand a topic deeply."""
        reader, file_access, _embed = get_deps()
        node = await _resolve_node(reader, {'id': id, 'title': title})
        if not node:
            return 'Node not found. Provide id or title.'

        ctx = await build_context(reader, file_access, node, depth)

        node_title = _q(node.title) or '(untitled)'
        parts = [f'# Context: {node_title}\n']

        parts.append('## Node Content\n')
        parts.append(ctx.content)
        parts.append('')

        if ctx.backlinks:
            parts.append(f'## Backlinks ({len(ctx.backlinks)})\n')
            for bl, excerpt in ctx.backlinks:
                bl_title = _q(bl.title) or '(untitled)'
                parts.append(f'### <- {bl_title}\n')
                parts.append(excerpt)
                parts.append('')

        if ctx.forward_links:
            parts.append(f'## Forward Links ({len(ctx.forward_links)})\n')
            for fl, excerpt in ctx.forward_links:
                fl_title = _q(fl.title) or '(untitled)'
                parts.append(f'### -> {fl_title}\n')
                parts.append(excerpt)
                parts.append('')

        if ctx.tags:
            parts.append(f'## Tags: {", ".join(ctx.tags)}\n')

        parts.append('## Summary\n')
        parts.append(f'- Total nodes in context: {ctx.summary["total_nodes"]}')
        parts.append(f'- Backlinks: {ctx.summary["backlink_count"]}')
        parts.append(f'- Forward links: {ctx.summary["forward_link_count"]}')
        if ctx.summary['tag_cloud']:
            parts.append(f'- Tag cloud: {ctx.summary["tag_cloud"]}')

        return '\n'.join(parts)

    @mcp.tool(name='roam_subgraph')
    async def roam_subgraph(
        id: str | None = None,
        title: str | None = None,
        depth: int = 1,
    ) -> str:
        """Get a subgraph around a node (N-degree neighbors). Useful for analysis."""
        reader, file_access, _embed = get_deps()
        node = await _resolve_node(reader, {'id': id, 'title': title})
        if not node:
            return 'Node not found. Provide id or title.'

        sg = await build_subgraph(reader, file_access, node, depth)

        parts = [f'# Subgraph around: {sg["center_title"]}\n']
        parts.append(f'- Total nodes: {sg["total_nodes"]}')
        parts.append(f'- Total links: {sg["total_links"]}')
        parts.append(f'- Nodes: {sg["node_titles"]}')
        if sg['tag_distribution']:
            parts.append(f'- Tags: {sg["tag_distribution"]}')

        return '\n'.join(parts)

    @mcp.tool(name='roam_tags')
    async def roam_tags(tag: str | None = None) -> str:
        """List all tags and their counts, or get all nodes with a specific tag."""
        reader, file_access, _embed = get_deps()

        if tag:
            nodes = await reader.get_nodes_by_tag(tag)
            if not nodes:
                return f'No nodes found with tag "{tag}".'
            parts = [f'Nodes tagged "{tag}" ({len(nodes)}):\n']
            for node in nodes:
                parts.append(f'- {_q(node.title)} (ID: {_q(node.id)})')
            return '\n'.join(parts)

        tags = await reader.get_tags()
        if not tags:
            return 'No tags found.'
        parts = [f'Tags ({len(tags)}):\n']
        for t, count in tags:
            parts.append(f'- {_q(t)} ({count} nodes)')
        return '\n'.join(parts)

    @mcp.tool(name='roam_recent')
    async def roam_recent(limit: int = 10) -> str:
        """Get recently modified org-roam notes."""
        reader, file_access, _embed = get_deps()
        nodes = await reader.get_recent(limit)

        parts = [f'Recently modified notes ({len(nodes)}):\n']
        for node in nodes:
            tags = [_q(t) or t for t in await reader.get_node_tags(node.id)]
            parts.append(_node_to_text(node, tags))

        return '\n'.join(parts)

    @mcp.tool(name='roam_daily')
    async def roam_daily(date: str | None = None) -> str:
        """Get or create today's (or a specific date's) daily note."""
        reader, file_access, _embed = get_deps()

        if not date:
            date = datetime.now().strftime('%Y-%m-%d')

        node = await reader.get_daily_note(date)
        if node:
            file_path = _q(node.file) or ''
            content = ''
            if file_path and file_access.exists(file_path):
                content = file_access.read_file(file_path)
            return f'# Daily Note: {date}\n\n{content}'

        daily_path = file_access.daily_path(date)
        if file_access.exists(daily_path):
            content = file_access.read_file(daily_path)
            return f'# Daily Note: {date}\n\n{content}'

        return f'No daily note found for {date}.'

    @mcp.tool(name='roam_capture')
    async def roam_capture(title: str, body: str = '', tags: list[str] | None = None) -> str:
        """Create a new org-roam note with title, body, and optional tags."""
        reader, file_access, _embed = get_deps()

        node_id, filepath = await create_note(file_access, reader, title, body, tags or [])

        return f'Created note: {title}\n- ID: {node_id}\n- File: {filepath}'

    @mcp.tool(name='roam_append')
    async def roam_append(id: str, content: str, heading: str | None = None) -> str:
        """Append content to an existing org-roam note."""
        reader, file_access, _embed = get_deps()

        try:
            await append_to_note(file_access, reader, id, content, heading)
            return f'Appended content to node {id}.'
        except ValueError as e:
            return f'Error: {e}'
        except FileNotFoundError as e:
            return f'Error: {e}'

    @mcp.tool(name='roam_research_dump')
    async def roam_research_dump(
        title: str,
        authors: list[str] | None = None,
        abstract: str = '',
        doi: str | None = None,
        url: str | None = None,
        findings: list[str] | None = None,
        source_type: str = 'paper',
        year: int | None = None,
        journal: str | None = None,
        topic: str | None = None,
    ) -> str:
        """Create a structured research note from paper/web data. Works with scite MCP output."""
        reader, file_access, _embed = get_deps()

        topic_node_id = None
        if topic:
            topic_node = await reader.get_node_by_title(topic)
            if topic_node:
                topic_node_id = _q(topic_node.id) or topic_node.id

        data = ResearchNoteData(
            title=title,
            authors=authors or [],
            abstract=abstract,
            doi=doi,
            url=url,
            findings=findings or [],
            source_type=source_type,
            year=year,
            journal=journal,
        )

        node_id, filepath = await research_dump(file_access, reader, data, topic_node_id)

        result = f'Created research note: {data.title}\n- ID: {node_id}\n- File: {filepath}'
        if topic_node_id:
            result += f'\n- Linked to topic: {topic}'

        return result

    @mcp.tool(name='roam_enhance')
    async def roam_enhance(
        id: str | None = None,
        title: str | None = None,
        dry_run: bool = False,
    ) -> str:
        """Use a local LLM (Ollama) to generate an enhanced summary for a note.
        Updates the Summary section in-place. Requires Ollama running locally."""
        reader, file_access, _embed = get_deps()
        from mcp_roam.llm import enhance_content

        node = await _resolve_node(reader, {'id': id, 'title': title})
        if not node:
            return 'Node not found. Provide id or title.'

        file_path = _q(node.file) or ''
        if not file_path or not file_access.exists(file_path):
            return 'File not found for this node.'

        content = file_access.read_file(file_path)
        node_title = _q(node.title) or '(untitled)'

        import re as _re
        body = content
        m = _re.search(r'\* Summary\n(.+?)(?=\n\* |\Z)', content, _re.DOTALL)
        if m:
            body = m.group(1).strip()
        elif '* Full Transcript' in content:
            body = content.split('* Full Transcript')[1]

        result = enhance_content(body, node_title)

        if not result.summary:
            return 'LLM enhancement failed. Is Ollama running?'

        if dry_run:
            return f'[DRY RUN] Enhanced Summary for {node_title}:\n\n{result.summary}'

        new_section = f'* Summary\n{result.summary}\n'
        if '* Summary' in content:
            new_content = _re.sub(
                r'\* Summary\n.+?(?=\n\* )',
                new_section,
                content,
                count=1,
                flags=_re.DOTALL,
            )
        else:
            insert = content.find('\n* ')
            new_content = content[:insert] + '\n' + new_section + content[insert:]

        file_access.write_file(file_path, new_content)
        return f'Enhanced summary for: {node_title}'

    # ========================================================================
    # Semantic search tools (embeddings)
    # ========================================================================

    @mcp.tool(name='roam_index')
    async def roam_index(
        id: str | None = None,
        title: str | None = None,
        all_nodes: bool = False,
        limit: int = 50,
        max_level: int = 1,
    ) -> str:
        """Index org-roam notes for semantic search. Parses subtrees and embeds them.
        Pass all_nodes=True to index everything (up to limit). Otherwise index a single note."""
        reader, file_access, embed_repo = get_deps()

        if embed_repo is None:
            return 'Semantic search disabled: sqlite-vec not installed. Run: pip install sqlite-vec'

        if all_nodes:
            from mcp_roam.repo import SqliteRepo
            with embed_repo._get_connection() as conn:
                cursor = conn.execute(
                    'SELECT id, file FROM nodes WHERE file IS NOT NULL'
                )
                all_rows = cursor.fetchall()
            indexed = 0
            skipped = 0
            errors = 0
            for row in all_rows[:limit]:
                node = await reader.get_node_by_id(row['id'].strip('"'))
                if not node:
                    continue
                file_path = _q(node.file) or ''
                if not file_path or not file_access.exists(file_path):
                    continue
                content = file_access.read_file(file_path)
                try:
                    result = embed_repo.index_node(
                        _q(node.id) or node.id,
                        content,
                        max_level,
                    )
                    if result['skipped']:
                        skipped += 1
                    else:
                        indexed += result['embedded']
                except Exception as e:
                    errors += 1
                    print(f'[roam_index] Error on {file_path}: {e}', file=__import__('sys').stderr)
            return (
                f'Indexed {indexed} units from {len(all_rows[:limit])} nodes '
                f'({skipped} already up-to-date, {errors} errors).'
            )

        node = await _resolve_node(reader, {'id': id, 'title': title})
        if not node:
            return 'Node not found. Provide id or title.'

        file_path = _q(node.file) or ''
        if not file_path or not file_access.exists(file_path):
            return 'File not found for this node.'

        content = file_access.read_file(file_path)
        result = embed_repo.index_node(
            _q(node.id) or node.id,
            content,
            max_level,
        )

        node_title = _q(node.title) or '(untitled)'
        if result['skipped']:
            return f'{node_title}: already indexed (up-to-date).'

        return (
            f'{node_title}: indexed {result["embedded"]}/{result["units"]} units '
            f'(type: structural segmentation, max_level={max_level}).'
        )

    @mcp.tool(name='roam_semantic_search')
    async def roam_semantic_search(
        query: str,
        k: int = 10,
        merge: bool = True,
        unit_type: str | None = None,
        rerank: bool = True,
    ) -> str:
        """Semantic search across indexed notes by meaning, not keywords.
        Returns matching passages with relevance scores.
        Set merge=False to see individual units; merge=True groups by note.
        Set rerank=False for faster search without cross-encoder reranking."""
        reader, file_access, embed_repo = get_deps()

        if embed_repo is None:
            return 'Semantic search disabled: sqlite-vec not installed.'

        fetch_k = k * 2 if (merge and rerank) else (k * 3 if merge else k)
        results = embed_repo.search(
            query, k=fetch_k, unit_type=unit_type, rerank=rerank,
        )

        if not results:
            return (
                f'No semantic results for "{query}". '
                f'Make sure notes are indexed (roam_index) and Ollama is running.'
            )

        def _score_str(r: dict) -> str:
            if 'rerank_score' in r:
                return f'rerank: {r["rerank_score"]:.3f}'
            return f'distance: {r["distance"]:.3f}'

        def _best_unit(units: list[dict]) -> dict:
            if 'rerank_score' in units[0]:
                return max(units, key=lambda u: u['rerank_score'])
            return min(units, key=lambda u: u['distance'])

        node_cache: dict[str, Any] = {}

        async def _resolve(nid: str) -> Any | None:
            if nid not in node_cache:
                node_cache[nid] = await reader.get_node_by_id(nid)
            return node_cache[nid]

        if merge:
            seen_nodes: dict[str, list[dict]] = {}
            for r in results:
                seen_nodes.setdefault(r['node_id'], []).append(r)

            parts = [f'Semantic search: "{query}" — {len(seen_nodes)} notes matched\n']
            for node_id, units in list(seen_nodes.items())[:k]:
                node = await _resolve(node_id)
                node_title = _q(node.title) if node else node_id
                file_path = _q(node.file) if node else ''
                best = _best_unit(units)
                parts.append(f'## {node_title} ({_score_str(best)})\n')
                parts.append(f'> [{best["heading_path"]}] {best["text"][:300]}')
                parts.append(f'\nID: {node_id}')
                if file_path:
                    parts.append(f'File: {file_path}')
                parts.append('---\n')
            return '\n'.join(parts)

        parts = [f'Semantic search: "{query}" — top {len(results)} units\n']
        for r in results:
            node = await _resolve(r['node_id'])
            node_title = _q(node.title) if node else r['node_id']
            file_path = _q(node.file) if node else ''
            parts.append(
                f'## {node_title} / [{r["unit_type"]}] {r["heading_path"]} '
                f'({_score_str(r)})\n'
            )
            parts.append(r['text'][:300])
            parts.append(f'\nID: {r["node_id"]}')
            if file_path:
                parts.append(f'File: {file_path}')
            parts.append('---\n')
        return '\n'.join(parts)

    @mcp.tool(name='roam_extract_claims')
    async def roam_extract_claims(
        id: str | None = None,
        title: str | None = None,
        max_claims: int = 10,
        dry_run: bool = False,
    ) -> str:
        """Extract atomic claims from a note using Ollama LLM, then embed them for semantic search.
        Best for dense notes (transcripts, papers). Requires Ollama running locally."""
        reader, file_access, embed_repo = get_deps()

        if embed_repo is None:
            return 'Semantic search disabled: sqlite-vec not installed.'

        from mcp_roam.llm import extract_claims

        node = await _resolve_node(reader, {'id': id, 'title': title})
        if not node:
            return 'Node not found. Provide id or title.'

        file_path = _q(node.file) or ''
        if not file_path or not file_access.exists(file_path):
            return 'File not found for this node.'

        content = file_access.read_file(file_path)
        node_title = _q(node.title) or '(untitled)'

        result = extract_claims(content, node_title, max_claims)

        if not result.claims:
            return f'No claims extracted from {node_title}. Is Ollama running?'

        if dry_run:
            lines = [f'[DRY RUN] {len(result.claims)} claims for {node_title}:\n']
            for i, claim in enumerate(result.claims, 1):
                lines.append(f'{i}. {claim}')
            return '\n'.join(lines)

        node_id_clean = _q(node.id) or node.id
        added = embed_repo.add_claims(node_id_clean, result.claims)

        return (
            f'Extracted and embedded {added}/{len(result.claims)} claims for {node_title}.\n'
            f'Claims are now searchable via roam_semantic_search.\n\n'
            + '\n'.join(f'{i}. {c}' for i, c in enumerate(result.claims[:5], 1))
            + (f'\n... and {len(result.claims) - 5} more.' if len(result.claims) > 5 else '')
        )

    @mcp.tool(name='roam_index_stats')
    async def roam_index_stats() -> str:
        """Show statistics about the embedding index."""
        reader, file_access, embed_repo = get_deps()

        if embed_repo is None:
            return 'Semantic search disabled: sqlite-vec not installed.'

        stats = embed_repo.get_stats()

        parts = ['Embedding Index Stats\n']
        parts.append(f'- Total units: {stats["total_units"]}')
        parts.append(f'- Indexed nodes: {stats["total_nodes"]}')

        if stats['by_type']:
            parts.append('\nBy type:')
            for utype, count in sorted(stats['by_type'].items(), key=lambda x: -x[1]):
                parts.append(f'  {utype}: {count}')

        return '\n'.join(parts)
