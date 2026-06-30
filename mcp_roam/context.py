"""Context builder — assembles rich context from the org-roam graph."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from mcp_roam.domain import (
    NodeContext,
    RoamNode,
    RoamRef,
    get_excerpt,
)
from mcp_roam.interfaces import FileAccess, RoamReader

MAX_DEPTH = 3
MAX_NODES = 30
EXCERPT_LINES = 50


def _strip_quotes(s: str | None) -> str | None:
    """Strip surrounding double quotes from a DB value."""
    if s and s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    return s


async def build_context(
    reader: RoamReader,
    file_access: FileAccess,
    node: RoamNode,
    depth: int = 1,
) -> NodeContext:
    """Build rich context for a node.

    Args:
        reader: Database reader.
        file_access: File system access.
        node: The target node.
        depth: How many hops to follow (1=backlinks only, 2=one more hop, max 3).

    Returns:
        NodeContext with content, backlinks, forward links, refs, tags, and summary.
    """
    depth = min(depth, MAX_DEPTH)

    # Load node content
    file_path = _strip_quotes(node.file) or ''
    content = ''
    if file_path and file_access.exists(file_path):
        content = file_access.read_file(file_path)

    # Load tags
    tags = await reader.get_node_tags(node.id)
    tags = [_strip_quotes(t) or t for t in tags]

    # Load backlinks with excerpts
    backlink_nodes = await reader.get_backlinks(node.id)
    backlinks: list[tuple[RoamNode, str]] = []
    for bl in backlink_nodes:
        bl_path = _strip_quotes(bl.file) or ''
        bl_content = ''
        if bl_path and file_access.exists(bl_path):
            bl_content = file_access.read_file(bl_path)
        backlinks.append((bl, get_excerpt(bl_content, EXCERPT_LINES)))

    # Load forward links with excerpts
    forward_nodes = await reader.get_forward_links(node.id)
    forward_links: list[tuple[RoamNode, str]] = []
    for fl in forward_nodes:
        fl_path = _strip_quotes(fl.file) or ''
        fl_content = ''
        if fl_path and file_access.exists(fl_path):
            fl_content = file_access.read_file(fl_path)
        forward_links.append((fl, get_excerpt(fl_content, EXCERPT_LINES)))

    # Load refs
    refs = await reader.get_node_refs(node.id)

    # Build summary
    all_nodes = {node} | {n for n, _ in backlinks} | {n for n, _ in forward_links}
    all_tags: list[str] = list(tags)
    for n, _ in backlinks:
        node_tags = await reader.get_node_tags(n.id)
        all_tags.extend(_strip_quotes(t) or t for t in node_tags)

    tag_cloud = dict(Counter(all_tags).most_common(20))

    summary = {
        'total_nodes': len(all_nodes),
        'backlink_count': len(backlinks),
        'forward_link_count': len(forward_links),
        'ref_count': len(refs),
        'tag_cloud': tag_cloud,
    }

    return NodeContext(
        node=node,
        content=content,
        backlinks=backlinks,
        forward_links=forward_links,
        linked_refs=refs,
        tags=tags,
        summary=summary,
    )


async def build_subgraph(
    reader: RoamReader,
    file_access: FileAccess,
    node: RoamNode,
    depth: int = 1,
) -> dict:
    """Build a subgraph summary around a node (for analysis).

    Returns a dict with nodes, links, and tag distribution.
    """
    from mcp_roam.domain import Subgraph

    depth = min(depth, MAX_DEPTH)

    visited: set[str] = set()
    nodes: set[RoamNode] = set()
    links: set[tuple[str, str, str]] = set()  # (source_id, dest_id, type)
    all_tags: list[str] = []

    async def _walk(current: RoamNode, remaining: int):
        cid = _strip_quotes(current.id) or current.id
        if cid in visited or len(visited) >= MAX_NODES:
            return
        visited.add(cid)
        nodes.add(current)

        # Collect tags
        node_tags = await reader.get_node_tags(current.id)
        all_tags.extend(_strip_quotes(t) or t for t in node_tags)

        if remaining <= 0:
            return

        # Walk backlinks
        for bl in await reader.get_backlinks(current.id):
            bl_id = _strip_quotes(bl.id) or bl.id
            links.add((bl_id, cid, 'backlink'))
            await _walk(bl, remaining - 1)

        # Walk forward links
        for fl in await reader.get_forward_links(current.id):
            fl_id = _strip_quotes(fl.id) or fl.id
            links.add((cid, fl_id, 'forward'))
            await _walk(fl, remaining - 1)

    await _walk(node, depth)

    tag_dist = dict(Counter(all_tags).most_common(20))

    return {
        'center_title': _strip_quotes(node.title) or node.title,
        'total_nodes': len(nodes),
        'total_links': len(links),
        'tag_distribution': tag_dist,
        'node_titles': sorted(_strip_quotes(n.title) or n.title or '' for n in nodes),
        'links': [(s, d, t) for s, d, t in links],
    }
