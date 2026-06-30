"""Note capture and append operations."""

from __future__ import annotations

import re
from datetime import datetime
from re import Match
from uuid import uuid4

from mcp_roam.domain import (
    BARE_LINK_RE,
    format_org_file,
    generate_filename,
    format_org_link,
    slug_from_title,
)
from mcp_roam.interfaces import FileAccess, RoamReader


def _strip_quotes(s: str | None) -> str | None:
    if s and s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    return s


def _slug_to_search_terms(slug: str) -> list[str]:
    """Convert a slug to progressively broader search terms.

    'hsi_defect_detection_fruit' -> ['hsi defect detection fruit', 'hsi defect detection', 'hsi defect']
    """
    words = slug.split('_')
    # Filter empty parts (from consecutive underscores)
    words = [w for w in words if w]
    terms = []
    for n in range(len(words), 1, -1):
        terms.append(' '.join(words[:n]))
    return terms


def _slug_match(target_slug: str, candidate_slug: str) -> bool:
    """Check if two slugs refer to the same node.

    Handles cases where target omits small words (in, of, the, a, an, and)
    that exist in the full title, or has minor spelling differences.
    """
    if target_slug == candidate_slug:
        return True
    target_words = set(target_slug.split('_'))
    candidate_words = set(candidate_slug.split('_'))
    # Remove stop words and compare content words
    stop_words = {'in', 'of', 'the', 'a', 'an', 'and', 'for', 'to', 'with', 'on', 'at'}
    target_content = target_words - stop_words
    candidate_content = candidate_words - stop_words
    if target_content and candidate_content and target_content == candidate_content:
        return True
    # Subset match (target is subset of candidate words)
    if target_words and target_words.issubset(candidate_words):
        return True
    return False


async def resolve_body_links(body: str, reader: RoamReader) -> str:
    """Find all [[title]] links in body and resolve them to [[id:UUID][title]].

    Scans body for bare [[target]] or [[target][desc]] links (not already [[id:...]]),
    looks up each target in the org-roam database by:
      1. Exact title match (case-insensitive)
      2. Alias match
      3. Slug match (search with progressive terms, fuzzy slug comparison)

    Unresolved links are left as-is.
    """
    matches = list(BARE_LINK_RE.finditer(body))
    if not matches:
        return body

    # Collect unique targets
    targets = list({m.group(1) for m in matches})

    # Resolve each target to (node_id, node_title)
    resolved: dict[str, tuple[str, str]] = {}
    for target in targets:
        node = await _lookup_target(target, reader)
        if node:
            nid = _strip_quotes(node.id) or node.id
            ntitle = _strip_quotes(node.title) or node.title
            resolved[target] = (nid, ntitle)

    # Replace all matches
    def _replace(m: Match) -> str:
        target = m.group(1)
        description = m.group(2)
        if target not in resolved:
            return m.group(0)  # Leave unresolved as-is
        node_id, node_title = resolved[target]
        display = description or node_title
        return f'[[id:{node_id}][{display}]]'

    return BARE_LINK_RE.sub(_replace, body)


async def _lookup_target(target: str, reader: RoamReader):
    """Try to find a node by title, alias, then by slug-based search."""
    # 1. Try exact title / alias lookup (as-is first)
    node = await reader.get_node_by_title(target)
    if node:
        return node

    # 2. Try case variations for short/simple targets
    #    e.g. 'fx10' -> try 'Fx10', 'FX10'; 'vnir' -> try 'Vnir', 'VNIR'
    if '_' not in target:
        variations = [
            target.capitalize(),    # fx10 -> Fxl0
            target.upper(),         # fx10 -> FX10
            target.title(),         # vnir -> Vnir
        ]
        for v in variations:
            node = await reader.get_node_by_title(v)
            if node:
                return node

    # 3. Slug-based search: try progressively shorter search terms
    target_slug = slug_from_title(target)
    search_terms = _slug_to_search_terms(target_slug)

    for term in search_terms:
        candidates = await reader.search_nodes(term, limit=20)
        for candidate in candidates:
            ctitle = _strip_quotes(candidate.title) or ''
            candidate_slug = slug_from_title(ctitle)
            if _slug_match(target_slug, candidate_slug):
                return candidate

    return None


async def create_note(
    file_access: FileAccess,
    reader: RoamReader,
    title: str,
    body: str = '',
    tags: list[str] | None = None,
) -> tuple[str, str]:
    """Create a new org-roam note.

    Args:
        file_access: File system access.
        reader: Database reader.
        title: Note title.
        body: Note body content.
        tags: Tags to add.

    Returns:
        (node_id, filepath) tuple.
    """
    # Resolve [[title]] -> [[id:UUID][title]] in body
    resolved_body = await resolve_body_links(body, reader)

    node_id = str(uuid4())
    filename = generate_filename(title)
    filepath = file_access.resolve_path(filename)
    content = format_org_file(title, node_id, resolved_body, tags)

    file_access.write_file(filepath, content)

    return node_id, str(filepath)


async def append_to_note(
    file_access: FileAccess,
    reader: RoamReader,
    node_id: str,
    content: str,
    heading: str | None = None,
) -> None:
    """Append content to an existing note.

    Args:
        file_access: File system access.
        reader: Database reader.
        node_id: The node ID to append to.
        content: Content to append.
        heading: Optional heading under which to append.
    """
    # Resolve [[title]] -> [[id:UUID][title]] in content
    resolved_content = await resolve_body_links(content, reader)

    # Normalize ID
    nid = _strip_quotes(node_id) or node_id
    if not nid.startswith('"'):
        nid = f'"{nid}"'

    node = await reader.get_node_by_id(nid)
    if not node:
        raise ValueError(f"Node not found: {node_id}")

    file_path = _strip_quotes(node.file)
    if not file_path or not file_access.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    current = file_access.read_file(file_path)

    timestamp = datetime.now().strftime('* %Y-%m-%d %H:%M')

    if heading:
        append_block = f'\n{timestamp}\n** {heading}\n{resolved_content}\n'
    else:
        append_block = f'\n{timestamp}\n{resolved_content}\n'

    file_access.write_file(file_path, current.rstrip() + '\n' + append_block)
