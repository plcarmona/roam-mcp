"""Domain types and org-mode parsing/serialization."""

from dataclasses import dataclass, field
from datetime import datetime
from re import Match
from typing import Any
from uuid import uuid4
import re


# ============================================================================
# DB Entity Types (from org-roam.db schema)
# ============================================================================

@dataclass(frozen=True)
class RoamNode:
    """A node in the org-roam graph (from 'nodes' table)."""
    id: str
    file: str
    level: int
    pos: int
    todo: str | None
    priority: str | None
    scheduled: str | None
    deadline: str | None
    title: str | None
    properties: str | None
    olp: str | None


@dataclass(frozen=True)
class RoamLink:
    """A link between nodes (from 'links' table)."""
    pos: int
    source: str
    dest: str
    type: str
    properties: str | None


@dataclass(frozen=True)
class RoamTag:
    """A tag on a node (from 'tags' table)."""
    node_id: str
    tag: str | None


@dataclass(frozen=True)
class RoamFile:
    """A file in the org-roam directory (from 'files' table)."""
    file: str
    title: str | None
    hash: str
    atime: int
    mtime: int


@dataclass(frozen=True)
class RoamAlias:
    """An alias for a node (from 'aliases' table)."""
    node_id: str
    alias: str | None


@dataclass(frozen=True)
class RoamRef:
    """A reference (from 'refs' table)."""
    node_id: str
    ref: str
    type: str


@dataclass(frozen=True)
class RoamCitation:
    """A citation (from 'citations' table)."""
    node_id: str
    cite_key: str
    pos: int
    properties: str | None


# ============================================================================
# Application Types
# ============================================================================

@dataclass(frozen=True)
class OrgLink:
    """Parsed org-mode link."""
    type: str  # 'id', 'https', 'http', 'file', etc.
    target: str
    description: str | None = None
    raw: str | None = None


@dataclass
class SearchResult:
    """A search result with context."""
    node: RoamNode
    tags: list[str]
    excerpt: str  # First 50 lines as preview


@dataclass
class NodeContext:
    """Rich context for a node: content + backlinks + linked refs."""
    node: RoamNode
    content: str
    backlinks: list[tuple[RoamNode, str]]  # (node, excerpt)
    forward_links: list[tuple[RoamNode, str]]  # (node, excerpt)
    linked_refs: list[RoamRef]
    tags: list[str]
    summary: dict[str, Any]  # node_count, tag_cloud, link_density


@dataclass
class Subgraph:
    """A subgraph around a node."""
    center: RoamNode
    nodes: set[RoamNode]
    links: set[tuple[str, str, str]]  # (source_id, dest_id, type)
    tags: dict[str, int]


@dataclass(frozen=True)
class ResearchNoteData:
    """Data for creating a research note."""
    title: str
    authors: list[str]
    abstract: str
    doi: str | None
    url: str | None
    findings: list[str]
    source_type: str  # 'paper', 'web', 'patent', etc.
    year: int | None = None
    journal: str | None = None


@dataclass(frozen=True)
class EmbedUnit:
    """A semantic unit extracted from a note, suitable for embedding."""
    node_id: str
    heading_path: str       # e.g. "Summary" or "Key Concepts > Carl Jung"
    text: str
    unit_type: str          # 'heading' | 'summary' | 'concept' | 'explicit' | 'claim'
    pos: int                # line number in source file


# ============================================================================
# Org-mode Parsing
# ============================================================================

PROPERTIES_RE = re.compile(
    r'^:PROPERTIES:\s*$'
    r'(.*?)'
    r'^:END:\s*$',
    re.MULTILINE | re.DOTALL
)

ORG_LINK_RE = re.compile(
    r'\[\[(https?://[^\]]+)\](?:\[([^\]]+)\])?\]|'
    r'\[\[id:([^\]]+)\](?:\[([^\]]+)\])?\]|'
    r'\[\[file:([^\]]+)\](?:\[([^\]]+)\])?\]|'
    r'\[\[([^\]]+)\](?:\[([^\]]+)\])?\]'
)


# Matches [[title]] or [[title][desc]] links that are NOT already [[id:...]], [[https://...]], or [[file:...]]
BARE_LINK_RE = re.compile(
    r"""\[\[(?!id:|https?:|file:)([^\]]+)\](?:\[([^\]]+)\])?\]"""
)

NODE_ID_RE = re.compile(r'^:ID:\s+([0-9a-f-]+)\s*$', re.MULTILINE)


def parse_properties_drawer(text: str) -> dict[str, str]:
    """Extract properties from :PROPERTIES: drawer."""
    match = PROPERTIES_RE.search(text)
    if not match:
        return {}

    drawer = match.group(1)
    props: dict[str, str] = {}

    # Match properties: :NAME: value
    # Must be at start of line (with MULTILINE), after optional whitespace
    for prop_match in re.finditer(r'^\s*:([A-Za-z0-9_-]+):\s+(.+?)\s*$', drawer, re.MULTILINE):
        name = prop_match.group(1).lower()
        value = prop_match.group(2).strip()
        props[name] = value

    return props


def parse_org_links(text: str) -> list[OrgLink]:
    """Parse all org-mode links from text."""
    links: list[OrgLink] = []

    for match in ORG_LINK_RE.finditer(text):
        raw = match.group(0)
        url = match.group(1)  # http/https
        url_id = match.group(3)  # id:
        file_path = match.group(5)  # file:
        fuzzy = match.group(7)  # fuzzy

        description = match.group(2) or match.group(4) or match.group(6) or match.group(8)

        if url:
            links.append(OrgLink(type='https', target=url, description=description, raw=raw))
        elif url_id:
            links.append(OrgLink(type='id', target=url_id, description=description, raw=raw))
        elif file_path:
            links.append(OrgLink(type='file', target=file_path, description=description, raw=raw))
        elif fuzzy:
            links.append(OrgLink(type='fuzzy', target=fuzzy, description=description, raw=raw))

    return links


def extract_node_id(content: str) -> str | None:
    """Extract node ID from :ID: property."""
    match = NODE_ID_RE.search(content)
    return match.group(1) if match else None


def slug_from_title(title: str) -> str:
    """Convert title to org-roam slug format."""
    # Lowercase, replace spaces with underscores, remove non-alphanumeric
    slug = title.lower()
    slug = re.sub(r'[^a-z0-9_\s]', '', slug)
    slug = re.sub(r'\s+', '_', slug)
    return slug[:100]  # Reasonable max length


def generate_filename(title: str) -> str:
    """Generate org-roam filename: YYYYMMDDHHMMSS-slug.org"""
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    slug = slug_from_title(title)
    return f"{timestamp}-{slug}.org"


def format_properties_drawer(node_id: str) -> str:
    """Format :PROPERTIES: drawer with ID."""
    return f""":PROPERTIES:
:ID:       {node_id}
:END:
"""


def format_org_file(title: str, node_id: str, body: str = '', tags: list[str] | None = None) -> str:
    """Format a complete org-roam file."""
    lines: list[str] = [
        format_properties_drawer(node_id),
        f"#+title: {title}",
    ]

    if tags:
        tag_str = ':'.join(tags)
        lines.append(f"#+filetags: :{tag_str}:")

    lines.append('')

    if body:
        lines.append(body)

    return '\n'.join(lines)


def format_org_link(target: str, description: str | None = None, link_type: str = 'id') -> str:
    """Format an org-mode link."""
    if link_type == 'id':
        prefix = f"id:{target}"
    elif link_type == 'file':
        prefix = f"file:{target}"
    elif link_type in ('http', 'https'):
        prefix = target
    else:
        prefix = target

    if description:
        return f"[[{prefix}][{description}]]"
    return f"[[{prefix}]]"


def parse_headings(text: str) -> list[tuple[int, str, str | None]]:
    """Parse org headings: (level, title, todo)."""
    headings: list[tuple[int, str, str | None]] = []
    for line in text.split('\n'):
        if line.startswith('*'):
            level = 0
            while level < len(line) and line[level] == '*':
                level += 1

            if level == len(line):
                continue  # Just asterisks

            rest = line[level:].strip()

            # Check for TODO keyword
            todo = None
            for keyword in ('TODO', 'DONE', 'WAITING', 'CANCELLED'):
                if rest.startswith(keyword + ' '):
                    todo = keyword
                    rest = rest[len(keyword) + 1:].strip()
                    break

            headings.append((level, rest, todo))

    return headings


def parse_org_tags(line: str) -> list[str]:
    """Parse tags from end of org heading: :tag1:tag2:"""
    # Tags are at end of line, like :tag1:tag2:
    match = re.search(r'((:\w+:)+)$', line)
    if not match:
        return []

    tag_str = match.group(1)
    # Split by : and filter empty strings
    return [t for t in tag_str.split(':') if t]


def get_excerpt(content: str, max_lines: int = 50) -> str:
    """Get first N lines as excerpt."""
    lines = content.split('\n')
    excerpt = '\n'.join(lines[:max_lines])
    if len(lines) > max_lines:
        excerpt += '\n... (truncated)'
    return excerpt