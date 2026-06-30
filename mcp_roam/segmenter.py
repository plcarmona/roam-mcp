"""Org-aware segmentation — splits notes into semantic units for embedding.

Replaces naive character-based chunking with structure-aware parsing:
- Subtrees delimited by org headings (*, **, ***)
- Explicit #+begin_embed / #+end_embed blocks
- Classification by known section types (summary, concept, etc.)
"""

from __future__ import annotations

import re

from mcp_roam.domain import EmbedUnit


EMBED_BEGIN_RE = re.compile(
    r'^#\+begin_embed\b(.*?)$',
    re.IGNORECASE,
)
EMBED_END_RE = re.compile(r'^#\+end_embed\s*$', re.IGNORECASE)
HEADING_RE = re.compile(r'^(\*+)\s+(.*)$')
PROPERTIES_END_RE = re.compile(r'^:END:\s*$')

MIN_TEXT_LENGTH = 20    # skip units too short to be meaningful
SMALL_UNIT_SIZE = 120   # units smaller than this are candidates for merging
MERGE_BATCH_SIZE = 800  # target merged batch size (chars)

SKIP_SECTIONS = {
    'full transcript',
    'transcript',
    'raw notes',
}


def _strip_heading_markup(line: str) -> str:
    """Remove TODO keyword and tags from a heading title."""
    rest = line.strip()
    for kw in ('TODO', 'DONE', 'WAITING', 'CANCELLED'):
        if rest.startswith(kw + ' '):
            rest = rest[len(kw) + 1:].strip()
            break
    rest = re.sub(r'(?::\w+:)+\s*$', '', rest)
    return rest.strip()


def _classify_unit(heading_path: str) -> str:
    """Infer unit type from heading path."""
    lower = heading_path.lower()
    if lower == 'summary':
        return 'summary'
    if 'key concept' in lower or lower.startswith('concepts'):
        return 'concept'
    if 'timestamp' in lower:
        return 'timestamp'
    if 'theme' in lower:
        return 'theme'
    if 'metadata' in lower:
        return 'metadata'
    return 'heading'


def _skip_line(line: str, in_properties: bool) -> tuple[bool, bool]:
    """Decide if a line should be skipped (metadata noise).

    Returns (skip, still_in_properties).
    """
    stripped = line.strip()

    if in_properties:
        if PROPERTIES_END_RE.match(stripped):
            return True, False
        return True, True

    if stripped == ':PROPERTIES:':
        return True, True

    if stripped.startswith('#+title:') or stripped.startswith('#+TITLE:'):
        return True, False
    if stripped.startswith('#+filetags:') or stripped.startswith('#+FILETAGS:'):
        return True, False

    return False, False


def parse_subtrees(
    content: str,
    node_id: str,
    max_level: int = 1,
) -> list[EmbedUnit]:
    """Split org content into embedding units by heading structure.

    Each unit is the content under a heading, up to the next heading at
    the same or higher level. Top-level preamble (before any heading)
    is captured as a 'preamble' unit.

    Args:
        content: Raw org-mode file content.
        node_id: The org-roam node ID this content belongs to.
        max_level: Deepest heading level to segment (1=*, 2=**, etc.).

    Returns:
        List of EmbedUnit, one per meaningful subtree.
    """
    lines = content.split('\n')
    units: list[EmbedUnit] = []

    # Heading stack: [(level, title, line_num)]
    stack: list[tuple[int, str, int]] = []
    # Current accumulation: (heading_path, start_line, collected_lines)
    current_path: str = ''
    current_start: int = 0
    current_lines: list[str] = []
    in_properties = False
    in_embed = False
    embed_args = ''
    embed_start = 0
    embed_lines: list[str] = []
    skip_until_level: int = 0  # if >0, skip lines until heading at this level or higher

    def _heading_path_from_stack() -> str:
        return ' > '.join(title for _, title, _ in stack)

    def _flush_current():
        text = '\n'.join(current_lines).strip()
        if len(text) >= MIN_TEXT_LENGTH:
            path = current_path or '(preamble)'
            units.append(EmbedUnit(
                node_id=node_id,
                heading_path=path,
                text=text,
                unit_type=_classify_unit(path),
                pos=current_start,
            ))

    for i, line in enumerate(lines):
        # Handle embed blocks first — they take precedence
        if in_embed:
            if EMBED_END_RE.match(line.strip()):
                text = '\n'.join(embed_lines).strip()
                if text:
                    units.append(EmbedUnit(
                        node_id=node_id,
                        heading_path=_heading_path_from_stack() or '(root)',
                        text=text,
                        unit_type='explicit',
                        pos=embed_start,
                    ))
                in_embed = False
                embed_lines = []
            else:
                embed_lines.append(line)
            continue

        embed_match = EMBED_BEGIN_RE.match(line.strip())
        if embed_match:
            in_embed = True
            embed_args = embed_match.group(1)
            embed_start = i
            embed_lines = []
            continue

        # Check for heading
        heading_match = HEADING_RE.match(line)
        if heading_match:
            stars = heading_match.group(1)
            level = len(stars)
            title = _strip_heading_markup(heading_match.group(2))

            # Check if we're in a skip section
            if skip_until_level and level <= skip_until_level:
                skip_until_level = 0
            elif skip_until_level:
                continue

            # Check if this section should be skipped
            if level == 1 and title.lower() in SKIP_SECTIONS:
                _flush_current()
                current_path = ''
                current_lines = []
                skip_until_level = 1
                continue

            # Only flush + start new unit for headings at or below max_level
            if level <= max_level:
                _flush_current()
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title, i))
                current_path = _heading_path_from_stack()
                current_start = i
                current_lines = []
                in_properties = False
            else:
                # Nested too deep — accumulate into parent
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title, i))
                current_lines.append(line)
            continue

        # Skip lines in skipped sections
        if skip_until_level:
            continue

        # Skip metadata noise
        skip, in_properties = _skip_line(line, in_properties)
        if skip:
            continue

        current_lines.append(line)

    _flush_current()

    return units


def parse_embed_blocks(
    content: str,
    node_id: str,
) -> list[EmbedUnit]:
    """Extract explicit #+begin_embed blocks from org content.

    Supports optional args on the begin line:
        #+begin_embed :type claim :tags HSI,agriculture

    The :type arg overrides the unit_type (default 'explicit').
    """
    lines = content.split('\n')
    units: list[EmbedUnit] = []
    in_embed = False
    embed_type = 'explicit'
    embed_start = 0
    embed_lines: list[str] = []
    heading_stack: list[tuple[int, str]] = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        heading_match = HEADING_RE.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            title = _strip_heading_markup(heading_match.group(2))
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))

        if in_embed:
            if EMBED_END_RE.match(stripped):
                text = '\n'.join(embed_lines).strip()
                if text:
                    path = ' > '.join(t for _, t in heading_stack) or '(root)'
                    units.append(EmbedUnit(
                        node_id=node_id,
                        heading_path=path,
                        text=text,
                        unit_type=embed_type,
                        pos=embed_start,
                    ))
                in_embed = False
                embed_lines = []
            else:
                embed_lines.append(line)
            continue

        begin_match = EMBED_BEGIN_RE.match(stripped)
        if begin_match:
            in_embed = True
            embed_start = i
            embed_lines = []
            args = begin_match.group(1)
            type_match = re.search(r':type\s+(\S+)', args)
            embed_type = type_match.group(1) if type_match else 'explicit'

    return units


def _parent_path(heading_path: str) -> str:
    """Get parent heading path (everything before last ' > ')."""
    if ' > ' in heading_path:
        return heading_path.rsplit(' > ', 1)[0]
    return ''


def _merge_small_siblings(units: list[EmbedUnit]) -> list[EmbedUnit]:
    """Merge consecutive small units that share the same parent heading.

    This prevents creating hundreds of tiny units for sections like
    'Key Concepts' where each concept is a one-liner.
    """
    if not units:
        return units

    merged: list[EmbedUnit] = []
    batch_parent: str = ''
    batch_texts: list[str] = []
    batch_headings: list[str] = []
    batch_pos: int = 0

    def _flush_batch():
        if not batch_texts:
            return
        combined = '\n'.join(batch_texts)
        if len(combined.strip()) >= MIN_TEXT_LENGTH:
            heading = f'{batch_parent} ({len(batch_texts)} items)' if len(batch_texts) > 1 else batch_headings[0]
            utype = _classify_unit(batch_parent or batch_headings[0])
            merged.append(EmbedUnit(
                node_id=units[0].node_id,
                heading_path=heading,
                text=combined,
                unit_type=utype,
                pos=batch_pos,
            ))

    for unit in units:
        parent = _parent_path(unit.heading_path)
        is_small = len(unit.text) < SMALL_UNIT_SIZE

        if is_small and parent == batch_parent:
            batch_texts.append(f'** {unit.heading_path.split(" > ")[-1]}\n{unit.text}')
            batch_headings.append(unit.heading_path)

            if sum(len(t) for t in batch_texts) >= MERGE_BATCH_SIZE:
                _flush_batch()
                batch_texts = []
                batch_headings = []
        else:
            _flush_batch()
            batch_texts = []
            batch_headings = []

            if is_small and parent:
                batch_parent = parent
                batch_pos = unit.pos
                batch_texts.append(f'** {unit.heading_path.split(" > ")[-1]}\n{unit.text}')
                batch_headings.append(unit.heading_path)
            else:
                batch_parent = ''
                if len(unit.text) >= MIN_TEXT_LENGTH:
                    merged.append(unit)

    _flush_batch()
    return merged


def segment(
    content: str,
    node_id: str,
    max_level: int = 1,
    merge_small: bool = True,
) -> list[EmbedUnit]:
    """Full segmentation: subtrees + explicit embed blocks.

    Embed blocks are extracted separately to avoid duplication — they
    are returned with their own type, and the subtree parser skips
    their content. Small sibling units (e.g. concept definitions) are
    merged to avoid creating hundreds of tiny embedding vectors.
    """
    subtree_units = parse_subtrees(content, node_id, max_level)
    embed_units = parse_embed_blocks(content, node_id)

    if merge_small:
        subtree_units = _merge_small_siblings(subtree_units)

    return subtree_units + embed_units
