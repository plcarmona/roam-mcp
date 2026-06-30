"""Research note dump — create structured notes from research data."""

from __future__ import annotations

from mcp_roam.domain import (
    ResearchNoteData,
    format_org_file,
    format_org_link,
    generate_filename,
)
from mcp_roam.interfaces import FileAccess, RoamReader
from mcp_roam.capture import create_note, _strip_quotes


def _build_research_body(data: ResearchNoteData, topic_node_id: str | None = None) -> str:
    """Build the org body for a research note."""
    lines: list[str] = []

    # Metadata drawer
    lines.append(':METADATA:')
    if data.doi:
        lines.append(f'+DOI: {data.doi}')
    if data.url:
        lines.append(f'+URL: {data.url}')
    if data.year:
        lines.append(f'+YEAR: {data.year}')
    if data.journal:
        lines.append(f'+JOURNAL: {data.journal}')
    if data.authors:
        lines.append(f'+AUTHORS: {", ".join(data.authors)}')
    lines.append(':END:')
    lines.append('')

    # Abstract
    if data.abstract:
        lines.append('* Abstract')
        lines.append('')
        lines.append(data.abstract)
        lines.append('')

    # Key Findings
    if data.findings:
        lines.append('* Key Findings')
        lines.append('')
        for finding in data.findings:
            lines.append(f'- {finding}')
        lines.append('')

    # Links section
    if topic_node_id:
        clean_id = _strip_quotes(topic_node_id) or topic_node_id
        lines.append('* Related')
        lines.append('')
        lines.append(f'Part of: {format_org_link(clean_id, "topic note")}')
        lines.append('')

    return '\n'.join(lines)


async def research_dump(
    file_access: FileAccess,
    reader: RoamReader,
    data: ResearchNoteData,
    topic_node_id: str | None = None,
) -> tuple[str, str]:
    """Create a structured research note.

    Args:
        file_access: File system access.
        reader: Database reader.
        data: Research note data.
        topic_node_id: Optional node ID to link back to.

    Returns:
        (node_id, filepath) tuple.
    """
    body = _build_research_body(data, topic_node_id)

    # Auto-tag based on source type
    tags = ['research', data.source_type]

    return await create_note(
        file_access=file_access,
        reader=reader,
        title=data.title,
        body=body,
        tags=tags,
    )
