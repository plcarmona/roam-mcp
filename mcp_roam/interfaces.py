"""Protocol interfaces for dependency inversion (DIP)."""

from typing import Protocol
from pathlib import Path

from mcp_roam.domain import (
    RoamNode,
    RoamLink,
    RoamTag,
    RoamFile,
    RoamRef,
)


class RoamReader(Protocol):
    """Protocol for reading from org-roam data."""

    async def get_node_by_id(self, node_id: str) -> RoamNode | None:
        """Get a node by its UUID."""
        ...

    async def get_node_by_title(self, title: str) -> RoamNode | None:
        """Get a node by title (exact or alias match)."""
        ...

    async def search_nodes(
        self,
        query: str,
        limit: int = 10
    ) -> list[RoamNode]:
        """Search nodes by title, alias, or tag."""
        ...

    async def get_backlinks(self, node_id: str) -> list[RoamNode]:
        """Get all nodes that link TO this node."""
        ...

    async def get_forward_links(self, node_id: str) -> list[RoamNode]:
        """Get all nodes this node links TO."""
        ...

    async def get_tags(self) -> list[tuple[str, int]]:
        """Get all tags with usage counts."""
        ...

    async def get_nodes_by_tag(self, tag: str) -> list[RoamNode]:
        """Get all nodes with a specific tag."""
        ...

    async def get_recent(self, limit: int = 10) -> list[RoamNode]:
        """Get recently modified nodes (by file mtime)."""
        ...

    async def get_daily_note(self, date: str) -> RoamNode | None:
        """Get a daily note by date (YYYY-MM-DD format)."""
        ...

    async def get_node_tags(self, node_id: str) -> list[str]:
        """Get tags for a specific node."""
        ...

    async def get_node_refs(self, node_id: str) -> list[RoamRef]:
        """Get references for a specific node."""
        ...


class RoamWriter(Protocol):
    """Protocol for writing to org-roam data."""

    async def create_node(
        self,
        title: str,
        body: str = '',
        tags: list[str] | None = None
    ) -> tuple[str, str]:
        """Create a new node.

        Returns:
            (node_id, filepath)
        """
        ...

    async def append_to_node(
        self,
        node_id: str,
        content: str,
        heading: str | None = None
    ) -> None:
        """Append content to an existing node."""
        ...


class EmbeddingStore(Protocol):
    """Protocol for embedding storage and semantic search."""

    def init_schema(self) -> None:
        """Create embedding tables if needed."""
        ...

    def is_indexed(self, node_id: str, file_hash: str) -> bool:
        """Check if a node is indexed with current hash."""
        ...

    def remove_node(self, node_id: str) -> int:
        """Remove all units for a node."""
        ...

    def index_node(
        self,
        node_id: str,
        content: str,
        max_level: int = 3,
    ) -> dict:
        """Segment and index a node's content."""
        ...

    def add_claims(
        self,
        node_id: str,
        claims: list[str],
        file_hash: str = '',
    ) -> int:
        """Embed extracted claims as units."""
        ...

    def search(
        self,
        query: str,
        k: int = 10,
        node_id: str | None = None,
        unit_type: str | None = None,
    ) -> list[dict]:
        """Semantic search."""
        ...

    def get_stats(self) -> dict:
        """Return index statistics."""
        ...


class FileAccess(Protocol):
    """Protocol for file system operations."""

    def resolve_path(self, filename: str) -> Path:
        """Resolve a filename to an absolute path in the roam directory."""
        ...

    def read_file(self, path: Path | str) -> str:
        """Read file content."""
        ...

    def write_file(self, path: Path | str, content: str) -> None:
        """Write file content (atomic write)."""
        ...

    def append_file(self, path: Path | str, content: str) -> None:
        """Append content to a file."""
        ...

    def daily_path(self, date: str) -> Path:
        """Get path for a daily note file."""
        ...

    def exists(self, path: Path | str) -> bool:
        """Check if a file exists."""
        ...

    def list_files(
        self,
        directory: Path | str,
        pattern: str = '*'
    ) -> list[Path]:
        """List files in a directory matching a pattern."""
        ...