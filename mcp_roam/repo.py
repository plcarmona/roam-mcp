"""SQLite repository for org-roam database (read-only)."""

import sqlite3
from contextlib import contextmanager
from typing import Any
from pathlib import Path

from mcp_roam.domain import RoamNode, RoamLink, RoamTag, RoamFile, RoamRef
from mcp_roam.interfaces import RoamReader


@contextmanager
def _get_connection(db_path: str | Path):
    """Get a read-only SQLite connection."""
    conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


class SqliteRepo(RoamReader):
    """Read-only repository for org-roam SQLite database."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def _normalize_id(self, node_id: str) -> str:
        """Normalize node ID to match DB storage (with quotes)."""
        if not node_id.startswith('"'):
            return f'"{node_id}"'
        return node_id

    async def get_node_by_id(self, node_id: str) -> RoamNode | None:
        """Get a node by its UUID."""
        node_id = self._normalize_id(node_id)
        with _get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT id, file, level, pos, todo, priority, scheduled, deadline, '
                'title, properties, olp FROM nodes WHERE id = ?',
                (node_id,)
            )
            row = cursor.fetchone()
            return self._row_to_node(row) if row else None

    async def get_node_by_title(self, title: str) -> RoamNode | None:
        """Get a node by title (exact or alias match)."""
        with _get_connection(self.db_path) as conn:
            cursor = conn.cursor()

            # Try exact title match first (with and without quotes)
            # org-roam stores titles with quotes like '"Earth Game"'
            titles_to_try = [title]
            if not title.startswith('"'):
                titles_to_try.append(f'"{title}"')

            for t in titles_to_try:
                cursor.execute(
                    'SELECT id, file, level, pos, todo, priority, scheduled, deadline, '
                    'title, properties, olp FROM nodes WHERE title = ?',
                    (t,)
                )
                row = cursor.fetchone()
                if row:
                    return self._row_to_node(row)

            # Try alias match
            cursor.execute(
                '''SELECT n.id, n.file, n.level, n.pos, n.todo, n.priority,
                          n.scheduled, n.deadline, n.title, n.properties, n.olp
                   FROM nodes n
                   JOIN aliases a ON n.id = a.node_id
                   WHERE a.alias = ?''',
                (title,)
            )
            row = cursor.fetchone()
            return self._row_to_node(row) if row else None

    async def search_nodes(
        self,
        query: str,
        limit: int = 10
    ) -> list[RoamNode]:
        """Search nodes by title, alias, or tag."""
        with _get_connection(self.db_path) as conn:
            cursor = conn.cursor()

            search_term = f'%{query}%'

            # Search by title
            cursor.execute(
                '''SELECT DISTINCT n.id, n.file, n.level, n.pos, n.todo, n.priority,
                          n.scheduled, n.deadline, n.title, n.properties, n.olp
                   FROM nodes n
                   WHERE n.title LIKE ?
                   LIMIT ?''',
                (search_term, limit)
            )
            nodes = [self._row_to_node(row) for row in cursor.fetchall()]

            # If we have room, search by alias
            if len(nodes) < limit:
                remaining = limit - len(nodes)
                cursor.execute(
                    '''SELECT DISTINCT n.id, n.file, n.level, n.pos, n.todo, n.priority,
                              n.scheduled, n.deadline, n.title, n.properties, n.olp
                     FROM nodes n
                     JOIN aliases a ON n.id = a.node_id
                     WHERE a.alias LIKE ?
                     LIMIT ?''',
                    (search_term, remaining)
                )
                for row in cursor.fetchall():
                    node = self._row_to_node(row)
                    if node not in nodes:
                        nodes.append(node)

            # If we still have room, search by tag
            if len(nodes) < limit:
                remaining = limit - len(nodes)
                cursor.execute(
                    '''SELECT DISTINCT n.id, n.file, n.level, n.pos, n.todo, n.priority,
                              n.scheduled, n.deadline, n.title, n.properties, n.olp
                     FROM nodes n
                     JOIN tags t ON n.id = t.node_id
                     WHERE t.tag LIKE ?
                     LIMIT ?''',
                    (search_term, remaining)
                )
                for row in cursor.fetchall():
                    node = self._row_to_node(row)
                    if node not in nodes:
                        nodes.append(node)

            return nodes

    async def get_backlinks(self, node_id: str) -> list[RoamNode]:
        """Get all nodes that link TO this node."""
        node_id = self._normalize_id(node_id)
        with _get_connection(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                '''SELECT DISTINCT n.id, n.file, n.level, n.pos, n.todo, n.priority,
                          n.scheduled, n.deadline, n.title, n.properties, n.olp
                   FROM nodes n
                   JOIN links l ON n.id = l.source
                   WHERE l.dest = ? AND l.type = ?
                   ORDER BY n.file, n.pos''',
                (node_id, '"id"')
            )
            return [self._row_to_node(row) for row in cursor.fetchall()]

    async def get_forward_links(self, node_id: str) -> list[RoamNode]:
        """Get all nodes this node links TO."""
        node_id = self._normalize_id(node_id)
        with _get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT DISTINCT n.id, n.file, n.level, n.pos, n.todo, n.priority,
                          n.scheduled, n.deadline, n.title, n.properties, n.olp
                   FROM nodes n
                   JOIN links l ON n.id = l.dest
                   WHERE l.source = ? AND l.type = ?
                   ORDER BY n.file, l.pos''',
                (node_id, '"id"')
            )
            return [self._row_to_node(row) for row in cursor.fetchall()]

    async def get_tags(self) -> list[tuple[str, int]]:
        """Get all tags with usage counts."""
        with _get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT tag, COUNT(*) as count
                   FROM tags
                   WHERE tag IS NOT NULL AND tag != ''
                   GROUP BY tag
                   ORDER BY count DESC'''
            )
            return [(row['tag'], row['count']) for row in cursor.fetchall()]

    async def get_nodes_by_tag(self, tag: str) -> list[RoamNode]:
        """Get all nodes with a specific tag."""
        with _get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT n.id, n.file, n.level, n.pos, n.todo, n.priority,
                          n.scheduled, n.deadline, n.title, n.properties, n.olp
                   FROM nodes n
                   JOIN tags t ON n.id = t.node_id
                   WHERE t.tag = ?''',
                (tag,)
            )
            return [self._row_to_node(row) for row in cursor.fetchall()]

    async def get_recent(self, limit: int = 10) -> list[RoamNode]:
        """Get recently modified nodes (by file mtime)."""
        with _get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT n.id, n.file, n.level, n.pos, n.todo, n.priority,
                          n.scheduled, n.deadline, n.title, n.properties, n.olp
                   FROM nodes n
                   JOIN files f ON n.file = f.file
                   ORDER BY f.mtime DESC
                   LIMIT ?''',
                (limit,)
            )
            return [self._row_to_node(row) for row in cursor.fetchall()]

    async def get_daily_note(self, date: str) -> RoamNode | None:
        """Get a daily note by date (YYYY-MM-DD format)."""
        with _get_connection(self.db_path) as conn:
            cursor = conn.cursor()

            # Daily notes typically have the date as the title
            cursor.execute(
                '''SELECT id, file, level, pos, todo, priority, scheduled, deadline,
                          title, properties, olp
                   FROM nodes
                   WHERE title = ?''',
                (date,)
            )
            row = cursor.fetchone()
            return self._row_to_node(row) if row else None

    async def get_node_tags(self, node_id: str) -> list[str]:
        """Get tags for a specific node."""
        with _get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT tag FROM tags WHERE node_id = ? AND tag IS NOT NULL',
                (node_id,)
            )
            return [row['tag'] for row in cursor.fetchall()]

    async def get_node_refs(self, node_id: str) -> list[RoamRef]:
        """Get references for a specific node."""
        with _get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT node_id, ref, type FROM refs WHERE node_id = ?',
                (node_id,)
            )
            return [
                RoamRef(node_id=row['node_id'], ref=row['ref'], type=row['type'])
                for row in cursor.fetchall()
            ]

    def _row_to_node(self, row: sqlite3.Row | None) -> RoamNode | None:
        """Convert a DB row to a RoamNode."""
        if row is None:
            return None

        return RoamNode(
            id=row['id'],
            file=row['file'],
            level=row['level'],
            pos=row['pos'],
            todo=row['todo'],
            priority=row['priority'],
            scheduled=row['scheduled'],
            deadline=row['deadline'],
            title=row['title'],
            properties=row['properties'],
            olp=row['olp'],
        )