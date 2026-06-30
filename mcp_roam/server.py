"""MCP server for org-roam knowledge graph — composition root (FastMCP)."""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from mcp_roam._tools import register_all
from mcp_roam.embeddings import EmbeddingRepo
from mcp_roam.files import RoamFileAccess
from mcp_roam.prompts import register_prompts
from mcp_roam.repo import SqliteRepo
from mcp_roam.youtube import register_youtube


@asynccontextmanager
async def server_lifespan(mcp_app: FastMCP):
    """Initialize and tear down server dependencies."""
    roam_dir = Path(os.environ.get('ROAM_DIR', '/home/pit/roam'))
    db_path = Path(os.environ.get('ROAM_DB', '/home/pit/.emacs.d/org-roam.db'))

    if not db_path.exists():
        print(f'Error: org-roam database not found at {db_path}', file=sys.stderr)
        sys.exit(1)

    if not roam_dir.exists():
        print(f'Error: roam directory not found at {roam_dir}', file=sys.stderr)
        sys.exit(1)

    reader = SqliteRepo(db_path)
    file_access = RoamFileAccess(roam_dir)

    embed_repo = EmbeddingRepo(db_path)
    try:
        embed_repo.init_schema()
    except ImportError:
        print(
            'Warning: sqlite-vec not installed — semantic search disabled.',
            file=sys.stderr,
        )
        embed_repo = None

    yield {
        'reader': reader,
        'file_access': file_access,
        'embed_repo': embed_repo,
    }


mcp = FastMCP(
    'mcp-roam',
    instructions='MCP server for org-roam — bridge OpenCode to your knowledge graph',
    lifespan=server_lifespan,
)


def _get_deps() -> tuple[SqliteRepo, RoamFileAccess, EmbeddingRepo | None]:
    """Get reader, file_access, and embed_repo from the current lifespan context.

    Only callable during a request (from inside a tool function).
    """
    ctx = mcp.get_context()
    lc = ctx.request_context.lifespan_context
    return lc['reader'], lc['file_access'], lc.get('embed_repo')


register_all(mcp, _get_deps)
register_prompts(mcp)
register_youtube(mcp)


def main():
    mcp.run(transport='stdio')


if __name__ == '__main__':
    sys.exit(main())
