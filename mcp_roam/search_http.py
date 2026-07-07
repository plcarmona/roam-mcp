"""Minimal localhost HTTP endpoint for semantic search over org-roam embeddings.

Speaks plain JSON (NOT MCP) so Emacs can consume it with url.el, independent of
the stdio MCP server. Shares the same org-roam.db (read-only for queries).

Env:
    ROAM_DB            path to org-roam.db  (default ~/.emacs.d/org-roam.db)
    ROAM_SEARCH_HOST   bind host            (default 127.0.0.1)
    ROAM_SEARCH_PORT   bind port            (default 8765)

Routes:
    GET  /health           -> {"ok": true}
    POST /search           -> {"query","k","rerank"} -> {"results": [...]}
                              each result carries node_id/title/file/heading_path/
                              text/distance/rerank_score so Emacs can open + jump.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from mcp_roam.embeddings import EmbeddingRepo

DB_PATH = Path(os.environ.get('ROAM_DB', '/home/pit/.emacs.d/org-roam.db'))
HOST = os.environ.get('ROAM_SEARCH_HOST', '127.0.0.1')
PORT = int(os.environ.get('ROAM_SEARCH_PORT', '8765'))


def _q(v: Any) -> str:
    """Strip the surrounding double quotes org-roam stores in the DB."""
    if v is None:
        return ''
    s = str(v)
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s


def _resolve_nodes(conn: sqlite3.Connection, node_ids: list[str]) -> dict[str, dict[str, str]]:
    """Batch node_id (clean uuid) -> {file, title} in one query."""
    out: dict[str, dict[str, str]] = {}
    clean = [n for n in set(node_ids) if n]
    if not clean:
        return out
    quoted = [f'"{n}"' for n in clean]
    placeholders = ','.join('?' * len(quoted))
    rows = conn.execute(
        f'SELECT id, file, title FROM nodes WHERE id IN ({placeholders})',
        quoted,
    ).fetchall()
    for r in rows:
        out[_q(r['id'])] = {'file': _q(r['file']), 'title': _q(r['title'])}
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: Any) -> None:
        body = json.dumps(payload).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:  # keep stderr quiet
        pass

    def do_GET(self) -> None:
        if self.path == '/health':
            self._send(200, {'ok': True})
        else:
            self._send(404, {'error': 'not found'})

    def do_POST(self) -> None:
        if self.path != '/search':
            self._send(404, {'error': 'not found'})
            return
        try:
            length = int(self.headers.get('Content-Length', 0))
            raw = self.rfile.read(length) if length else b'{}'
            data = json.loads(raw)
        except Exception as e:
            self._send(400, {'error': f'bad json: {e}'})
            return

        query = str(data.get('query', '')).strip()
        try:
            k = max(1, min(int(data.get('k', 10)), 50))
        except (TypeError, ValueError):
            k = 10
        rerank = bool(data.get('rerank', False))

        if not query:
            self._send(200, {'query': '', 'count': 0, 'results': []})
            return

        repo: EmbeddingRepo = self.server.embed_repo  # type: ignore[attr-defined]
        try:
            hits = repo.search(query, k=k, rerank=rerank)
        except Exception as e:
            print(f'[search_http] search error: {e}', file=sys.stderr)
            self._send(500, {'error': str(e)})
            return

        # drop code-indexed units (use roam_code_search for those)
        hits = [h for h in hits if not h['node_id'].startswith('code:')]

        try:
            with sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True) as conn:
                conn.row_factory = sqlite3.Row
                meta = _resolve_nodes(conn, [h['node_id'] for h in hits])
        except Exception as e:
            print(f'[search_http] resolve error: {e}', file=sys.stderr)
            meta = {}

        results = []
        for h in hits[:k]:
            m = meta.get(h['node_id'], {})
            results.append({
                'node_id': h['node_id'],
                'title': m.get('title', ''),
                'file': m.get('file', ''),
                'heading_path': h['heading_path'],
                'unit_type': h['unit_type'],
                'text': h['text'],
                'distance': h.get('distance'),
                'rerank_score': h.get('rerank_score'),
            })

        self._send(200, {'query': query, 'count': len(results), 'results': results})


def main() -> None:
    if not DB_PATH.exists():
        print(f'[search_http] DB not found: {DB_PATH}', file=sys.stderr)
        sys.exit(1)

    repo = EmbeddingRepo(DB_PATH)
    try:
        repo.init_schema()
    except ImportError:
        print('[search_http] sqlite-vec not installed — semantic search disabled.',
              file=sys.stderr)
        sys.exit(1)

    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    httpd.embed_repo = repo  # type: ignore[attr-defined]
    print(f'[search_http] listening on http://{HOST}:{PORT}', file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()


if __name__ == '__main__':
    main()
