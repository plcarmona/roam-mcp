"""Embedding store — sqlite-vec vectors + Ollama embeddings for org-roam.

Tables (in org-roam.db with embed_ prefix, safe from org-roam clear):
    embed_units  — metadata for each embedding unit
    embed_vec    — virtual vec0 table with the actual vectors
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import struct
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from mcp_roam.domain import EmbedUnit
from mcp_roam.segmenter import segment

OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'localhost:11434')
OLLAMA_EMBED_MODEL = os.environ.get('OLLAMA_EMBED_MODEL', 'snowflake-arctic-embed2')
EMBED_DIM = 1024
MAX_TEXT_CHARS = 2000  # truncate long units before embedding

RERANKER_MODEL = os.environ.get(
    'OLLAMA_RERANKER_MODEL', 'awenleven/Qwen3-Reranker-4B:Q4_K_M'
)
RERANK_INSTRUCTION = (
    'Given a web search query, retrieve relevant passages that answer the query'
)
RERANK_MAX_CANDIDATES = 20
RERANK_WORKERS = 8
RERANK_DOC_CHARS = 800


def _file_hash(content: str) -> str:
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def _serialize_vec(vec: list[float]) -> bytes:
    return struct.pack(f'{len(vec)}f', *vec)


EMBED_BATCH_SIZE = 64  # texts per /v1/embeddings request


def _embed_batch(
    texts: list[str], model: str = '', timeout: int = 120
) -> list[list[float]]:
    """Batched embed via Ollama OpenAI-compat /v1/embeddings (array input).

    ~16x faster than sequential _embed_text calls. Returns one vector per text.
    """
    model = model or OLLAMA_EMBED_MODEL
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        chunk = [t[:MAX_TEXT_CHARS] for t in texts[i : i + EMBED_BATCH_SIZE]]
        req = urllib.request.Request(
            f'http://{OLLAMA_HOST}/v1/embeddings',
            data=json.dumps({'model': model, 'input': chunk}).encode(),
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        out += [d['embedding'] for d in data.get('data', [])]
    return out


def _embed_text(text: str, model: str = '', timeout: int = 30) -> list[float] | None:
    """Get embedding vector from Ollama. Returns None on failure."""
    model = model or OLLAMA_EMBED_MODEL
    url = f'http://{OLLAMA_HOST}/api/embeddings'
    payload = json.dumps({
        'model': model,
        'prompt': text[:MAX_TEXT_CHARS],
    }).encode()
    req = urllib.request.Request(
        url, data=payload, headers={'Content-Type': 'application/json'}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data.get('embedding')
    except Exception as e:
        print(f'[embeddings] Ollama error: {e}', file=sys.stderr)
        return None


def _rerank_single(query: str, doc: str, timeout: int = 15) -> float:
    """Score query-doc relevance via Qwen3-Reranker. Returns 0-1."""
    url = f'http://{OLLAMA_HOST}/v1/chat/completions'
    payload = json.dumps({
        'model': RERANKER_MODEL,
        'messages': [
            {
                'role': 'system',
                'content': (
                    'Judge whether the Document meets the requirements '
                    'based on the Query and the Instruct provided. '
                    'Note that the answer can only be "yes" or "no".'
                ),
            },
            {
                'role': 'user',
                'content': (
                    f'<Instruct>: {RERANK_INSTRUCTION}\n'
                    f'<Query>: {query}\n'
                    f'<Document>: {doc[:RERANK_DOC_CHARS]}'
                ),
            },
            {
                'role': 'assistant',
                'content': '<think>\n\n</think>\n\n',
            },
        ],
        'max_tokens': 1,
        'logprobs': True,
        'top_logprobs': 5,
        'temperature': 0,
    }).encode()

    req = urllib.request.Request(
        url, data=payload, headers={'Content-Type': 'application/json'}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        top_lp = data['choices'][0]['logprobs']['content'][0]['top_logprobs']
        yes_lp = next(
            (t['logprob'] for t in top_lp if t['token'] == 'yes'), -10.0
        )
        no_lp = next(
            (t['logprob'] for t in top_lp if t['token'] == 'no'), -10.0
        )
        yes_score = math.exp(yes_lp)
        no_score = math.exp(no_lp)
        return yes_score / (yes_score + no_score)
    except Exception as e:
        print(f'[rerank] error: {e}', file=sys.stderr)
        return 0.0


class EmbeddingRepo:
    """Manages embedding storage in org-roam.db via sqlite-vec."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    @contextmanager
    def _get_connection(self):
        """Get a RW connection with sqlite-vec loaded."""
        conn = sqlite3.connect(
            f'file:{self.db_path}?mode=rwc',
            uri=True,
            timeout=10,
        )
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA busy_timeout = 5000')
        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        except ImportError:
            print(
                '[embeddings] sqlite-vec not installed — run: pip install sqlite-vec',
                file=sys.stderr,
            )
            conn.close()
            raise
        try:
            yield conn
        finally:
            conn.close()

    def init_schema(self) -> None:
        """Create embedding tables if they don't exist."""
        with self._get_connection() as conn:
            conn.execute(f'''
                CREATE VIRTUAL TABLE IF NOT EXISTS embed_vec USING vec0(
                    embedding float[{EMBED_DIM}]
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS embed_units (
                    rowid INTEGER PRIMARY KEY,
                    node_id TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    heading_path TEXT NOT NULL,
                    unit_type TEXT NOT NULL,
                    text TEXT NOT NULL,
                    pos INTEGER DEFAULT 0
                )
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_embed_node
                ON embed_units(node_id)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_embed_hash
                ON embed_units(file_hash)
            ''')
            conn.commit()

    def is_indexed(self, node_id: str, file_hash: str) -> bool:
        """Check if a node is already indexed with this hash."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                'SELECT 1 FROM embed_units WHERE node_id = ? AND file_hash = ? LIMIT 1',
                (node_id, file_hash),
            )
            return cursor.fetchone() is not None

    def remove_node(self, node_id: str) -> int:
        """Remove all embedding units for a node. Returns count deleted."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                'SELECT rowid FROM embed_units WHERE node_id = ?',
                (node_id,),
            )
            rowids = [row['rowid'] for row in cursor.fetchall()]
            if not rowids:
                return 0
            placeholders = ','.join('?' * len(rowids))
            conn.execute(
                f'DELETE FROM embed_vec WHERE rowid IN ({placeholders})',
                rowids,
            )
            conn.execute(
                f'DELETE FROM embed_units WHERE rowid IN ({placeholders})',
                rowids,
            )
            conn.commit()
            return len(rowids)

    def index_node(
        self,
        node_id: str,
        content: str,
        max_level: int = 3,
    ) -> dict[str, Any]:
        """Segment and index a single node's content.

        Returns stats dict: {units, embedded, skipped, stale}.
        """
        file_hash = _file_hash(content)

        if self.is_indexed(node_id, file_hash):
            return {'units': 0, 'embedded': 0, 'skipped': True, 'stale': False}

        self.remove_node(node_id)

        units = segment(content, node_id, max_level)
        if not units:
            return {'units': 0, 'embedded': 0, 'skipped': False, 'stale': False}

        embedded = 0
        with self._get_connection() as conn:
            for unit in units:
                vec = _embed_text(unit.text)
                if vec is None:
                    continue

                cursor = conn.execute(
                    'INSERT INTO embed_vec (embedding) VALUES (?)',
                    (_serialize_vec(vec),),
                )
                vec_rowid = cursor.lastrowid

                conn.execute(
                    '''INSERT INTO embed_units
                       (rowid, node_id, file_hash, heading_path, unit_type, text, pos)
                       VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (vec_rowid, node_id, file_hash, unit.heading_path,
                     unit.unit_type, unit.text, unit.pos),
                )
                embedded += 1

            conn.commit()

        return {
            'units': len(units),
            'embedded': embedded,
            'skipped': False,
            'stale': False,
        }

    def add_claims(self, node_id: str, claims: list[str], file_hash: str = '') -> int:
        """Embed extracted claims as separate units linked to a node."""
        if not file_hash:
            file_hash = f'claims_{node_id}'
        added = 0
        with self._get_connection() as conn:
            for claim in claims:
                vec = _embed_text(claim)
                if vec is None:
                    continue
                cursor = conn.execute(
                    'INSERT INTO embed_vec (embedding) VALUES (?)',
                    (_serialize_vec(vec),),
                )
                vec_rowid = cursor.lastrowid
                conn.execute(
                    '''INSERT INTO embed_units
                       (rowid, node_id, file_hash, heading_path, unit_type, text, pos)
                       VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (vec_rowid, node_id, file_hash, '(claim)',
                     'claim', claim, 0),
                )
                added += 1
            conn.commit()
        return added

    def index_units(
        self,
        node_id: str,
        file_hash: str,
        units: list[EmbedUnit],
    ) -> int:
        """Batch-index pre-segmented units (e.g., code symbols from tree-sitter).

        Uses _embed_batch for speed. Removes old units for the node first.
        Returns number of units embedded.
        """
        if not units:
            return 0
        self.remove_node(node_id)
        vecs = _embed_batch([u.text for u in units])
        with self._get_connection() as conn:
            for vec, u in zip(vecs, units):
                cursor = conn.execute(
                    'INSERT INTO embed_vec (embedding) VALUES (?)',
                    (_serialize_vec(vec),),
                )
                conn.execute(
                    '''INSERT INTO embed_units
                       (rowid, node_id, file_hash, heading_path, unit_type, text, pos)
                       VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (cursor.lastrowid, node_id, file_hash,
                     u.heading_path, u.unit_type, u.text, u.pos),
                )
            conn.commit()
        return len(units)

    def index_units_bulk(
        self,
        items: list[tuple[str, str, list[EmbedUnit]]],
    ) -> int:
        """Batch-index multiple files' units in ONE embed pass.

        items: [(node_id, file_hash, [unit, ...]), ...]
        Removes old units for all nodes first, then embeds all texts in
        batched /v1/embeddings calls, then inserts in one DB transaction.
        """
        if not items:
            return 0
        # remove old units for all nodes in one connection
        node_ids = [nid for nid, _, _ in items]
        with self._get_connection() as conn:
            for node_id in node_ids:
                cursor = conn.execute(
                    'SELECT rowid FROM embed_units WHERE node_id = ?',
                    (node_id,),
                )
                rowids = [row['rowid'] for row in cursor.fetchall()]
                if rowids:
                    placeholders = ','.join('?' * len(rowids))
                    conn.execute(
                        f'DELETE FROM embed_vec WHERE rowid IN ({placeholders})',
                        rowids,
                    )
                    conn.execute(
                        f'DELETE FROM embed_units WHERE rowid IN ({placeholders})',
                        rowids,
                    )
            conn.commit()
        # flatten all texts + metadata
        all_texts: list[str] = []
        meta: list[tuple[str, str, EmbedUnit]] = []
        for node_id, fh, units in items:
            for u in units:
                all_texts.append(u.text)
                meta.append((node_id, fh, u))
        if not all_texts:
            return 0
        # one batched embed pass
        vecs = _embed_batch(all_texts)
        # one transaction for all inserts
        with self._get_connection() as conn:
            for vec, (node_id, fh, u) in zip(vecs, meta):
                cursor = conn.execute(
                    'INSERT INTO embed_vec (embedding) VALUES (?)',
                    (_serialize_vec(vec),),
                )
                conn.execute(
                    '''INSERT INTO embed_units
                       (rowid, node_id, file_hash, heading_path, unit_type, text, pos)
                       VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (cursor.lastrowid, node_id, fh,
                     u.heading_path, u.unit_type, u.text, u.pos),
                )
            conn.commit()
        return len(all_texts)

    def _rerank(
        self, query: str, results: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        """Rerank search results using Qwen3-Reranker via Ollama."""
        candidates = results[:RERANK_MAX_CANDIDATES]

        with ThreadPoolExecutor(max_workers=RERANK_WORKERS) as executor:
            future_to_idx = {
                executor.submit(_rerank_single, query, r['text']): i
                for i, r in enumerate(candidates)
            }
            scores = [0.0] * len(candidates)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    scores[idx] = future.result()
                except Exception:
                    scores[idx] = 0.0

        for i, r in enumerate(candidates):
            r['rerank_score'] = round(scores[i], 4)

        candidates.sort(key=lambda r: r['rerank_score'], reverse=True)
        return candidates[:top_k]

    def search(
        self,
        query: str,
        k: int = 10,
        node_id: str | None = None,
        unit_type: str | None = None,
        rerank: bool = False,
    ) -> list[dict[str, Any]]:
        """Semantic search across embedding units.

        Returns list of {node_id, heading_path, unit_type, text, pos, distance}.
        When rerank=True, also includes rerank_score (0-1) and results are
        re-ordered by reranker relevance.
        """
        query_vec = _embed_text(query)
        if query_vec is None:
            return []

        fetch_k = (
            max(k, min(k * 2, RERANK_MAX_CANDIDATES)) if rerank else k
        )

        with self._get_connection() as conn:
            sql = '''
                SELECT vu.rowid, vu.distance, eu.node_id, eu.heading_path,
                       eu.unit_type, eu.text, eu.pos
                FROM embed_vec vu
                JOIN embed_units eu ON vu.rowid = eu.rowid
                WHERE vu.embedding MATCH ?
                  AND k = ?
            '''
            params: list[Any] = [_serialize_vec(query_vec), fetch_k]

            if node_id:
                sql += ' AND eu.node_id = ?'
                params.append(node_id)
            if unit_type:
                sql += ' AND eu.unit_type = ?'
                params.append(unit_type)

            sql += ' ORDER BY vu.distance'

            cursor = conn.execute(sql, params)
            results = [
                {
                    'node_id': row['node_id'],
                    'heading_path': row['heading_path'],
                    'unit_type': row['unit_type'],
                    'text': row['text'][:500],
                    'pos': row['pos'],
                    'distance': round(row['distance'], 4),
                }
                for row in cursor.fetchall()
            ]

        if rerank and len(results) > 1:
            results = self._rerank(query, results, k)

        return results

    def get_stats(self) -> dict[str, Any]:
        """Return index statistics."""
        with self._get_connection() as conn:
            total = conn.execute(
                'SELECT COUNT(*) as c FROM embed_units'
            ).fetchone()['c']
            by_type = conn.execute(
                'SELECT unit_type, COUNT(*) as c FROM embed_units GROUP BY unit_type'
            ).fetchall()
            nodes = conn.execute(
                'SELECT COUNT(DISTINCT node_id) as c FROM embed_units'
            ).fetchone()['c']
            return {
                'total_units': total,
                'total_nodes': nodes,
                'by_type': {row['unit_type']: row['c'] for row in by_type},
            }
