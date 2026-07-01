"""Code indexing — tree-sitter parser + relationship graph + file watcher.

Indexes source code symbols into the shared embedding store (reusing
embed_units/embed_vec with ``code:`` prefixed node_ids — no new vector
tables) and builds an in-memory relationship graph (calls / imports /
contains) for semantic code search and caller/callee traversal.

The file watcher (watchfiles) keeps the index live on every save.

Public API (used by _tools.py):
    CodeGraph(repo)            — create with shared EmbeddingRepo
    .init_schema()             — create code_projects table
    .index_project(root, project)
    .warm(root, project)       — rebuild in-memory graph (no re-embed)
    .reindex_file(path, root, project)
    .search(query, k)
    .neighbors(node_id, sym)
    .start_watcher(root, project)
    .stop_watcher()
    .watcher_status()
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp_roam.domain import EmbedUnit
from mcp_roam.embeddings import EmbeddingRepo

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EXCLUDE = {
    '.venv', '__pycache__', '.git', '.jj', '.pytest_cache', '.ruff_cache',
    '.mypy_cache', '.tox', 'node_modules', '.next', 'build', 'dist', '.eggs',
    'yourmt3', 'vendor', 'third_party',
}

BODY_EXCERPT_CHARS = 1200
MAX_TEXT = 2000


# ---------------------------------------------------------------------------
# Parser — source file -> symbols + imports via tree-sitter
# ---------------------------------------------------------------------------

_PARSERS: dict[str, Any] = {}


def get_parser(lang: str) -> Any:
    """Return a cached tree-sitter Parser for a language."""
    if lang in _PARSERS:
        return _PARSERS[lang]
    if lang == 'python':
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser
        language = Language(tspython.language())
    elif lang in ('typescript', 'tsx'):
        import tree_sitter_typescript as tsts
        from tree_sitter import Language, Parser
        language = Language(
            tsts.language_tsx() if lang == 'tsx' else tsts.language_typescript()
        )
    else:
        raise ValueError(f'no grammar for {lang!r}')
    parser = Parser(language)
    _PARSERS[lang] = parser
    return parser


def detect_lang(path: Path) -> str | None:
    ext = path.suffix.lower()
    return {'.py': 'python', '.ts': 'typescript', '.tsx': 'tsx'}.get(ext)


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: str            # 'function' | 'method' | 'class'
    path: str            # dotted: "webui/server.py > Session > load_midi"
    signature: str
    docstring: str
    body_excerpt: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class Import:
    module: str
    names: tuple[str, ...]
    line: int


@dataclass(frozen=True)
class ParsedFile:
    node_id: str         # "code:Piano::webui/server.py"
    relpath: str
    lang: str
    file_hash: str
    symbols: tuple[Symbol, ...]
    imports: tuple[Import, ...]


def _signature(node, source: bytes) -> str:
    body = node.child_by_field_name('body')
    end = body.start_byte if body else node.end_byte
    return source[node.start_byte:end].decode().strip()


def _docstring(body, source: bytes) -> str:
    if not body:
        return ''
    for child in body.children:
        if not child.is_named:
            continue
        inner = child.children[0] if child.children else None
        if inner is not None and inner.type == 'string':
            return inner.text.decode().strip()
        break
    return ''


def _body_excerpt(body, source: bytes) -> str:
    if not body:
        return ''
    raw = source[body.start_byte:body.end_byte].decode(errors='replace')
    if len(raw) > BODY_EXCERPT_CHARS:
        raw = raw[:BODY_EXCERPT_CHARS].rstrip() + '\n    ...'
    return raw.strip()


def _walk_symbols(node, source: bytes, relpath: str,
                  ancestors: list[str], out: list[Symbol]) -> None:
    if node.type == 'function_definition':
        name_node = node.child_by_field_name('name')
        name = name_node.text.decode() if name_node else '<anon>'
        kind = 'method' if ancestors else 'function'
        path = ' > '.join([relpath, *ancestors, name])
        body = node.child_by_field_name('body')
        out.append(Symbol(
            name=name, kind=kind, path=path,
            signature=_signature(node, source),
            docstring=_docstring(body, source),
            body_excerpt=_body_excerpt(body, source),
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
        ))
        for child in node.children:
            if child.is_named:
                _walk_symbols(child, source, relpath, ancestors + [name], out)
        return
    if node.type == 'class_definition':
        name_node = node.child_by_field_name('name')
        name = name_node.text.decode() if name_node else '<anon>'
        path = ' > '.join([relpath, *ancestors, name])
        body = node.child_by_field_name('body')
        out.append(Symbol(
            name=name, kind='class', path=path,
            signature=_signature(node, source),
            docstring=_docstring(body, source),
            body_excerpt=_body_excerpt(body, source),
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
        ))
        for child in node.children:
            if child.is_named:
                _walk_symbols(child, source, relpath, ancestors + [name], out)
        return
    for child in node.children:
        if child.is_named:
            _walk_symbols(child, source, relpath, ancestors, out)


def _collect_imports(root, source: bytes) -> list[Import]:
    imports: list[Import] = []

    def _kw_index(node, kw):
        for i, c in enumerate(node.children):
            if c.type == kw:
                return i
        return None

    def _local_name(node) -> str:
        if node.type == 'identifier':
            return node.text.decode()
        if node.type == 'aliased_import':
            alias = node.child_by_field_name('alias')
            nm = node.child_by_field_name('name')
            return alias.text.decode() if alias else (
                nm.text.decode() if nm else ''
            )
        if node.type == 'dotted_name':
            return node.text.decode()
        return ''

    def visit(node):
        if node.type == 'import_statement':
            idx = _kw_index(node, 'import')
            after = node.children[idx + 1:] if idx is not None else []
            mod, names = '', []
            for c in after:
                if not c.is_named:
                    continue
                if c.type == 'dotted_name':
                    mod = c.text.decode()
                    names.append(c.text.decode().split('.')[0])
                elif c.type == 'aliased_import':
                    full = c.child_by_field_name('name')
                    if full:
                        mod = full.text.decode()
                    names.append(_local_name(c))
            if mod:
                imports.append(Import(module=mod, names=tuple(names),
                                      line=node.start_point[0] + 1))
        elif node.type == 'import_from_statement':
            children = node.children
            idx = _kw_index(node, 'import')
            before = children[:idx] if idx is not None else []
            after = children[idx + 1:] if idx is not None else children
            mod = ''.join(c.text.decode() for c in before
                          if c.type in ('dotted_name', 'identifier'))
            names = [_local_name(c) for c in after
                     if c.is_named and _local_name(c)]
            imports.append(Import(module=mod, names=tuple(names),
                                  line=node.start_point[0] + 1))
        for child in node.children:
            if child.is_named:
                visit(child)

    visit(root)
    return imports


def _project_root(path: Path, project: str) -> Path:
    for parent in path.parents:
        if parent.name == project:
            return parent
    return path.parent


def parse_file(path: Path, project: str) -> ParsedFile:
    path = Path(path)
    lang = detect_lang(path) or 'python'
    source = path.read_bytes()
    relpath = str(path.relative_to(_project_root(path, project)))
    node_id = f'code:{project}::{relpath}'
    parser = get_parser(lang)
    tree = parser.parse(source)
    root = tree.root_node
    symbols: list[Symbol] = []
    _walk_symbols(root, source, relpath, [], symbols)
    imports = _collect_imports(root, source)
    return ParsedFile(
        node_id=node_id, relpath=relpath, lang=lang,
        file_hash=hashlib.md5(source).hexdigest(),
        symbols=tuple(symbols), imports=tuple(imports),
    )


def symbols_to_embed_units(pf: ParsedFile) -> list[EmbedUnit]:
    """Convert ParsedFile symbols -> EmbedUnit list for batch embedding."""
    units = []
    for sym in pf.symbols:
        parts = [sym.signature, sym.docstring, '', sym.body_excerpt]
        text = '\n'.join(p for p in parts if p).strip()
        if len(text) > MAX_TEXT:
            text = text[:MAX_TEXT]
        if len(text) < 12:
            continue
        units.append(EmbedUnit(
            node_id=pf.node_id,
            heading_path=sym.path,
            unit_type=sym.kind,
            text=text,
            pos=sym.start_line,
        ))
    return units


# ---------------------------------------------------------------------------
# Graph — relationship edges + CodeGraph
# ---------------------------------------------------------------------------

@dataclass
class Edge:
    src_node: str
    src_sym: str
    dst_node: str
    dst_sym: str
    kind: str          # call | import | contains


def _module_path(relpath: str) -> str:
    p = relpath.replace('/', '.').replace('\\', '.')
    if p.endswith('.py'):
        p = p[:-3]
    return p


def _collect_calls(root) -> list[tuple[list[str], str, int]]:
    """Yield (enclosing_symbol_names, callee_text, line) per call node."""
    edges: list[tuple[list[str], str, int]] = []

    def walk(node, enclosing: list[str]):
        if node.type == 'function_definition':
            nm = node.child_by_field_name('name')
            name = nm.text.decode() if nm else '<anon>'
            for c in node.children:
                if c.is_named:
                    walk(c, enclosing + [name])
            return
        if node.type == 'class_definition':
            nm = node.child_by_field_name('name')
            name = nm.text.decode() if nm else '<anon>'
            for c in node.children:
                if c.is_named:
                    walk(c, enclosing + [name])
            return
        if node.type == 'call':
            fn = node.child_by_field_name('function')
            callee = fn.text.decode() if fn else '<expr>'
            edges.append((enclosing, callee, node.start_point[0] + 1))
        for c in node.children:
            if c.is_named:
                walk(c, enclosing)

    walk(root, [])
    return edges


class CodeGraph:
    """Code indexing graph: parse + embed + relationship edges + watcher."""

    def __init__(self, repo: EmbeddingRepo):
        self.repo = repo
        self.parsed: dict[str, ParsedFile] = {}
        self.module_to_node: dict[str, str] = {}
        self.global_syms: dict[tuple[str, str], tuple[str, str]] = {}
        self.edges: list[Edge] = []
        self._lock = threading.RLock()
        # watcher state
        self._watcher_thread: threading.Thread | None = None
        self._watcher_stop = threading.Event()
        self._watcher_root: Path | None = None
        self._watcher_project: str | None = None
        self._watcher_events: list[dict] = []

    # ---- schema ----

    def init_schema(self) -> None:
        with self.repo._get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS code_projects (
                    project   TEXT PRIMARY KEY,
                    root_path TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def _register_project(self, project: str, root: Path) -> None:
        with self.repo._get_connection() as conn:
            conn.execute(
                '''INSERT INTO code_projects (project, root_path, updated_at)
                   VALUES (?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(project) DO UPDATE SET
                       root_path = excluded.root_path,
                       updated_at = CURRENT_TIMESTAMP''',
                (project, str(root)),
            )
            conn.commit()

    def warm_all(self) -> list[str]:
        """On startup, warm all previously-indexed projects from DB."""
        warmed: list[str] = []
        try:
            with self.repo._get_connection() as conn:
                rows = conn.execute(
                    'SELECT project, root_path FROM code_projects'
                ).fetchall()
        except Exception:
            return []
        for row in rows:
            root = Path(row['root_path'])
            if root.exists():
                self.warm(root, row['project'])
                warmed.append(row['project'])
        return warmed

    # ---- bulk index ----

    def index_project(
        self, root: Path, project: str,
        exclude: set[str] | None = None,
    ) -> dict:
        root = Path(root)
        exclude = exclude or DEFAULT_EXCLUDE
        t0 = time.perf_counter()

        files = [f for f in sorted(root.rglob('*.py'))
                 if not any(p in exclude for p in f.parts)]
        for f in files:
            try:
                pf = parse_file(f, project)
            except Exception:
                continue
            self.parsed[pf.node_id] = pf

        for pf in self.parsed.values():
            self._refresh_global_syms(pf)

        with self._lock:
            self.edges.clear()
            for pf in self.parsed.values():
                self._build_file_edges(pf, root)

        t_parse = time.perf_counter() - t0
        t1 = time.perf_counter()

        # batch-embed ALL symbols in one pass (not per-file — ~16x faster)
        items: list[tuple[str, str, list[EmbedUnit]]] = []
        for pf in self.parsed.values():
            units = symbols_to_embed_units(pf)
            if units:
                items.append((pf.node_id, pf.file_hash, units))
        embedded = self.repo.index_units_bulk(items)

        t_embed = time.perf_counter() - t1
        self._register_project(project, root)

        return {
            'files': len(self.parsed), 'symbols': embedded,
            'edges': len(self.edges), 'parse_s': round(t_parse, 2),
            'embed_s': round(t_embed, 2),
        }

    def warm(self, root: Path, project: str,
             exclude: set[str] | None = None) -> dict:
        """Rebuild in-memory graph from existing DB (no re-embed). Fast."""
        root = Path(root)
        exclude = exclude or DEFAULT_EXCLUDE
        t0 = time.perf_counter()
        files = [f for f in sorted(root.rglob('*.py'))
                 if not any(p in exclude for p in f.parts)]
        for f in files:
            try:
                pf = parse_file(f, project)
            except Exception:
                continue
            self.parsed[pf.node_id] = pf
        for pf in self.parsed.values():
            self._refresh_global_syms(pf)
        with self._lock:
            self.edges.clear()
            for pf in self.parsed.values():
                self._build_file_edges(pf, root)
        return {
            'files': len(self.parsed), 'edges': len(self.edges),
            'ms': round((time.perf_counter() - t0) * 1000),
        }

    # ---- incremental re-index ----

    def reindex_file(self, path: Path, root: Path, project: str) -> dict:
        t0 = time.perf_counter()
        pf = parse_file(path, project)

        if self.repo.is_indexed(pf.node_id, pf.file_hash):
            return {'status': 'skipped', 'node_id': pf.node_id,
                    'ms': round((time.perf_counter() - t0) * 1000)}

        with self._lock:
            self.parsed[pf.node_id] = pf
            self._refresh_global_syms(pf)
            before = len(self.edges)
            self.edges = [e for e in self.edges
                          if e.src_node != pf.node_id and e.dst_node != pf.node_id]
            self._build_file_edges(pf, root)
            edges_delta = len(self.edges) - before

        units = symbols_to_embed_units(pf)
        self.repo.index_units(pf.node_id, pf.file_hash, units)
        ms = round((time.perf_counter() - t0) * 1000)
        return {'status': 'reindexed', 'node_id': pf.node_id,
                'symbols': len(units), 'edges_delta': edges_delta, 'ms': ms}

    # ---- per-file helpers ----

    def _refresh_global_syms(self, pf: ParsedFile) -> None:
        mod = _module_path(pf.relpath)
        self.module_to_node[mod] = pf.node_id
        self.global_syms = {
            k: v for k, v in self.global_syms.items()
            if not (k[0] == mod and v[0] == pf.node_id)
        }
        for s in pf.symbols:
            if s.kind in ('function', 'class'):
                self.global_syms[(mod, s.name)] = (pf.node_id, s.name)

    def _build_file_edges(self, pf: ParsedFile, root: Path) -> None:
        node_id = pf.node_id
        owner = None
        for s in pf.symbols:
            if s.kind == 'class':
                owner = s.name
            elif s.kind == 'method' and owner:
                self.edges.append(Edge(node_id, owner, node_id, s.name, 'contains'))
        file_imports: dict[str, str] = {}
        for imp in pf.imports:
            tgt = self.module_to_node.get(imp.module)
            if tgt:
                file_imports.update({n: imp.module for n in imp.names})
                for n in imp.names:
                    tgt_sym = self.global_syms.get((imp.module, n))
                    if tgt_sym:
                        self.edges.append(
                            Edge(node_id, '(file)', tgt_sym[0], n, 'import'))
        parser = get_parser(pf.lang)
        src = (root / pf.relpath).read_bytes()
        tree = parser.parse(src)
        sym_names = {s.name for s in pf.symbols}
        method_by_owner: dict[str, dict[str, str]] = defaultdict(dict)
        for s in pf.symbols:
            if s.kind == 'method':
                owner2 = s.path.split(' > ')[-2]
                method_by_owner[owner2][s.name] = s.name
        for enclosing, callee, _line in _collect_calls(tree.root_node):
            if not enclosing:
                continue
            caller_sym = enclosing[-1]
            callee_clean = callee.split('(')[0]
            head = callee_clean.split('.')[0]
            resolved = None
            if callee_clean.startswith('self.'):
                m = callee_clean[5:].split('(')[0]
                owner3 = enclosing[-2] if len(enclosing) > 1 else None
                if owner3 and m in method_by_owner.get(owner3, {}):
                    resolved = (node_id, m)
            if not resolved and head in file_imports:
                srcmod = file_imports[head]
                tgt = (self.global_syms.get((srcmod, callee_clean.split('.')[-1]))
                       or self.global_syms.get((srcmod, head)))
                if tgt:
                    resolved = tgt
            if not resolved and callee_clean in sym_names:
                resolved = (node_id, callee_clean)
            if resolved:
                self.edges.append(
                    Edge(node_id, caller_sym, resolved[0], resolved[1], 'call'))

    # ---- query API ----

    def search(self, query: str, k: int = 8) -> list[dict]:
        results = self.repo.search(query, k=k)
        return [r for r in results if r['node_id'].startswith('code:')]

    def neighbors(self, node_id: str, sym: str) -> dict:
        with self._lock:
            callers, callees, imports = [], [], []
            for e in self.edges:
                if e.dst_node == node_id and e.dst_sym == sym and e.kind == 'call':
                    callers.append((e.src_node, e.src_sym))
                elif e.src_node == node_id and e.src_sym == sym and e.kind == 'call':
                    callees.append((e.dst_node, e.dst_sym))
                elif e.src_node == node_id and e.kind == 'import':
                    imports.append((e.dst_node, e.dst_sym))
            return {
                'callers': sorted(set(callers)),
                'callees': sorted(set(callees)),
                'imports': sorted(set(imports)),
            }

    # ---- watcher ----

    def start_watcher(self, root: Path, project: str,
                      exclude: set[str] | None = None) -> dict:
        if self._watcher_thread and self._watcher_thread.is_alive():
            return {'status': 'already_running', 'root': str(self._watcher_root)}
        exclude = exclude or DEFAULT_EXCLUDE
        self._watcher_root = Path(root)
        self._watcher_project = project
        self._watcher_stop.clear()
        self._watcher_events.clear()
        self._watcher_thread = threading.Thread(
            target=self._watcher_loop,
            args=(Path(root), project, exclude),
            daemon=True,
        )
        self._watcher_thread.start()
        return {'status': 'started', 'root': str(root), 'project': project}

    def stop_watcher(self) -> dict:
        if not self._watcher_thread or not self._watcher_thread.is_alive():
            return {'status': 'not_running'}
        self._watcher_stop.set()
        self._watcher_thread.join(timeout=3)
        return {'status': 'stopped'}

    def watcher_status(self) -> dict:
        alive = self._watcher_thread and self._watcher_thread.is_alive()
        return {
            'running': bool(alive),
            'root': str(self._watcher_root) if self._watcher_root else None,
            'project': self._watcher_project,
            'recent_events': self._watcher_events[-10:],
        }

    def _watcher_loop(
        self, root: Path, project: str, exclude: set[str],
    ) -> None:
        from watchfiles import Change, watch

        def py_filter(change: Change, path: str) -> bool:
            return (path.endswith('.py')
                    and not any(x in Path(path).parts for x in exclude))

        try:
            for changes in watch(root, debounce=300, watch_filter=py_filter):
                if self._watcher_stop.is_set():
                    break
                for change, path_str in sorted(changes):
                    path = Path(path_str)
                    if change == Change.deleted:
                        node_id = f'code:{project}::{path.relative_to(root)}'
                        self.repo.remove_node(node_id)
                        self._log_event('deleted', path.name, 0)
                        continue
                    try:
                        stats = self.reindex_file(path, root, project)
                        self._log_event(
                            stats['status'], path.name, stats.get('ms', 0),
                        )
                    except Exception as e:
                        self._log_event('error', path.name, 0, str(e))
        except Exception as e:
            self._log_event('watcher_error', str(root), 0, str(e))

    def _log_event(
        self, status: str, filename: str, ms: int, error: str = '',
    ) -> None:
        self._watcher_events.append({
            'time': time.strftime('%H:%M:%S'),
            'status': status, 'file': filename, 'ms': ms, 'error': error,
        })
        if len(self._watcher_events) > 100:
            self._watcher_events = self._watcher_events[-50:]
