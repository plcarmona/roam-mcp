"""Code indexing v2 — multi-language tree-sitter parser + relationship graph +
file watcher.

Indexes source code symbols into the shared embedding store (reusing
embed_units/embed_vec with ``code:`` prefixed node_ids) and builds an
in-memory relationship graph (calls / imports / contains) for semantic code
search and caller/callee traversal.

v2 changes vs v1 (see code-index-eval-SUMMARY.md for the defect analysis):
  D1 fixed — actually parses .ts/.tsx (and any supported ext), not just .py.
  D2 fixed — relpath is always relative to the index root; project name is
            decoupled from dir name (no more _project_root name-guessing that
            broke the standard src/<pkg> layout).
  D3 fixed — index_project(replace=True) drops the project's prior state first
            (idempotent, no cross-project contamination); graph state is
            namespaced by project so edges never resolve across projects.
  rerank   — CodeGraph.search reranks by default (the reranker already existed
            but wasn't wired onto the code path).
  lifecycle — new remove_project() / list_projects() helpers.

Public API (kept stable so _tools.py / server.py need no changes):
    CodeGraph(repo)            — create with shared EmbeddingRepo
    .init_schema()             — create code_projects table
    .index_project(root, project, *, replace=True)
    .warm(root, project)       — rebuild in-memory graph (no re-embed)
    .warm_all()                — warm all registered projects
    .reindex_file(path, root, project)
    .remove_project(project)
    .list_projects()
    .search(query, k=8, *, project=None, rerank=True)
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
    'yourmt3', 'vendor', 'third_party', '.astro',
}

# extension -> language name (matches detect_lang below)
LANG_BY_EXT = {'.py': 'python', '.ts': 'typescript', '.tsx': 'tsx'}
SUPPORTED_EXTS = set(LANG_BY_EXT)

BODY_EXCERPT_CHARS = 1200
MAX_TEXT = 2000

# --- language-aware node-type tables --------------------------------------
CLASS_TYPES = {
    'python': {'class_definition'},
    'typescript': {'class_declaration'},
    'tsx': {'class_declaration'},
}
# Python: a function_definition under a class is a method (handled via ancestors).
# TS: function_declaration is a free function; methods are method_definition.
FUNC_TYPES = {
    'python': {'function_definition'},
    'typescript': {'function_declaration'},
    'tsx': {'function_declaration'},
}
METHOD_TYPES = {
    'python': set(),
    'typescript': {'method_definition'},
    'tsx': {'method_definition'},
}
# value node types that make a `const x = <value>` a named function
NAMED_VALUE_FUNCS = {'arrow_function', 'function', 'function_expression'}
# node types that introduce a scope whose calls we attribute to a named symbol
SCOPE_TYPES = {
    'python': {'function_definition', 'class_definition'},
    'typescript': {'function_declaration', 'class_declaration',
                   'method_definition', 'variable_declarator'},
    'tsx': {'function_declaration', 'class_declaration',
            'method_definition', 'variable_declarator'},
}


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
    return LANG_BY_EXT.get(path.suffix.lower())


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
    project: str
    root: str            # absolute index root (D2 fix: edges read relpath relative to THIS)
    relpath: str
    lang: str
    file_hash: str
    symbols: tuple[Symbol, ...]
    imports: tuple[Import, ...]


def _sig_up_to_body(node, source: bytes) -> str:
    body = node.child_by_field_name('body')
    end = body.start_byte if body else node.end_byte
    return source[node.start_byte:end].decode().strip()


def _docstring_python(body, source: bytes) -> str:
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


def _leading_comment(source_text: str, start_byte: int) -> str:
    """Grab a /** */ or // block immediately before a TS symbol (best effort)."""
    pre = source_text[:start_byte].rstrip()
    if pre.endswith('*/'):
        idx = pre.rfind('/*')
        if idx != -1:
            return pre[idx + 2:].strip('* \n')[:300]
    return ''


def _body_text(body, source: bytes) -> str:
    if not body:
        return ''
    raw = source[body.start_byte:body.end_byte].decode(errors='replace')
    if len(raw) > BODY_EXCERPT_CHARS:
        raw = raw[:BODY_EXCERPT_CHARS].rstrip() + '\n    ...'
    return raw.strip()


def _name_of(node):
    nm = node.child_by_field_name('name')
    return nm.text.decode() if nm else '<anon>'


def _walk_symbols(node, source: bytes, lang: str, relpath: str,
                  ancestors: list[str], out: list[Symbol]) -> None:
    t = node.type

    def emit(kind: str, name: str, body) -> None:
        path = ' > '.join([relpath, *ancestors, name])
        doc = (_docstring_python(body, source) if lang == 'python'
               else _leading_comment(source.decode(errors='replace'),
                                     node.start_byte))
        out.append(Symbol(
            name=name, kind=kind, path=path,
            signature=_sig_up_to_body(node, source),
            docstring=doc,
            body_excerpt=_body_text(body, source),
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
        ))

    if t in CLASS_TYPES.get(lang, ()):           # class
        name = _name_of(node)
        emit('class', name, node.child_by_field_name('body'))
        for c in node.children:
            if c.is_named:
                _walk_symbols(c, source, lang, relpath, ancestors + [name], out)
        return
    if t in FUNC_TYPES.get(lang, ()):            # function (method if nested in class)
        name = _name_of(node)
        kind = 'method' if ancestors else 'function'
        emit(kind, name, node.child_by_field_name('body'))
        for c in node.children:
            if c.is_named:
                _walk_symbols(c, source, lang, relpath, ancestors + [name], out)
        return
    if t in METHOD_TYPES.get(lang, ()):          # TS method_definition
        name = _name_of(node)
        emit('method', name, node.child_by_field_name('body'))
        for c in node.children:
            if c.is_named:
                _walk_symbols(c, source, lang, relpath, ancestors + [name], out)
        return
    # TS named arrow/const function:  const foo = () => {} / function expr
    if lang in ('typescript', 'tsx') and t == 'variable_declarator':
        value = node.child_by_field_name('value')
        if value is not None and value.type in NAMED_VALUE_FUNCS:
            name = _name_of(node)
            body = value.child_by_field_name('body')
            path = ' > '.join([relpath, *ancestors, name])
            doc = _leading_comment(source.decode(errors='replace'), node.start_byte)
            out.append(Symbol(
                name=name, kind='function', path=path,
                signature=source[node.start_byte:(body.start_byte if body else node.end_byte)].decode().strip(),
                docstring=doc, body_excerpt=_body_text(body, source),
                start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1,
            ))
            for c in value.children:
                if c.is_named:
                    _walk_symbols(c, source, lang, relpath, ancestors + [name], out)
            return
    for c in node.children:
        if c.is_named:
            _walk_symbols(c, source, lang, relpath, ancestors, out)


def _collect_imports_python(root) -> list[Import]:
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
                nm.text.decode() if nm else '')
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


def _collect_imports_ts(root) -> list[Import]:
    """Coarse TS import extraction: module = source string, names = imported
    identifiers (default + named + namespace). Good enough for edge resolution.
    """
    imports: list[Import] = []

    def visit(node):
        if node.type == 'import_statement':
            mod, names = '', []
            for c in node.children:
                if c.is_named and c.type == 'string':
                    mod = c.text.decode().strip().strip('\'"`')
                if c.is_named:
                    for ident in _identifiers(c):
                        names.append(ident)
            if mod:
                imports.append(Import(module=mod, names=tuple(dict.fromkeys(names)),
                                      line=node.start_point[0] + 1))
        for child in node.children:
            if child.is_named:
                visit(child)

    visit(root)
    return imports


def _identifiers(node) -> list[str]:
    out: list[str] = []
    # only descend into import_clause-ish children, not the source string
    if node.type == 'string':
        return out
    nm = node.child_by_field_name('name')
    if node.type == 'identifier' and nm is None:
        out.append(node.text.decode())
    for c in node.children:
        if c.is_named:
            out.extend(_identifiers(c))
    return out


def _collect_imports(root, lang: str) -> list[Import]:
    return _collect_imports_python(root) if lang == 'python' else _collect_imports_ts(root)


def parse_file(path: Path, root: Path, project: str) -> ParsedFile:
    path = Path(path)
    lang = detect_lang(path) or 'python'
    source = path.read_bytes()
    relpath = str(path.relative_to(root))           # D2 fix: relative to the index root
    node_id = f'code:{project}::{relpath}'
    parser = get_parser(lang)
    tree = parser.parse(source)
    ast_root = tree.root_node
    symbols: list[Symbol] = []
    _walk_symbols(ast_root, source, lang, relpath, [], symbols)
    imports = _collect_imports(ast_root, lang)
    return ParsedFile(
        node_id=node_id, project=project, root=str(root), relpath=relpath,
        lang=lang, file_hash=hashlib.md5(source).hexdigest(),
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

@dataclass(frozen=True)
class Edge:
    src_node: str
    src_sym: str
    dst_node: str
    dst_sym: str
    kind: str          # call | import | contains


def _module_path(relpath: str) -> str:
    """Convert a relpath to a dotted module key, stripping any source ext."""
    p = relpath.replace('\\', '/')
    for ext in ('.py', '.ts', '.tsx', '.js', '.jsx'):
        if p.endswith(ext):
            p = p[: -len(ext)]
            break
    return p.replace('/', '.')


def _collect_calls_python(root) -> list[tuple[list[str], str, int]]:
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


def _scope_name(node, lang: str) -> str | None:
    """Name of a scope-introducing node, or None if it's not a function scope
    (e.g. a class — we don't attribute calls to classes)."""
    t = node.type
    if t in CLASS_TYPES.get(lang, ()):
        return None
    if t in FUNC_TYPES.get(lang, ()) or t in METHOD_TYPES.get(lang, ()):
        return _name_of(node)
    if lang in ('typescript', 'tsx') and t == 'variable_declarator':
        value = node.child_by_field_name('value')
        if value is not None and value.type in NAMED_VALUE_FUNCS:
            return _name_of(node)
    return None


def _collect_calls_ts(root, lang: str) -> list[tuple[list[str], str, int]]:
    edges: list[tuple[list[str], str, int]] = []

    def walk(node, enclosing: list[str]):
        t = node.type
        if t in SCOPE_TYPES.get(lang, ()):
            nm = _name_of(node)
            new_scope = enclosing + [nm] if _scope_name(node, lang) else enclosing
            for c in node.children:
                if c.is_named:
                    walk(c, new_scope)
            return
        if t == 'call_expression':
            fn = node.child_by_field_name('function')
            callee = fn.text.decode() if fn else '<expr>'
            edges.append((enclosing, callee, node.start_point[0] + 1))
        for c in node.children:
            if c.is_named:
                walk(c, enclosing)

    walk(root, [])
    return edges


def _collect_calls(root, lang: str) -> list[tuple[list[str], str, int]]:
    return _collect_calls_python(root) if lang == 'python' else _collect_calls_ts(root, lang)


class CodeGraph:
    """Code indexing graph: parse + embed + relationship edges + watcher.

    Graph state is namespaced by project so multiple indexed projects never
    produce cross-project edges (D3 fix).
    """

    def __init__(self, repo: EmbeddingRepo):
        self.repo = repo
        self.parsed: dict[str, ParsedFile] = {}
        # (project, module) -> node_id
        self.module_to_node: dict[tuple[str, str], str] = {}
        # (project, module, name) -> (node_id, name)
        self.global_syms: dict[tuple[str, str, str], tuple[str, str]] = {}
        self.edges: list[Edge] = []
        self._lock = threading.RLock()
        # watcher state (single watcher at a time)
        self._watcher_thread: threading.Thread | None = None
        self._watcher_stop = threading.Event()
        self._watcher_root: Path | None = None
        self._watcher_project: str | None = None
        self._watcher_events: list[dict] = []
        # persistence / self-heal state
        self._loaded: bool = False
        self._loaded_projects: set[str] = set()

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
            # v2 persistence (graph survives restart; warm = pure SELECT)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS code_files (
                    node_id  TEXT PRIMARY KEY,
                    project  TEXT NOT NULL,
                    relpath  TEXT NOT NULL,
                    lang     TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    sym_count INTEGER DEFAULT 0
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS code_symbols (
                    rowid     INTEGER PRIMARY KEY,   -- == embed_units.rowid
                    node_id   TEXT NOT NULL,
                    project   TEXT NOT NULL,
                    relpath   TEXT NOT NULL,
                    sym_path  TEXT NOT NULL,
                    name      TEXT NOT NULL,
                    kind      TEXT NOT NULL,
                    start_line INTEGER DEFAULT 0
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS code_edges (
                    project  TEXT NOT NULL,
                    src_node TEXT NOT NULL,
                    src_sym  TEXT NOT NULL,
                    dst_node TEXT NOT NULL,
                    dst_sym  TEXT NOT NULL,
                    kind     TEXT NOT NULL,
                    PRIMARY KEY (project, src_node, src_sym, dst_node, dst_sym, kind)
                )
            ''')
            conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_code_files_proj ON code_files(project)')
            conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_code_sym_proj ON code_symbols(project)')
            conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_code_sym_name ON code_symbols(name)')
            conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_code_edges_proj ON code_edges(project)')
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

    def remove_project(self, project: str) -> dict:
        """Drop a project's symbols, edges, and embeddings (in-memory + DB)."""
        with self._lock:
            prefix = f'code:{project}::'
            gone = [nid for nid in self.parsed if nid.startswith(prefix)]
            for nid in gone:
                self.parsed.pop(nid, None)
            # surgical removal (don't _rebuild_global_syms — self.parsed may be
            # incomplete after a DB warm; global_syms/module_to_node are the
            # source of truth in that state)
            self.global_syms = {k: v for k, v in self.global_syms.items()
                                if k[0] != project}
            self.module_to_node = {k: v for k, v in self.module_to_node.items()
                                   if k[0] != project}
            self.edges = [e for e in self.edges
                          if not e.src_node.startswith(prefix)
                          and not e.dst_node.startswith(prefix)]
            self._loaded_projects.discard(project)
        removed = 0
        with self.repo._get_connection() as conn:
            cur = conn.execute(
                'SELECT rowid FROM embed_units WHERE node_id LIKE ?',
                (prefix + '%',),
            )
            rowids = [r['rowid'] for r in cur.fetchall()]
            if rowids:
                ph = ','.join('?' * len(rowids))
                conn.execute(f'DELETE FROM embed_vec WHERE rowid IN ({ph})', rowids)
                conn.execute(f'DELETE FROM embed_units WHERE rowid IN ({ph})', rowids)
                removed = len(rowids)
            conn.execute('DELETE FROM code_projects WHERE project = ?', (project,))
            conn.execute('DELETE FROM code_files    WHERE project = ?', (project,))
            conn.execute('DELETE FROM code_symbols  WHERE project = ?', (project,))
            conn.execute('DELETE FROM code_edges    WHERE project = ?', (project,))
            conn.commit()
        return {'project': project, 'files_removed': len(gone),
                'units_removed': removed}

    def list_projects(self) -> list[dict]:
        try:
            with self.repo._get_connection() as conn:
                rows = conn.execute(
                    'SELECT project, root_path, updated_at FROM code_projects '
                    'ORDER BY project'
                ).fetchall()
        except Exception:
            return []
        return [dict(r) for r in rows]

    # ---- module resolution (direct + unique suffix match) ----

    def _resolve_module(self, project: str, module_key: str) -> str | None:
        """node_id for an imported module. Direct key first; else a UNIQUE
        suffix match (so 'poly.research.stats' resolves the in-project
        'src.poly.research.stats' even when the import omits the 'src' root).
        Unambiguous-only — preserves edge precision."""
        nid = self.module_to_node.get((project, module_key))
        if nid:
            return nid
        if not module_key:
            return None
        suf = '.' + module_key
        hits = [v for (p, m), v in self.module_to_node.items()
                if p == project and (m == module_key or m.endswith(suf))]
        return hits[0] if len(hits) == 1 else None

    def _import_to_module_key(self, imp_module: str, pf: ParsedFile) -> str:
        """Normalize an import module string to a module key for resolution.
        TS relative imports ('../../lib/x') are resolved against the importing
        file's directory; everything else is treated as a dotted path."""
        m = imp_module.replace('\\', '/').strip()
        if m.startswith('.') and pf.lang in ('typescript', 'tsx'):
            import posixpath
            base = posixpath.dirname(pf.relpath)
            resolved = posixpath.normpath(posixpath.join(base, m))
            return _module_path(resolved)  # also strips any ext
        return _module_path(m)

    def _resolve_sym(self, project: str, module_key: str,
                     name: str) -> tuple[str, str] | None:
        """(node_id, name) for an imported/called symbol. Direct module key
        first; else a UNIQUE suffix match on the module (so an import written
        as 'poly.research.stats' resolves a symbol whose owning file registered
        as 'src.poly.research.stats'). Unambiguous-only — preserves precision."""
        direct = self.global_syms.get((project, module_key, name))
        if direct:
            return direct
        if not module_key:
            return None
        suf = '.' + module_key
        hits = [v for (p, m, n), v in self.global_syms.items()
                if p == project and n == name
                and (m == module_key or m.endswith(suf))]
        return hits[0] if len(hits) == 1 else None

    # ---- persistence (graph survives restart; warm = pure SELECT) ----

    def _persist_project(self, project: str, pfs: list[ParsedFile]) -> None:
        """Write code_files / code_symbols / code_edges for `project`,
        replacing prior rows. Symbols are sourced from embed_units (rowids are
        shared with the vector table), so call AFTER embedding."""
        prefix = f'code:{project}::'
        with self.repo._get_connection() as conn:
            conn.execute('DELETE FROM code_files WHERE project = ?', (project,))
            conn.execute('DELETE FROM code_symbols WHERE project = ?', (project,))
            conn.execute('DELETE FROM code_edges WHERE project = ?', (project,))
            for pf in pfs:
                conn.execute(
                    'INSERT INTO code_files(node_id,project,relpath,lang,file_hash,sym_count) '
                    'VALUES (?,?,?,?,?,?)',
                    (pf.node_id, project, pf.relpath, pf.lang, pf.file_hash, len(pf.symbols)))
            # symbols: pull rowids from embed_units (shared with vectors)
            rows = conn.execute(
                'SELECT rowid, node_id, heading_path, unit_type, pos '
                'FROM embed_units WHERE node_id LIKE ?',
                (prefix + '%',)).fetchall()
            for r in rows:
                relpath = r['node_id'].split('::', 1)[-1]
                name = r['heading_path'].split(' > ')[-1]
                conn.execute(
                    'INSERT OR REPLACE INTO code_symbols '
                    '(rowid,node_id,project,relpath,sym_path,name,kind,start_line) '
                    'VALUES (?,?,?,?,?,?,?,?)',
                    (r['rowid'], r['node_id'], project, relpath,
                     r['heading_path'], name, r['unit_type'], r['pos']))
            # edges (in-memory, already namespaced by project)
            for e in self.edges:
                if e.src_node.startswith(prefix) or e.dst_node.startswith(prefix):
                    conn.execute(
                        'INSERT OR IGNORE INTO code_edges '
                        '(project,src_node,src_sym,dst_node,dst_sym,kind) '
                        'VALUES (?,?,?,?,?,?)',
                        (project, e.src_node, e.src_sym, e.dst_node, e.dst_sym, e.kind))
            conn.commit()

    def _load_project(self, project: str) -> bool:
        """Populate in-memory graph (parsed + module maps + edges) from DB.
        No parse, no embed — cheap. Returns True if the project had data.
        Reconstructs minimal ParsedFile objects so that _rebuild_global_syms
        (called by index_project/remove_project) always has a complete picture."""
        from collections import defaultdict
        with self.repo._get_connection() as conn:
            frows = conn.execute(
                'SELECT node_id, relpath, lang, file_hash FROM code_files '
                'WHERE project = ?', (project,)).fetchall()
            if not frows:
                return False
            srows = conn.execute(
                'SELECT node_id, sym_path, name, kind, start_line FROM code_symbols '
                'WHERE project = ?', (project,)).fetchall()
            erows = conn.execute(
                'SELECT src_node,src_sym,dst_node,dst_sym,kind FROM code_edges '
                'WHERE project = ?', (project,)).fetchall()
            root = conn.execute(
                'SELECT root_path FROM code_projects WHERE project = ?',
                (project,)).fetchone()

        root_path = root['root_path'] if root else ''
        syms_by_node: dict[str, list] = defaultdict(list)
        for r in srows:
            syms_by_node[r['node_id']].append(r)

        for r in frows:
            nid = r['node_id']
            file_syms = tuple(
                Symbol(name=s['name'], kind=s['kind'], path=s['sym_path'],
                       signature='', docstring='', body_excerpt='',
                       start_line=s['start_line'], end_line=0)
                for s in syms_by_node.get(nid, ()))
            pf = ParsedFile(
                node_id=nid, project=project, root=root_path,
                relpath=r['relpath'], lang=r['lang'], file_hash=r['file_hash'],
                symbols=file_syms, imports=())
            self.parsed[nid] = pf
            self._refresh_global_syms(pf)

        for r in erows:
            self.edges.append(Edge(r['src_node'], r['src_sym'],
                                   r['dst_node'], r['dst_sym'], r['kind']))
        self.edges = list(dict.fromkeys(self.edges))  # dedup across loads
        self._loaded_projects.add(project)
        return True

    def _load_all(self) -> list[str]:
        loaded = []
        for p in self.list_projects():
            if self._load_project(p['project']):
                loaded.append(p['project'])
        self._loaded = True
        return loaded

    def _ensure_loaded(self) -> None:
        """Self-heal: lazily hydrate the in-memory graph from DB on first use."""
        if not self._loaded:
            self._load_all()

    def warm_all(self) -> list[str]:
        """On startup, hydrate the in-memory graph for all registered projects
        from DB (pure SELECT — no re-parse, no re-embed)."""
        loaded = self._load_all()
        return loaded

    def warm(self, root: Path, project: str,
             exclude: set[str] | None = None) -> dict:
        """Load a project's graph from DB into memory (no re-parse). `root` is
        accepted for API compatibility and recorded for the watcher."""
        t0 = time.perf_counter()
        ok = self._load_project(project)
        self._loaded = True
        if ok:
            self._watcher_root = Path(root)
        return {
            'files': 0, 'edges': len([e for e in self.edges
                                      if project in e.src_node]),
            'ms': round((time.perf_counter() - t0) * 1000), 'loaded': ok,
        }

    # ---- bulk index ----

    def _glob_sources(self, root: Path, exclude: set[str]) -> list[Path]:
        files: list[Path] = []
        for ext in SUPPORTED_EXTS:
            for f in root.rglob(f'*{ext}'):
                if not any(p in exclude for p in f.parts):
                    files.append(f)
        files.sort()
        return files

    def index_project(
        self, root: Path, project: str,
        exclude: set[str] | None = None,
        *, replace: bool = True, incremental: bool = True,
    ) -> dict:
        root = Path(root)
        exclude = exclude or DEFAULT_EXCLUDE
        t0 = time.perf_counter()

        if replace:
            self.remove_project(project)

        files = self._glob_sources(root, exclude)
        proj_pfs: list[ParsedFile] = []
        for f in files:
            try:
                pf = parse_file(f, root, project)
            except Exception:
                continue
            self.parsed[pf.node_id] = pf
            proj_pfs.append(pf)

        # incremental prune: drop files that were in the project but no longer on disk
        if incremental and not replace:
            self._prune_deleted(project, {pf.relpath for pf in proj_pfs})

        self._rebuild_global_syms()
        with self._lock:
            prefix = f'code:{project}::'
            self.edges = [e for e in self.edges
                          if not e.src_node.startswith(prefix)
                          and not e.dst_node.startswith(prefix)]
            for pf in proj_pfs:
                self._build_file_edges(pf)
            self.edges = list(dict.fromkeys(self.edges))  # dedup (frozen=hashable)

        t_parse = time.perf_counter() - t0
        t1 = time.perf_counter()

        # embed — incremental skips unchanged files (replace=True re-embeds all)
        items: list[tuple[str, str, list[EmbedUnit]]] = []
        skipped = 0
        for pf in proj_pfs:
            if incremental and self.repo.is_indexed(pf.node_id, pf.file_hash):
                skipped += 1
                continue
            units = symbols_to_embed_units(pf)
            if units:
                items.append((pf.node_id, pf.file_hash, units))
        embedded = self.repo.index_units_bulk(items)

        t_embed = time.perf_counter() - t1
        self._persist_project(project, proj_pfs)
        self._register_project(project, root)
        self._loaded_projects.add(project)
        self._loaded = True

        prefix = f'code:{project}::'
        return {
            'files': len(proj_pfs),
            'symbols': embedded,
            'skipped': skipped,
            'edges': len([e for e in self.edges
                          if e.src_node.startswith(prefix)
                          or e.dst_node.startswith(prefix)]),
            'parse_s': round(t_parse, 2),
            'embed_s': round(t_embed, 2),
        }

    def _prune_deleted(self, project: str, current_relpaths: set[str]) -> int:
        """Remove embed_units + symbols for project files no longer on disk."""
        prefix = f'code:{project}::'
        with self.repo._get_connection() as conn:
            rows = conn.execute(
                'SELECT node_id, relpath FROM code_files WHERE project = ?',
                (project,)).fetchall()
            gone_nodes = [r['node_id'] for r in rows
                          if r['relpath'] not in current_relpaths]
            for nid in gone_nodes:
                self.parsed.pop(nid, None)
                self.repo.remove_node(nid)
            if gone_nodes:
                conn.execute(
                    'DELETE FROM code_files WHERE node_id IN (%s)'
                    % ','.join('?' * len(gone_nodes)), gone_nodes)
                conn.execute(
                    'DELETE FROM code_symbols WHERE node_id IN (%s)'
                    % ','.join('?' * len(gone_nodes)), gone_nodes)
                conn.commit()
            return len(gone_nodes)

    # ---- incremental re-index ----

    def reindex_file(self, path: Path, root: Path, project: str) -> dict:
        t0 = time.perf_counter()
        pf = parse_file(path, root, project)

        if self.repo.is_indexed(pf.node_id, pf.file_hash):
            return {'status': 'skipped', 'node_id': pf.node_id,
                    'ms': round((time.perf_counter() - t0) * 1000)}

        with self._lock:
            # surgical update — don't call _rebuild_global_syms (that assumes
            # self.parsed is fully populated, which is false after a DB warm)
            mod_key = _module_path(pf.relpath)
            self.global_syms = {k: v for k, v in self.global_syms.items()
                                if not (k[0] == project and k[1] == mod_key)}
            self.module_to_node.pop((project, mod_key), None)
            self.parsed[pf.node_id] = pf
            self._refresh_global_syms(pf)
            self.edges = [e for e in self.edges
                          if e.src_node != pf.node_id and e.dst_node != pf.node_id]
            self._build_file_edges(pf)
            self.edges = list(dict.fromkeys(self.edges))  # dedup

        units = symbols_to_embed_units(pf)
        self.repo.index_units(pf.node_id, pf.file_hash, units)
        self._persist_file(pf)                       # incremental DB update for this file
        ms = round((time.perf_counter() - t0) * 1000)
        return {'status': 'reindexed', 'node_id': pf.node_id,
                'symbols': len(units), 'ms': ms}

    def _persist_file(self, pf: ParsedFile) -> None:
        """Targeted DB update for a single re-indexed file (used by the watcher)."""
        nid = pf.node_id
        with self.repo._get_connection() as conn:
            conn.execute(
                'INSERT OR REPLACE INTO code_files '
                '(node_id,project,relpath,lang,file_hash,sym_count) VALUES (?,?,?,?,?,?)',
                (nid, pf.project, pf.relpath, pf.lang, pf.file_hash, len(pf.symbols)))
            conn.execute('DELETE FROM code_symbols WHERE node_id = ?', (nid,))
            for r in conn.execute(
                    'SELECT rowid,heading_path,unit_type,pos FROM embed_units '
                    'WHERE node_id = ?', (nid,)).fetchall():
                conn.execute(
                    'INSERT OR REPLACE INTO code_symbols '
                    '(rowid,node_id,project,relpath,sym_path,name,kind,start_line) '
                    'VALUES (?,?,?,?,?,?,?,?)',
                    (r['rowid'], nid, pf.project, pf.relpath, r['heading_path'],
                     r['heading_path'].split(' > ')[-1], r['unit_type'], r['pos']))
            conn.execute(
                'DELETE FROM code_edges WHERE src_node = ? OR dst_node = ?',
                (nid, nid))
            for e in self.edges:
                if e.src_node == nid or e.dst_node == nid:
                    conn.execute(
                        'INSERT OR IGNORE INTO code_edges '
                        '(project,src_node,src_sym,dst_node,dst_sym,kind) VALUES (?,?,?,?,?,?)',
                        (pf.project, e.src_node, e.src_sym, e.dst_node, e.dst_sym, e.kind))
            conn.commit()

    # ---- per-file helpers ----

    def _rebuild_global_syms(self) -> None:
        self.module_to_node.clear()
        self.global_syms.clear()
        for pf in self.parsed.values():
            self._refresh_global_syms(pf)

    def _refresh_global_syms(self, pf: ParsedFile) -> None:
        mod = _module_path(pf.relpath)
        self.module_to_node[(pf.project, mod)] = pf.node_id
        for s in pf.symbols:
            if s.kind in ('function', 'class'):
                self.global_syms[(pf.project, mod, s.name)] = (pf.node_id, s.name)

    def _build_file_edges(self, pf: ParsedFile) -> None:
        node_id = pf.node_id
        project = pf.project
        owner = None
        for s in pf.symbols:
            if s.kind == 'class':
                owner = s.name
            elif s.kind == 'method' and owner:
                self.edges.append(Edge(node_id, owner, node_id, s.name, 'contains'))
        # imports -> resolve within the same project only (no cross-project edges)
        file_imports: dict[str, str] = {}
        for imp in pf.imports:
            mod_key = self._import_to_module_key(imp.module, pf)
            tgt = self._resolve_module(project, mod_key)
            if tgt:
                file_imports.update({n: mod_key for n in imp.names})
                for n in imp.names:
                    tgt_sym = self._resolve_sym(project, mod_key, n)
                    if tgt_sym:
                        self.edges.append(
                            Edge(node_id, '(file)', tgt_sym[0], n, 'import'))
        # re-read source from THIS file's root (D2 fix: pf.root, not a single root)
        try:
            src = (Path(pf.root) / pf.relpath).read_bytes()
        except Exception:
            return
        parser = get_parser(pf.lang)
        tree = parser.parse(src)
        sym_names = {s.name for s in pf.symbols}
        method_by_owner: dict[str, dict[str, str]] = defaultdict(dict)
        for s in pf.symbols:
            if s.kind == 'method':
                parts = s.path.split(' > ')
                if len(parts) >= 2:
                    method_by_owner[parts[-2]][s.name] = s.name
        for enclosing, callee, _line in _collect_calls(tree.root_node, pf.lang):
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
                tgt = (self._resolve_sym(project, srcmod, callee_clean.split('.')[-1])
                       or self._resolve_sym(project, srcmod, head))
                if tgt:
                    resolved = tgt
            if not resolved and callee_clean in sym_names:
                resolved = (node_id, callee_clean)
            if resolved:
                self.edges.append(
                    Edge(node_id, caller_sym, resolved[0], resolved[1], 'call'))

    # ---- query API ----

    def search(self, query: str, k: int = 8, *,
               project: str | None = None, kind: str | None = None,
               rerank: bool = True) -> list[dict]:
        """Semantic search across indexed code symbols by meaning.
        kind: filter by node type ('function' | 'class' | 'method')."""
        self._ensure_loaded()
        # fetch dense candidates (more if scoping, so the filter still leaves k)
        fetch = max(k * 3, 24) if (project or kind) else k
        results = self.repo.search(query, k=fetch, unit_type=kind, rerank=False)
        results = [r for r in results if r['node_id'].startswith('code:')]
        if project:
            results = [r for r in results
                       if r['node_id'].startswith(f'code:{project}::')]
        if rerank and len(results) > 1:
            results = self.repo._rerank(query, results, k)
        else:
            results = results[:k]
        return results

    def neighbors(self, node_id: str, sym: str, *, depth: int = 1) -> dict:
        """Direct (depth=1) or transitive (depth>1) relationship lookup.
        depth=1 uses in-memory edges (fast, backward compatible).
        depth>1 uses recursive CTE on code_edges (multi-hop traversal)."""
        self._ensure_loaded()
        if depth <= 1:
            return self._neighbors_1hop(node_id, sym)
        callers = self.transitive_callers(node_id, sym, depth=depth)
        callees = self.transitive_callees(node_id, sym, depth=depth)
        imports = self._neighbors_1hop(node_id, sym)['imports']
        return {
            'callers': [(c['node_id'], c['sym']) for c in callers],
            'callees': [(c['node_id'], c['sym']) for c in callees],
            'imports': imports,
        }

    def _neighbors_1hop(self, node_id: str, sym: str) -> dict:
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

    # ================================================================
    # GRAPH QUERY LAYER
    # Impl: SQLite recursive CTE on code_edges table.
    # Migration: replace method bodies with PG SQL/PGQ GRAPH_TABLE patterns:
    #   callers → MATCH (src)-[:CALL*1..depth]->(target WHERE node_id=? AND sym=?)
    #   callees → MATCH (anchor WHERE ...)-[:CALL*1..depth]->(dst)
    # Method signatures are SQL-agnostic — swap bodies only when migrating.
    # ================================================================

    def _transitive_reachable(
        self, node_id: str, sym: str, *,
        direction: str,                    # 'in' (callers) | 'out' (callees) | 'both'
        edge_kinds: tuple[str, ...] = ('call',),
        depth: int = 3,
    ) -> list[dict]:
        """All nodes within `depth` hops of (node_id, sym).
        Returns [{node_id, sym, hop}] sorted by proximity. Cycle-safe."""
        directions = ('in', 'out') if direction == 'both' else (direction,)
        best: dict[tuple[str, str], dict] = {}
        for d in directions:
            for r in self._transitive_reachable_one(
                node_id, sym, direction=d, edge_kinds=edge_kinds, depth=depth,
            ):
                key = (r['node_id'], r['sym'])
                if key not in best or r['hop'] < best[key]['hop']:
                    best[key] = r
        return sorted(best.values(), key=lambda r: r['hop'])

    def _transitive_reachable_one(
        self, node_id: str, sym: str, *,
        direction: str,
        edge_kinds: tuple[str, ...],
        depth: int,
    ) -> list[dict]:
        """Single-direction recursive CTE on code_edges.

        For 'in' (callers): follow dst→src edges backwards from the anchor.
        For 'out' (callees): follow src→dst edges forwards from the anchor.
        Cycle-safe via path tracking; bounded by depth."""
        if direction == 'in':
            seed_col, next_col, join_col = 'dst', 'src', 'dst'
        else:
            seed_col, next_col, join_col = 'src', 'dst', 'src'

        ph = ','.join('?' * len(edge_kinds))
        sql = f'''
            WITH RECURSIVE chain(node_id, sym, hop, path) AS (
                SELECT e.{next_col}_node, e.{next_col}_sym, 1,
                       ',' || e.{next_col}_node || ':' || e.{next_col}_sym || ','
                FROM code_edges e
                WHERE e.{seed_col}_node = ? AND e.{seed_col}_sym = ?
                  AND e.kind IN ({ph})
                UNION ALL
                SELECT e.{next_col}_node, e.{next_col}_sym, c.hop + 1,
                       c.path || e.{next_col}_node || ':' || e.{next_col}_sym || ','
                FROM code_edges e
                JOIN chain c
                  ON e.{join_col}_node = c.node_id AND e.{join_col}_sym = c.sym
                WHERE e.kind IN ({ph})
                  AND c.hop < ?
                  AND c.path NOT LIKE '%,' || e.{next_col}_node || ':'
                                       || e.{next_col}_sym || ',%'
            )
            SELECT node_id, sym, MIN(hop) AS hop
            FROM chain
            GROUP BY node_id, sym
            ORDER BY hop
        '''
        params = [node_id, sym, *edge_kinds, *edge_kinds, depth]
        with self.repo._get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [{'node_id': r['node_id'], 'sym': r['sym'], 'hop': r['hop']}
                for r in rows]

    def transitive_callers(self, node_id: str, sym: str, *,
                           depth: int = 3) -> list[dict]:
        """Who calls X, directly or indirectly (up to `depth` hops)?
        Returns [{node_id, sym, hop}] sorted by proximity."""
        self._ensure_loaded()
        return self._transitive_reachable(
            node_id, sym, direction='in', edge_kinds=('call',), depth=depth)

    def transitive_callees(self, node_id: str, sym: str, *,
                           depth: int = 3) -> list[dict]:
        """What does X call, directly or indirectly (up to `depth` hops)?
        Returns [{node_id, sym, hop}] sorted by proximity."""
        self._ensure_loaded()
        return self._transitive_reachable(
            node_id, sym, direction='out', edge_kinds=('call',), depth=depth)

    def search_near(
        self, query: str, anchor_node: str, anchor_sym: str, *,
        depth: int = 2, k: int = 8,
        project: str | None = None, kind: str | None = None,
    ) -> list[dict]:
        """Semantic search constrained to the graph neighborhood of a node.
        Finds code matching `query` but only within `depth` hops of the anchor.
        Answers: 'find error-handling patterns in code that depends on X'."""
        self._ensure_loaded()
        # 1. graph neighborhood (call + import edges, both directions)
        reachable = self._transitive_reachable(
            anchor_node, anchor_sym,
            direction='both', edge_kinds=('call', 'import'), depth=depth)
        near_nodes = {anchor_node}
        near_nodes.update(r['node_id'] for r in reachable)
        if len(near_nodes) <= 1:
            return []  # anchor has no resolved edges — empty neighborhood
        # 2. semantic search with generous fetch, filter to neighborhood
        fetch = max(k * 5, 40)
        results = self.repo.search(query, k=fetch, unit_type=kind, rerank=False)
        results = [r for r in results if r['node_id'] in near_nodes]
        if project:
            results = [r for r in results
                       if r['node_id'].startswith(f'code:{project}::')]
        # 3. rerank the filtered set
        if len(results) > 1:
            results = self.repo._rerank(query, results, k)
        return results[:k]

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

        def src_filter(change: Change, path: str) -> bool:
            p = Path(path)
            return (p.suffix.lower() in SUPPORTED_EXTS
                    and not any(x in p.parts for x in exclude))

        try:
            for changes in watch(root, debounce=300, watch_filter=src_filter):
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
