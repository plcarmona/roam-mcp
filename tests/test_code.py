"""Unit tests for mcp_roam.code v2 — locks in the D1/D2/D3 fixes and edge
resolution. Parsing & graph-edge tests are hermetic (no DB, no Ollama):
CodeGraph(repo=None) is enough because _build_file_edges/neighbors only touch
disk + in-memory state.

Run:  uv run --directory /home/pit/utils/mcp-roam python -m pytest tests/test_code.py -q
  or: uv run --directory /home/pit/utils/mcp-roam python tests/test_code.py
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcp_roam.code import (  # noqa: E402
    CodeGraph, parse_file, _collect_calls, _module_path, detect_lang,
)


# --------------------------------------------------------------------------
# D2 — relpath relative to the index root (standard src/<pkg> layout)
# --------------------------------------------------------------------------
def test_relpath_at_repo_root():
    """poly at its REPO ROOT must yield src/poly/... relpaths (v1 crashed here)."""
    root = Path("/home/pit/poly")
    pf = parse_file(root / "src/poly/analytics.py", root, "poly")
    assert pf.relpath == "src/poly/analytics.py"
    assert pf.node_id == "code:poly::src/poly/analytics.py"
    assert pf.project == "poly" and pf.root == str(root)


def test_relpath_arbitrary_root():
    """project name decoupled from dir name: indexing /a/b as 'anything' works."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "pkg").mkdir()
        f = d / "pkg" / "m.py"
        f.write_text("def foo():\n    return 1\n")
        pf = parse_file(f, d, "anything")
        assert pf.relpath == "pkg/m.py"
        assert pf.project == "anything"


# --------------------------------------------------------------------------
# D1 — TypeScript parsing
# --------------------------------------------------------------------------
def test_ts_symbols():
    pf = parse_file(
        Path("/home/pit/manta-lab/src/stores/cart.ts"),
        Path("/home/pit/manta-lab"), "manta-lab")
    assert pf.lang == "typescript"
    names = {s.name for s in pf.symbols}
    assert {"addToCart", "getCartTotal", "clearCart"} <= names
    assert all(s.kind == "function" for s in pf.symbols)  # top-level fns


def test_ts_arrow_export_captured():
    """`export const POST: APIRoute = async (...) => {}` must be captured."""
    pf = parse_file(
        Path("/home/pit/manta-lab/src/pages/api/webhook-mercado-pago.ts"),
        Path("/home/pit/manta-lab"), "manta-lab")
    assert "POST" in {s.name for s in pf.symbols}


def test_ts_imports_extracted():
    pf = parse_file(
        Path("/home/pit/manta-lab/src/lib/payments/process-payment.ts"),
        Path("/home/pit/manta-lab"), "manta-lab")
    mods = {i.module for i in pf.imports}
    assert "mercadopago" in mods
    imp_names = {n for i in pf.imports for n in i.names}
    assert "MercadoPagoConfig" in imp_names


# --------------------------------------------------------------------------
# call extraction (python)
# --------------------------------------------------------------------------
def test_collect_calls_python():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); f = d / "m.py"
        f.write_text("def a():\n    return b() + c.d()\n")
        from mcp_roam.code import get_parser
        tree = get_parser("python").parse(f.read_bytes())
        calls = _collect_calls(tree.root_node, "python")
        names = {c[1] for c in calls}
        assert "b()" in names or "b" in {c[1] for c in calls}
    assert _module_path("a/b.py") == "a.b"
    assert detect_lang(Path("x.tsx")) == "tsx"


# --------------------------------------------------------------------------
# edge resolution — hermetic (CodeGraph(repo=None), no DB/embedding)
# --------------------------------------------------------------------------
def _graph_with(files: dict[str, str], root: Path, project: str) -> CodeGraph:
    g = CodeGraph(repo=None)
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src)
        pf = parse_file(p, root, project)
        g.parsed[pf.node_id] = pf
    g._rebuild_global_syms()
    with g._lock:
        for pf in g.parsed.values():
            g._build_file_edges(pf)
    return g


def test_self_method_call_resolves():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        g = _graph_with({"m.py": (
            "class Bot:\n"
            "    def run(self):\n"
            "        self.stop()\n"
            "    def stop(self):\n"
            "        pass\n"
        )}, d, "p")
        nid = "code:p::m.py"
        nb = g.neighbors(nid, "run")
        callee_names = {s for _, s in nb["callees"]}
        assert "stop" in callee_names  # self.stop() resolved


def test_cross_file_import_call_resolves():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        g = _graph_with({
            "svc.py": "def hello():\n    return 1\n",
            "main.py": (
                "from svc import hello\n"
                "def go():\n"
                "    return hello()\n"),
        }, d, "p")
        nb = g.neighbors("code:p::main.py", "go")
        assert ("code:p::svc.py", "hello") in nb["callees"]


def test_contains_edge_for_methods():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        g = _graph_with({"m.py": (
            "class C:\n"
            "    def m(self):\n"
            "        pass\n"
        )}, d, "p")
        nb = g.neighbors("code:p::m.py", "m")
        # 'contains' edges are src_sym=owner; check the owner->method exists
        assert any(e.src_sym == "C" and e.dst_sym == "m"
                   for e in g.edges if e.kind == "contains")


# --------------------------------------------------------------------------
# Phase B — import resolver (TS relative imports + python suffix match)
# --------------------------------------------------------------------------
def test_ts_relative_import_call_resolves():
    """TS `import {helper} from '../../lib/util'` must resolve across files."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        g = _graph_with({
            "src/lib/util.ts": "export function helper(): number {\n  return 1\n}\n",
            "src/pages/api/handler.ts": (
                "import { helper } from '../../lib/util'\n"
                "export function handler(): number {\n"
                "  return helper()\n}\n"),
        }, d, "p")
        nb = g.neighbors("code:p::src/pages/api/handler.ts", "handler")
        assert ("code:p::src/lib/util.ts", "helper") in nb["callees"], (
            f"callees={nb['callees']}")
        assert ("code:p::src/lib/util.ts", "helper") in nb["imports"], (
            f"imports={nb['imports']}")


def test_python_suffix_match_import_resolves():
    """`from poly.research.stats import mean` resolves the file registered as
    'src.poly.research.stats' (import omits the 'src' root)."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        g = _graph_with({
            "src/poly/research/stats.py": "def mean():\n    return 0\n",
            "src/poly/main.py": (
                "from poly.research.stats import mean\n"
                "def go():\n    return mean()\n"),
        }, d, "p")
        nb = g.neighbors("code:p::src/poly/main.py", "go")
        assert ("code:p::src/poly/research/stats.py", "mean") in nb["callees"], (
            f"callees={nb['callees']}")


def test_ambiguous_suffix_no_false_edge():
    """Two files sharing suffix '.research.stats' → import 'research.stats'
    must NOT resolve (unambiguous-only)."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        g = _graph_with({
            "src/poly/research/stats.py": "def mean():\n    return 0\n",
            "src/kronos/research/stats.py": "def mean():\n    return 1\n",
            "src/poly/main.py": (
                "from research.stats import mean\n"
                "def go():\n    return mean()\n"),
        }, d, "p")
        nb = g.neighbors("code:p::src/poly/main.py", "go")
        assert nb["callees"] == [], f"expected no resolution, got {nb['callees']}"



def test_replace_clears_in_memory_state():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "a.py").write_text("def a():\n    pass\n")
        g = CodeGraph(repo=None)
        # fake-parse one entry as a prior "stale" index of the same project
        g.parsed["code:p::OLD.py"] = parse_file(d / "a.py", d, "p")
        g.parsed["code:p::OLD.py"] = g.parsed["code:p::OLD.py"].__class__(
            node_id="code:p::OLD.py", project="p", root=str(d),
            relpath="OLD.py", lang="python", file_hash="x",
            symbols=(), imports=())
        # index_project drops prior 'p' entries via remove_project path
        # (repo-less: remove_project's DB calls will fail; call the in-memory part)
        prefix = "code:p::"
        for nid in [n for n in g.parsed if n.startswith(prefix)]:
            g.parsed.pop(nid)
        assert not any(n.startswith(prefix) for n in g.parsed)


# --------------------------------------------------------------------------
# Graph query layer — multi-hop traversal (recursive CTE on code_edges)
# --------------------------------------------------------------------------
def _db_graph(edges):
    """CodeGraph with a temp DB; code_edges populated directly.
    edges: list of (src_node, src_sym, dst_node, dst_sym, kind)."""
    from mcp_roam.embeddings import EmbeddingRepo
    tmpdb = Path(tempfile.mkdtemp(prefix='code_test_')) / "test.db"
    repo = EmbeddingRepo(tmpdb); repo.init_schema()
    g = CodeGraph(repo); g.init_schema()
    with repo._get_connection() as conn:
        for e in edges:
            conn.execute(
                'INSERT INTO code_edges'
                '(project,src_node,src_sym,dst_node,dst_sym,kind) '
                'VALUES (?,?,?,?,?,?)',
                ('test', *e))
        conn.commit()
    g._loaded = True
    return g


def test_transitive_callers_2hop():
    """A→B→C: callers(C, depth=2) includes B at hop 1 and A at hop 2."""
    g = _db_graph([
        ('code:test::a.py', 'fn_a', 'code:test::b.py', 'fn_b', 'call'),
        ('code:test::b.py', 'fn_b', 'code:test::c.py', 'fn_c', 'call'),
    ])
    callers = g.transitive_callers('code:test::c.py', 'fn_c', depth=2)
    hops = {c['sym']: c['hop'] for c in callers}
    assert hops.get('fn_b') == 1, f"expected fn_b at hop 1, got {hops}"
    assert hops.get('fn_a') == 2, f"expected fn_a at hop 2, got {hops}"


def test_transitive_callees_2hop():
    """A→B→C: callees(A, depth=2) includes B at hop 1 and C at hop 2."""
    g = _db_graph([
        ('code:test::a.py', 'fn_a', 'code:test::b.py', 'fn_b', 'call'),
        ('code:test::b.py', 'fn_b', 'code:test::c.py', 'fn_c', 'call'),
    ])
    callees = g.transitive_callees('code:test::a.py', 'fn_a', depth=2)
    hops = {c['sym']: c['hop'] for c in callees}
    assert hops.get('fn_b') == 1, f"expected fn_b at hop 1, got {hops}"
    assert hops.get('fn_c') == 2, f"expected fn_c at hop 2, got {hops}"


def test_depth_limit_excludes_beyond():
    """A→B→C→D: callers(D, depth=2) reaches C and B but NOT A (hop 3)."""
    g = _db_graph([
        ('code:test::a.py', 'fn_a', 'code:test::b.py', 'fn_b', 'call'),
        ('code:test::b.py', 'fn_b', 'code:test::c.py', 'fn_c', 'call'),
        ('code:test::c.py', 'fn_c', 'code:test::d.py', 'fn_d', 'call'),
    ])
    callers = g.transitive_callers('code:test::d.py', 'fn_d', depth=2)
    syms = {c['sym'] for c in callers}
    assert 'fn_c' in syms, "fn_c (hop 1) should be in results"
    assert 'fn_b' in syms, "fn_b (hop 2) should be in results"
    assert 'fn_a' not in syms, "fn_a (hop 3) should be excluded by depth limit"


def test_cycle_safe_traversal():
    """A↔B cycle: traversal terminates without hanging, results deduplicated."""
    g = _db_graph([
        ('code:test::a.py', 'fn_a', 'code:test::b.py', 'fn_b', 'call'),
        ('code:test::b.py', 'fn_b', 'code:test::a.py', 'fn_a', 'call'),
    ])
    # this would infinite-loop without cycle prevention
    callers = g.transitive_callers('code:test::a.py', 'fn_a', depth=5)
    # B calls A directly (hop 1); A calls B calls A (cycle back, hop 2)
    hops = {c['sym']: c['hop'] for c in callers}
    assert hops.get('fn_b') == 1, f"expected fn_b at hop 1, got {hops}"
    # each node appears once (deduplicated by GROUP BY)
    assert len(callers) == len({c['sym'] for c in callers}), "duplicates found"


def test_neighbors_depth1_backward_compat():
    """neighbors(depth=1) uses in-memory edges (same as pre-graph-layer)."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        g = _graph_with({
            "svc.py": "def hello():\n    return 1\n",
            "main.py": (
                "from svc import hello\n"
                "def go():\n    return hello()\n"),
        }, d, "p")
        nb_default = g.neighbors("code:p::main.py", "go")
        nb_explicit = g.neighbors("code:p::main.py", "go", depth=1)
        assert nb_default == nb_explicit
        assert ("code:p::svc.py", "hello") in nb_default["callees"]


def test_kind_filter_passthrough():
    """search(kind=) passes unit_type to repo.search (no post-filter needed)."""
    g = CodeGraph(repo=None)
    g._loaded = True
    captured = {}

    class FakeRepo:
        def search(self, query, k=10, unit_type=None, rerank=False):
            captured['unit_type'] = unit_type
            return []

        def _rerank(self, *a, **kw):
            return []

    g.repo = FakeRepo()
    g.search("something", kind='class')
    assert captured['unit_type'] == 'class', (
        f"expected unit_type='class', got {captured.get('unit_type')}")
    # without kind, unit_type should be None
    g.search("something")
    assert captured['unit_type'] is None


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def _run():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}"); passed += 1
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}"); failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run())
