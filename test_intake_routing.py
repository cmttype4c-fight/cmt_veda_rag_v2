"""
test_intake_routing.py
----------------------
Regression tests for the live bug:

    POST /admin/intake/bulk/approve -> 500
    psycopg2.errors.InvalidTextRepresentation: invalid input syntax for
    type uuid: "bulk"

Cause: `POST /admin/intake/{document_id}/approve` was declared BEFORE
`POST /admin/intake/bulk/approve`; FastAPI/Starlette dispatch to the first
matching route, so "bulk" became the document_id. SQLite accepts any text as
an id, which is why the earlier suites never saw it; PostgreSQL does not.

Two layers:

  A. STATIC (no third-party dependencies, always runs). Reads the route
     declarations out of main.py with `ast` and applies Starlette's
     first-match rule. Includes a sanity check that the emulator reproduces
     the original bug when the old declaration order is restored.

  B. HTTP (needs `fastapi`; reported as SKIPPED where it is not installed,
     e.g. the development sandbox -- NOT counted as passed). Drives the real
     app through TestClient against a database that, like PostgreSQL, rejects
     any non-UUID document id, and records every id it is asked for.

Run: python3 test_intake_routing.py   (or pytest)
"""

import ast
import os
import re
import sys
import shutil
import tempfile
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PREFIX = "/admin/intake"
_METHODS = {"get", "post", "patch", "put", "delete"}


# ----------------------------------------------------------------- A. static
def _declared_intake_routes():
    """[(METHOD, full_path, handler_name)] in declaration order, plus the
    reserved-segment set, parsed from main.py without importing it."""
    tree = ast.parse(open(os.path.join(HERE, "main.py"), encoding="utf-8").read())
    routes, reserved = [], set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_RESERVED_INTAKE_SEGMENTS" for t in node.targets):
            call = node.value  # frozenset({...})
            reserved = {e.value for e in ast.walk(call) if isinstance(e, ast.Constant)}
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                        and isinstance(dec.func.value, ast.Name) and dec.func.value.id == "intake_router"
                        and dec.func.attr in _METHODS and dec.args
                        and isinstance(dec.args[0], ast.Constant)):
                    routes.append((dec.func.attr.upper(), PREFIX + dec.args[0].value, node.name))
    return routes, reserved


def _compile(path):
    # Starlette: "{name}" matches one path segment ([^/]+)
    return re.compile("^" + re.sub(r"\{[^}/]+\}", r"[^/]+", path) + "$")


def _dispatch(routes, method, path):
    """Starlette's rule: first route whose path matches AND whose method is
    allowed wins (a path match with the wrong method is only 'partial')."""
    for m, p, handler in routes:
        if _compile(p).match(path) and m == method:
            return handler
    return None


def test_bulk_approve_dispatches_to_bulk_handler():
    routes, _ = _declared_intake_routes()
    assert routes, "no intake routes found in main.py"
    assert _dispatch(routes, "POST", f"{PREFIX}/bulk/approve") == "intake_bulk_approve"
    assert _dispatch(routes, "POST", f"{PREFIX}/bulk") == "intake_bulk_register"
    print("PASS: POST /admin/intake/bulk/approve -> bulk handler (not {document_id}/approve)")


def test_single_approve_still_dispatches_to_single_handler():
    routes, _ = _declared_intake_routes()
    doc_id = str(uuid.uuid4())
    assert _dispatch(routes, "POST", f"{PREFIX}/{doc_id}/approve") == "intake_approve"
    assert _dispatch(routes, "POST", f"{PREFIX}/{doc_id}/reject") == "intake_reject"
    assert _dispatch(routes, "POST", f"{PREFIX}/{doc_id}/retry") == "intake_retry"
    assert _dispatch(routes, "GET", f"{PREFIX}/{doc_id}") == "intake_get"
    assert _dispatch(routes, "PATCH", f"{PREFIX}/{doc_id}") == "intake_patch"
    print("PASS: POST /admin/intake/{document_id}/approve (and siblings) still dispatch correctly")


def test_static_routes_are_declared_before_dynamic_ones():
    routes, _ = _declared_intake_routes()
    order = [(m, p) for m, p, _ in routes]
    first_dynamic_post = min(i for i, (m, p) in enumerate(order) if m == "POST" and "{" in p)
    for static in (("POST", f"{PREFIX}/bulk"), ("POST", f"{PREFIX}/bulk/approve"), ("POST", f"{PREFIX}/pdf")):
        assert order.index(static) < first_dynamic_post, f"{static} declared after a dynamic POST route"
    print("PASS: every static POST route is declared before the first dynamic POST route")


def test_bulk_is_a_reserved_segment_never_a_document_id():
    routes, reserved = _declared_intake_routes()
    assert {"bulk", "pdf"} <= reserved, reserved
    # Defence in depth: even a static-looking path that has no static handler
    # (so it falls through to the dynamic route) is rejected by the guard.
    assert _dispatch(routes, "POST", f"{PREFIX}/bulk/retry") == "intake_retry"
    assert "bulk" in reserved  # -> _intake_document_id raises 404 before the handler body runs
    src = open(os.path.join(HERE, "main.py"), encoding="utf-8").read()
    for handler in ("intake_get", "intake_patch", "intake_reject", "intake_approve", "intake_retry"):
        assert re.search(rf"def {handler}\(document_id: IntakeDocumentId", src), f"{handler} lacks the reserved-segment guard"
    print('PASS: "bulk"/"pdf" are reserved; all dynamic handlers use the guard')


def test_emulator_reproduces_the_original_bug_with_old_order():
    """Sanity check on the test itself: with the OLD declaration order the
    emulator sends bulk/approve to the single-document handler."""
    routes, _ = _declared_intake_routes()
    old = [r for r in routes if r[2] != "intake_bulk_approve"]
    old.append(("POST", f"{PREFIX}/bulk/approve", "intake_bulk_approve"))  # declared last, as before the fix
    assert _dispatch(old, "POST", f"{PREFIX}/bulk/approve") == "intake_approve"
    print("PASS: emulator reproduces the original collision when the old order is restored")


# ------------------------------------------------------------------- B. HTTP
def _import_app():
    try:
        from fastapi.testclient import TestClient  # noqa: F401
        import main
        return main
    except ImportError:
        return None


def _is_uuid(value):
    try:
        uuid.UUID(str(value))
        return True
    except ValueError:
        return False


def test_http_bulk_and_single_approve_against_uuid_strict_db():
    main = _import_app()
    if main is None:
        print("SKIPPED: fastapi is not installed here -- HTTP-level routing test NOT run")
        return "skipped"
    from fastapi.testclient import TestClient
    from db import SqliteBackend
    from embeddings import HashingTfidfEmbeddings
    from vector_store import NumpyVectorStore
    from worker import InlineQueueAdapter
    from ingestion_pipeline import IngestionInput
    import intake

    class UuidStrictDB(SqliteBackend):
        """Behaves like PostgreSQL for ids: non-UUID -> error. Records every id."""
        seen = []

        def get_document(self, document_id):
            UuidStrictDB.seen.append(document_id)
            if not _is_uuid(document_id):
                raise RuntimeError(f'invalid input syntax for type uuid: "{document_id}"')
            return super().get_document(document_id)

        def transition_document_state(self, document_id, new_state, actor):
            if not _is_uuid(document_id):
                raise RuntimeError(f'invalid input syntax for type uuid: "{document_id}"')
            return super().transition_document_state(document_id, new_state, actor)

    tmp = tempfile.mkdtemp(prefix="cmt_route_")
    try:
        db = UuidStrictDB(os.path.join(tmp, "t.db"))
        emb = HashingTfidfEmbeddings(n_features=512)
        adapter = InlineQueueAdapter(db, emb, NumpyVectorStore(os.path.join(tmp, "vs", "s"), dim=512))
        main.state.update(db=db, inline_queue_adapter=adapter, enqueue_fn=adapter.enqueue_fn)
        main.config.RAG_ADMIN_API_KEY = "admin-" + uuid.uuid4().hex   # test-only random value
        headers = {"X-Admin-API-Key": main.config.RAG_ADMIN_API_KEY}
        body = ("Charcot-Marie-Tooth disease is a hereditary peripheral neuropathy. " * 40)

        def register(title, doi):
            return intake.register_intake(
                db, IngestionInput(raw_text=body, format="text", title=title, doi=doi), "discovery", actor="t")

        d1, d2 = register("one", "10.1/a"), register("two", "10.1/b")
        client = TestClient(main.app, raise_server_exceptions=False)

        r = client.post(f"{PREFIX}/bulk/approve", headers=headers,
                        json={"document_ids": [d1.document_id], "actor": "admin"})
        assert r.status_code == 200, (r.status_code, r.text)
        assert "bulk" not in UuidStrictDB.seen, UuidStrictDB.seen
        assert r.json().get("success_count") == 1, r.json()

        r = client.post(f"{PREFIX}/{d2.document_id}/approve", headers=headers, json={"actor": "admin"})
        assert r.status_code == 200, (r.status_code, r.text)
        assert r.json()["document_id"] == d2.document_id

        r = client.post(f"{PREFIX}/bulk/retry", headers=headers, json={"actor": "admin"})
        assert r.status_code == 404, (r.status_code, r.text)
        assert "bulk" not in UuidStrictDB.seen
        print("PASS: HTTP bulk/approve=200, single approve=200, bulk/retry=404; 'bulk' never reached the DB")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


_TESTS = [
    test_bulk_approve_dispatches_to_bulk_handler,
    test_single_approve_still_dispatches_to_single_handler,
    test_static_routes_are_declared_before_dynamic_ones,
    test_bulk_is_a_reserved_segment_never_a_document_id,
    test_emulator_reproduces_the_original_bug_with_old_order,
    test_http_bulk_and_single_approve_against_uuid_strict_db,
]

if __name__ == "__main__":
    failures = skipped = 0
    for t in _TESTS:
        try:
            if t() == "skipped":
                skipped += 1
        except Exception as e:
            failures += 1
            print(f"FAIL: {t.__name__}: {e!r}")
    print(f"\n{len(_TESTS) - failures - skipped}/{len(_TESTS)} passed, {skipped} skipped (need fastapi), {failures} failed")
    sys.exit(1 if failures else 0)
