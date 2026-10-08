"""
test_main_importable.py
-----------------------
`main.py` must IMPORT. Decorators and annotations are evaluated at import time,
so a model class defined below the route that uses it (NameError), or a name
that shadows another import (fastapi.Path vs pathlib.Path), makes
`uvicorn main:app` crash-loop -- exactly what happened with commit 896dddc.

FastAPI is not installable in the development sandbox, so this test:
  * imports the REAL main.py when fastapi is installed (e.g. the container), or
  * otherwise imports the REAL main.py against minimal stand-ins for fastapi /
    pydantic that execute every decorator, annotation and module-level
    statement exactly as Python would, and RECORD the route table.
Either way it then checks the recorded route table with Starlette's
first-match rule. The stand-ins validate nothing about FastAPI itself; they
exist to run main.py's own top level. (Labelled in the output.)

Run: python3 test_main_importable.py   (or pytest)
"""

import importlib
import os
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


class _Router:
    def __init__(self, *a, prefix="", **k):
        self.prefix, self.routes = prefix, []

    def _reg(self, method):
        def deco(path, **kw):
            def wrap(fn):
                self.routes.append((method, self.prefix + path, fn.__name__, kw.get("response_model")))
                return fn
            return wrap
        return deco

    def __getattr__(self, name):
        if name in ("get", "post", "patch", "put", "delete"):
            return self._reg(name.upper())
        raise AttributeError(name)


class _FastAPI(_Router):
    def __init__(self, *a, **k):
        super().__init__()
        self.included = []

    def include_router(self, router):
        self.included.append(router)

    def mount(self, *a, **k):
        pass


def _install_stand_ins():
    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI, fastapi.APIRouter = _FastAPI, _Router
    fastapi.HTTPException = type("HTTPException", (Exception,), {"__init__": lambda self, status_code=500, detail="": None})
    for name in ("Header", "File", "Form", "Depends", "Path"):
        setattr(fastapi, name, lambda *a, **k: None)
    fastapi.UploadFile = type("UploadFile", (), {})
    responses = types.ModuleType("fastapi.responses"); responses.FileResponse = object
    static = types.ModuleType("fastapi.staticfiles"); static.StaticFiles = object
    pydantic = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **kw): self.__dict__.update(kw)
        def model_dump(self, **k): return dict(self.__dict__)
    pydantic.BaseModel = BaseModel
    mods = {"fastapi": fastapi, "fastapi.responses": responses, "fastapi.staticfiles": static, "pydantic": pydantic}
    saved = {k: sys.modules.get(k) for k in mods}
    sys.modules.update(mods)
    return saved


def _restore(saved):
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


def _load_main():
    sys.modules.pop("main", None)
    try:
        import fastapi  # noqa: F401
        real = True
    except ImportError:
        real = False
    saved = None if real else _install_stand_ins()
    try:
        return importlib.import_module("main"), real
    finally:
        if saved is not None:
            _restore(saved)


def _routes(main):
    routes = []
    for r in getattr(main.app, "routes", []) or []:           # real FastAPI route table
        for m in sorted(getattr(r, "methods", set()) - {"HEAD", "OPTIONS"}):
            routes.append((m, r.path, r.endpoint.__name__, None))
    if routes:
        return routes
    for router in main.app.included:                           # stand-in route table, in include order
        routes.extend(router.routes)
    return routes


def _dispatch(routes, method, path):
    for m, p, name, _ in routes:
        if m == method and re.match("^" + re.sub(r"\{[^}/]+\}", "[^/]+", p) + "$", path):
            return name
    return None


def test_main_imports_cleanly():
    main, real = _load_main()
    assert hasattr(main, "app")
    for name in ("IntakeBulkResultResponse", "IntakeBulkRegisterRequest", "IntakeBulkApproveRequest", "IntakeBulkItem"):
        assert hasattr(main, name), name
    # pathlib.Path must not be shadowed (static_dir = Path(__file__).parent / "static")
    import pathlib
    assert main.Path is pathlib.Path, "fastapi.Path shadows pathlib.Path"
    print(f"PASS: main.py imports ({'REAL fastapi' if real else 'stand-in fastapi/pydantic -- runs main.py top level only'})")


def test_recorded_route_table_dispatches_bulk_before_dynamic():
    main, real = _load_main()
    routes = _routes(main)
    assert routes, "no routes recorded"
    P = "/admin/intake"
    assert _dispatch(routes, "POST", f"{P}/bulk/approve") == "intake_bulk_approve"
    assert _dispatch(routes, "POST", f"{P}/bulk") == "intake_bulk_register"
    assert _dispatch(routes, "POST", f"{P}/8f1c2d3e-0000-4000-8000-000000000001/approve") == "intake_approve"
    # every route declared with a response_model had it defined at that point (no NameError) -> import succeeded
    print(f"PASS: route table from the imported module: bulk/approve -> intake_bulk_approve; "
          f"{{id}}/approve -> intake_approve ({len(routes)} routes)")


_TESTS = [test_main_imports_cleanly, test_recorded_route_table_dispatches_bulk_before_dynamic]

if __name__ == "__main__":
    failures = 0
    for t in _TESTS:
        try:
            t()
        except Exception as e:
            import traceback; traceback.print_exc(limit=4)
            failures += 1
            print(f"FAIL: {t.__name__}: {e!r}")
    print(f"\n{len(_TESTS) - failures}/{len(_TESTS)} passed")
    sys.exit(1 if failures else 0)
