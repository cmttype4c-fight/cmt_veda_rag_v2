"""
test_runtime_setup.py
---------------------
Deployment-fix tests for runtime_setup.validate_production_config():
the exact failure seen on the VPS (an unresolved
YOUR_EXISTING_RAG_POSTGRES_DSN placeholder) must now be rejected at
startup with a message that names the variable and does not echo its
value, and valid production-shaped configuration must pass.

Run: python3 test_runtime_setup.py   (or pytest)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import runtime_setup
from runtime_setup import ConfigError, validate_production_config

_KEYS = ["RAG_DB_BACKEND", "RAG_POSTGRES_DSN", "RAG_API_KEY", "RAG_ADMIN_API_KEY",
         "RAG_QUEUE_BACKEND", "REDIS_URL", "RAG_GENERATION_BACKEND", "RAG_VECTOR_BACKEND"]
GOOD = dict(
    RAG_DB_BACKEND="postgres",
    RAG_POSTGRES_DSN="postgresql://rag_user:" + "pw-" + "x1" + "@db.internal:5432/ragdb",
    RAG_API_KEY="k" + "-user", RAG_ADMIN_API_KEY="k" + "-admin",
    RAG_QUEUE_BACKEND="redis_rq", REDIS_URL="redis://redis:6379/0",
    RAG_GENERATION_BACKEND="mock", RAG_VECTOR_BACKEND="faiss",
)


def _run(overrides, env=None):
    saved = {k: getattr(config, k) for k in _KEYS}
    try:
        for k, v in {**GOOD, **overrides}.items():
            setattr(config, k, v)
        return validate_production_config(env if env is not None else {})
    finally:
        for k, v in saved.items():
            setattr(config, k, v)


def _expect_error(overrides, env=None, must_name=(), must_not_contain=()):
    try:
        _run(overrides, env)
    except ConfigError as e:
        msg = str(e)
        for needle in must_name:
            assert needle in msg, (needle, msg)
        for needle in must_not_contain:
            assert needle not in msg, f"secret/placeholder value leaked into message: {needle}"
        return
    raise AssertionError("expected ConfigError")


def test_vps_failure_placeholder_dsn_is_rejected_without_echoing_value():
    bad = "YOUR_EXISTING_RAG_POSTGRES_DSN"
    _expect_error({"RAG_POSTGRES_DSN": bad}, env={"RAG_POSTGRES_DSN": bad},
                  must_name=("RAG_POSTGRES_DSN",), must_not_contain=(bad,))
    print("PASS: the VPS's placeholder DSN is rejected at startup, variable named, value not echoed")


def test_malformed_dsn_without_placeholder_text_is_rejected():
    _expect_error({"RAG_POSTGRES_DSN": "not-a-dsn"}, must_name=("RAG_POSTGRES_DSN",),
                  must_not_contain=("not-a-dsn",))
    print("PASS: a malformed DSN is rejected with the same clear error")


def test_valid_uri_and_keyvalue_dsn_pass():
    assert _run({}) == []
    assert _run({"RAG_POSTGRES_DSN": "host=db user=u password=p dbname=d"}) == []
    print("PASS: postgresql:// URI and key=value DSNs are accepted")


def test_password_containing_todo_is_not_a_false_positive():
    dsn = "postgresql://u:" + "myTODO-" + "pass@db:5432/d"
    assert _run({"RAG_POSTGRES_DSN": dsn}, env={"RAG_POSTGRES_DSN": dsn}) == []
    print("PASS: real passwords containing words like TODO are not rejected")


def test_uninterpolated_variable_reference_is_rejected():
    v = "${RAG_POSTGRES_DSN}"
    _expect_error({"RAG_POSTGRES_DSN": v}, env={"RAG_POSTGRES_DSN": v}, must_name=("RAG_POSTGRES_DSN",))
    print("PASS: a literal ${VAR} left uninterpolated is rejected")


def test_missing_admin_key_and_obsolete_rag_admin_y():
    _expect_error({"RAG_ADMIN_API_KEY": ""}, must_name=("RAG_ADMIN_API_KEY",))
    _expect_error({"RAG_ADMIN_API_KEY": ""}, env={"RAG_ADMIN_Y": "something"},
                  must_name=("RAG_ADMIN_Y", "RAG_ADMIN_API_KEY"), must_not_contain=("something",))
    warnings = _run({}, env={"RAG_ADMIN_Y": "x"})
    assert any("RAG_ADMIN_Y" in w for w in warnings)
    print("PASS: missing admin key fails; RAG_ADMIN_Y is flagged (error if it is the only one, warning otherwise)")


def test_all_problems_reported_in_one_pass():
    try:
        _run({"RAG_POSTGRES_DSN": "", "RAG_ADMIN_API_KEY": "", "RAG_API_KEY": ""})
    except ConfigError as e:
        msg = str(e)
        assert "RAG_POSTGRES_DSN" in msg and "RAG_ADMIN_API_KEY" in msg and "RAG_API_KEY" in msg, msg
        print("PASS: every configuration problem is reported at once, not one redeploy at a time")
        return
    raise AssertionError("expected ConfigError")


def test_development_defaults_unaffected():
    assert _run({"RAG_DB_BACKEND": "sqlite", "RAG_POSTGRES_DSN": "", "RAG_ADMIN_API_KEY": ""}) == []
    print("PASS: sqlite/dev defaults are not subjected to production-only checks")


_TESTS = [
    test_vps_failure_placeholder_dsn_is_rejected_without_echoing_value,
    test_malformed_dsn_without_placeholder_text_is_rejected,
    test_valid_uri_and_keyvalue_dsn_pass,
    test_password_containing_todo_is_not_a_false_positive,
    test_uninterpolated_variable_reference_is_rejected,
    test_missing_admin_key_and_obsolete_rag_admin_y,
    test_all_problems_reported_in_one_pass,
    test_development_defaults_unaffected,
]

if __name__ == "__main__":
    failures = 0
    for t in _TESTS:
        try:
            t()
        except Exception as e:
            failures += 1
            print(f"FAIL: {t.__name__}: {e!r}")
    print(f"\n{len(_TESTS) - failures}/{len(_TESTS)} passed")
    sys.exit(1 if failures else 0)
