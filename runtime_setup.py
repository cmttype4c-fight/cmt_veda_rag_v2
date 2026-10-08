"""
runtime_setup.py
----------------
Single place that builds the DB / embedder / vector store from config.py,
and validates the production environment BEFORE anything tries to connect.

Why this exists (deployment fix, post-Round-4):
  * worker.py's RQ job body needs the same db/embedder/vector_store
    objects the API builds in main.py's lifespan. It previously imported
    `build_db`/`build_embedder`/`build_vector_store` from main.py, which
    did not exist -- the first real RQ job would have crashed. Both
    processes now import these from here.
  * A deployment that supplies an unresolved placeholder (for example the
    literal string YOUR_EXISTING_RAG_POSTGRES_DSN) used to fail deep
    inside psycopg2 with "invalid dsn: missing '=' after ...". Now it
    fails immediately with a message that names the variable. The VALUE
    of a variable is never printed, because DSNs and API keys are
    secrets.

Nothing here changes the API contract, the lifecycle, or the choice of
backends (PostgreSQL, Redis/RQ, FAISS).
"""

import logging
import os
import re

import config

logger = logging.getLogger("cmt-veda-ai")

# Variables whose values must never be a template placeholder.
_PLACEHOLDER_CHECKED = (
    "RAG_API_KEY", "RAG_ADMIN_API_KEY", "RAG_POSTGRES_DSN", "RAG_REDIS_URL",
    "RAG_GGUF_MODEL_PATH", "RAG_VECTOR_STORE_PATH",
)

# Unambiguous template forms only: YOUR_SOMETHING, CHANGE_ME, REPLACE_ME,
# a value that is entirely <something>, or an uninterpolated ${VAR}. Words
# like TODO are deliberately NOT matched: they can legitimately appear
# inside a real password.
_PLACEHOLDER_RE = re.compile(
    r"(^|[^A-Za-z0-9])(YOUR_[A-Z0-9_]+|CHANGE_?ME|REPLACE_?ME)([^A-Za-z0-9]|$)"
    r"|^<[^>]*>$|\$\{[^}]*\}",
    re.IGNORECASE,
)


class ConfigError(RuntimeError):
    """Raised at startup for missing/invalid configuration. The message
    names variables, never their values."""


def _looks_like_placeholder(value: str) -> bool:
    return bool(_PLACEHOLDER_RE.search(value or ""))


def _dsn_is_wellformed(dsn: str) -> bool:
    """libpq accepts either a URI (postgresql://... / postgres://...) or a
    space-separated key=value string. Anything else is rejected with the
    same error psycopg2 would give, but earlier and without echoing it."""
    dsn = (dsn or "").strip()
    if not dsn:
        return False
    if dsn.startswith(("postgresql://", "postgres://")):
        return True
    return bool(re.search(r"(^|\s)[A-Za-z_]+\s*=\s*\S", dsn))


def validate_production_config(env=None) -> list[str]:
    """Returns a list of non-fatal warnings; raises ConfigError listing
    every fatal problem at once (so an operator fixes them in one pass
    instead of one redeploy per variable).

    Strict checks apply when RAG_DB_BACKEND=postgres, i.e. a production
    deployment. The sqlite/mock development defaults are left alone so the
    test-suite and local runs are unaffected.
    """
    env = os.environ if env is None else env
    problems: list[str] = []
    warnings: list[str] = []

    for name in _PLACEHOLDER_CHECKED:
        val = env.get(name, "")
        if val and _looks_like_placeholder(val):
            problems.append(
                f"{name} contains an unresolved placeholder value. Supply the "
                f"real value through the deployment env file (see "
                f"deploy/README.md); do not leave template text in place."
            )

    if not config.RAG_API_KEY:
        problems.append("RAG_API_KEY is not set.")

    production = config.RAG_DB_BACKEND == "postgres"
    if production:
        if not config.RAG_ADMIN_API_KEY:
            problems.append(
                "RAG_ADMIN_API_KEY is not set. (The only admin-key variable "
                "this application reads is RAG_ADMIN_API_KEY.)"
            )
        if not _dsn_is_wellformed(config.RAG_POSTGRES_DSN):
            problems.append(
                "RAG_POSTGRES_DSN is missing or is not a valid PostgreSQL "
                "connection string (expected a postgresql:// URI or "
                "'host=... dbname=... user=... password=...')."
            )
        if config.RAG_QUEUE_BACKEND == "redis_rq" and not config.REDIS_URL:
            problems.append("RAG_REDIS_URL is not set but RAG_QUEUE_BACKEND=redis_rq.")
        if config.RAG_QUEUE_BACKEND == "inline":
            warnings.append(
                "RAG_QUEUE_BACKEND=inline in a postgres deployment: intake "
                "jobs will run inside API requests. Production should use "
                "redis_rq."
            )
        if config.RAG_GENERATION_BACKEND == "llama_cpp" and not os.path.isfile(config.GGUF_MODEL_PATH):
            problems.append(
                "RAG_GENERATION_BACKEND=llama_cpp but the file named by "
                "RAG_GGUF_MODEL_PATH does not exist inside the container "
                "(check the models volume mount and the file name)."
            )
        if config.RAG_VECTOR_BACKEND == "numpy":
            warnings.append("RAG_VECTOR_BACKEND=numpy in a postgres deployment; production uses faiss.")

    # Misnamed admin-key variable seen on a previous live container.
    if env.get("RAG_ADMIN_Y") and not config.RAG_ADMIN_API_KEY:
        problems.append(
            "RAG_ADMIN_Y is set but is NOT read by this application. Its "
            "value must be supplied as RAG_ADMIN_API_KEY."
        )
    elif env.get("RAG_ADMIN_Y"):
        warnings.append("RAG_ADMIN_Y is set but ignored; remove it from the env file.")

    if problems:
        raise ConfigError(
            "Invalid RAG v2 configuration:\n  - " + "\n  - ".join(problems)
        )
    for w in warnings:
        logger.warning("CONFIG WARNING: %s", w)
    return warnings


def build_db():
    from db import get_backend
    return get_backend(
        config.RAG_DB_BACKEND, sqlite_path=config.RAG_SQLITE_PATH,
        postgres_dsn=config.RAG_POSTGRES_DSN,
    )


def build_embedder():
    from embeddings import get_embedding_backend
    return get_embedding_backend(config.RAG_EMBEDDING_BACKEND)


def build_vector_store(embedder=None):
    from vector_store import get_vector_store
    embedder = embedder or build_embedder()
    return get_vector_store(config.RAG_VECTOR_BACKEND, config.RAG_VECTOR_STORE_PATH, embedder.dim)
