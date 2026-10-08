"""Pre-build guard for the Docker build context (run BEFORE `docker compose build`).

Emulates Docker's .dockerignore semantics (patterns are path.Clean()ed, so a
trailing '/' is dropped; '**/' matches at any depth) and fails if:
  * any source file the image needs would be excluded (e.g. vector_store.py);
  * runtime data under ./runtime (Redis AOF, FAISS index, documents) or an env
    file would be sent to the daemon;
  * .dockerignore contains a '!' exception (forces Docker to walk excluded dirs).
Usage: python3 deploy/check_context.py [repo_root]
"""
import fnmatch
import os
import posixpath
import subprocess
import sys

root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), ".."))
SOURCE_SUFFIXES = (".py", ".sh", ".sql", ".txt", ".yml")
MUST_EXCLUDE = [
    "runtime/redis/appendonlydir/appendonly.aof.1.base.rdb",
    "runtime/redis/appendonlydir/appendonly.aof.manifest",
    "runtime/vector_store/index.faiss",
    "runtime/docs/some.pdf",
    "runtime/anything/at/any/depth.bin",
    "rag-v2.env",
    "deploy/rag-v2.env",
    ".env",
    "models/model.gguf",
]


def load_patterns():
    with open(os.path.join(root, ".dockerignore"), encoding="utf-8") as fh:
        return [l.strip() for l in fh if l.strip() and not l.strip().startswith("#")]


def _match(pattern, path):
    q = posixpath.normpath(pattern.lstrip("/"))
    parts = path.split("/")
    prefixes = ["/".join(parts[: i + 1]) for i in range(len(parts))]  # path and each parent dir
    if q.startswith("**/"):
        tail = q[3:]
        return any(fnmatch.fnmatchcase(p, tail) or fnmatch.fnmatchcase(p.split("/")[-1], tail) for p in prefixes)
    return any(fnmatch.fnmatchcase(p, q) for p in prefixes)


def excluded(path, patterns):
    return any(_match(p, path) for p in patterns)


def tracked_files():
    try:
        out = subprocess.check_output(["git", "-C", root, "ls-files"], text=True)
        return out.split("\n")
    except Exception:
        found = []
        for d, dirs, files in os.walk(root):
            dirs[:] = [x for x in dirs if x not in (".git", "runtime")]
            found += [os.path.relpath(os.path.join(d, f), root) for f in files]
        return found


def main():
    patterns = load_patterns()
    errors = []
    for p in patterns:
        if p.startswith("!"):
            errors.append(f".dockerignore has an exception ({p}); remove it so excluded dirs are never walked")
    for f in filter(None, tracked_files()):
        if (f.endswith(SOURCE_SUFFIXES) or f in ("Dockerfile", ".dockerignore")) and excluded(f, patterns):
            if f == ".dockerignore":
                continue
            errors.append(f"source file would be EXCLUDED from the image: {f}")
    for f in MUST_EXCLUDE:
        if not excluded(f, patterns):
            errors.append(f"path would be SENT in the build context (must be excluded): {f}")
    if errors:
        sys.exit("build-context check FAILED:\n  - " + "\n  - ".join(errors))
    print(f"build-context check ok ({len(patterns)} patterns; all source files included; runtime data and env files excluded)")


if __name__ == "__main__":
    main()
