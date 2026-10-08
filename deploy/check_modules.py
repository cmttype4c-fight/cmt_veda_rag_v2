"""Build-time guard: every first-party module imported (transitively) by the
entry points must exist next to them. Fails the image build with a clear list
instead of letting the container crash-loop with ModuleNotFoundError.
Usage: python deploy/check_modules.py [app_dir]"""
import ast, os, sys

app = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), ".."))
local = {f[:-3] for f in os.listdir(app) if f.endswith(".py")}
# names of first-party modules known to the repo (from git/source), supplied via the source tree itself
ENTRY = ["main", "worker", "runtime_setup"]

def imports(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names: yield a.name.split(".")[0]
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            yield n.module.split(".")[0]

# The set of first-party module names = every .py file that SHOULD exist. We
# derive "should exist" from the entry points' own imports, resolved against
# stdlib/third-party by exclusion: a name is first-party if a file of that name
# is present OR it is listed in MANIFEST below (names the code imports locally).
MANIFEST = {"config","db","embeddings","vector_store","ingestion","ingestion_pipeline","retrieval",
            "generation","conversation","entities","discovery","intake","worker","runtime_setup",
            "url_safety","diagnose_retrieval","migrate_corpus"}
missing, seen, todo = set(), set(), list(ENTRY)
while todo:
    m = todo.pop()
    if m in seen: continue
    seen.add(m)
    p = os.path.join(app, m + ".py")
    if not os.path.isfile(p):
        missing.add(m); continue
    for dep in imports(p):
        if dep in MANIFEST or dep in local:
            todo.append(dep)
if missing:
    sys.exit("MISSING first-party modules in image: " + ", ".join(sorted(missing)))
print("module check ok:", ", ".join(sorted(seen)))
