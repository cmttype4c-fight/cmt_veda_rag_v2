#!/usr/bin/env python3
"""
benchmark.py
------------
Real benchmark harness for CMT Veda RAG v2 (completion follow-up, section
AI/35). This is meant to be copied to and run FROM your actual deployment
(the Hostinger KVM2 or wherever RAG v2 ends up), against a running
instance of main.py with a real Postgres backend, real FAISS/sentence-
transformers, and the real Qwen2.5 GGUF model loaded.

**This script has NOT been run in the sandbox that built it.** There is
no live server, no GGUF model, and no representative VPS available there.
Every number in DELIVERABLES.md's performance section is explicitly
labeled NOT IMPLEMENTED / NOT MEASURED for exactly this reason — do not
treat this script's existence as evidence performance has been validated.

Usage:
    python3 benchmark.py --base-url http://localhost:8000 \
        --api-key $RAG_API_KEY --questions questions.txt

`questions.txt` — one question per line; at least 10 recommended for the
warm-request percentile measurement (section AI asks for "at least 10
sequential queries").

Requires: `requests` (pip install requests). `psutil` is optional — if
present, CPU/RAM are sampled during each phase; if absent, the script
still runs and just tells you to check `top`/`htop` manually alongside it,
per section AI's requirement to report CPU/RAM during concurrency tests.
"""

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    raise SystemExit("This script requires `requests`: pip install requests")

try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False


DEFAULT_QUESTIONS = [
    "What is Charcot-Marie-Tooth disease?",
    "What causes CMT?",
    "What are common symptoms of CMT?",
    "What is CMT4C?",
    "What is SH3TC2?",
    "What is p.Arg1109X?",
    "What role does HDAC6 play in CMT?",
    "How is CMT diagnosed?",
    "What treatments are supported by evidence for CMT?",
    "What is the long-term efficacy of an unstudied compound in CMT?",
]


def ask(base_url: str, api_key: str, question: str, timeout: float = 120.0) -> dict:
    t0 = time.perf_counter()
    resp = requests.post(
        f"{base_url}/api/ask",
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        json={"question": question, "user_type": "patient", "answer_length": "detailed", "table_format": "auto"},
        timeout=timeout,
    )
    elapsed = time.perf_counter() - t0
    ok = resp.status_code == 200
    return {"question": question, "elapsed_s": elapsed, "status": resp.status_code, "ok": ok}


def wait_for_health(base_url: str, timeout_s: float = 300.0) -> float:
    """Section AI 'cold start': time from process start until the health
    endpoint reports ready. This function measures from WHEN IT'S CALLED,
    not from process launch — start it immediately after you start/restart
    the service for the number to mean what section AI asks for."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        try:
            r = requests.get(f"{base_url}/api/health", timeout=5)
            if r.status_code == 200 and r.json().get("ready"):
                return time.perf_counter() - t0
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"Service did not become ready within {timeout_s}s")


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def summarize(elapsed: list[float]) -> dict:
    return {
        "n": len(elapsed),
        "min": min(elapsed) if elapsed else None,
        "avg": statistics.fmean(elapsed) if elapsed else None,
        "median": statistics.median(elapsed) if elapsed else None,
        "p95": percentile(elapsed, 0.95) if elapsed else None,
        "max": max(elapsed) if elapsed else None,
    }


def run_warm_sequential(base_url, api_key, questions, n=10):
    print(f"\n--- Warm sequential ({n} queries) ---")
    results = []
    qs = (questions * ((n // len(questions)) + 1))[:n]
    for q in qs:
        r = ask(base_url, api_key, q)
        print(f"  {r['elapsed_s']:.2f}s  status={r['status']}  {q[:60]}")
        results.append(r)
    elapsed = [r["elapsed_s"] for r in results if r["ok"]]
    failures = [r for r in results if not r["ok"]]
    summary = summarize(elapsed)
    print("Summary:", json.dumps(summary, indent=2))
    if failures:
        print(f"FAILURES: {len(failures)}/{len(results)}")
    return results


def run_concurrency(base_url, api_key, questions, concurrency: int, requests_per_worker: int = 3,
                     sequential_baseline_avg_s: float = None):
    print(f"\n--- Concurrency: {concurrency} simultaneous users ---")
    proc = psutil.Process() if HAVE_PSUTIL else None
    system_cpu_before = psutil.cpu_percent(interval=None) if HAVE_PSUTIL else None

    tasks = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        t_start = time.perf_counter()
        for i in range(concurrency):
            for j in range(requests_per_worker):
                q = questions[(i + j) % len(questions)]
                tasks.append(pool.submit(ask, base_url, api_key, q))
        results = []
        for fut in as_completed(tasks):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({"question": "?", "elapsed_s": None, "status": None, "ok": False, "error": str(e)})
        wall_time = time.perf_counter() - t_start

    if HAVE_PSUTIL:
        system_cpu_after = psutil.cpu_percent(interval=1.0)
        mem = psutil.virtual_memory()
        print(f"  System CPU% during/after burst: {system_cpu_after}")
        print(f"  System RAM used: {mem.percent}% ({mem.used / 1e9:.2f} GB / {mem.total / 1e9:.2f} GB)")
    else:
        print("  [psutil not installed — check `top`/`htop` on the server during this phase manually]")

    elapsed = [r["elapsed_s"] for r in results if r.get("ok") and r["elapsed_s"] is not None]
    failures = [r for r in results if not r.get("ok")]
    summary = summarize(elapsed)
    summary["wall_time_s"] = wall_time
    summary["failures"] = len(failures)
    summary["total_requests"] = len(results)

    # Explicit queueing-delay estimate (item 10 asks for "queueing" as its
    # own reported metric, not just folded into latency numbers). Under
    # _gen_lock serialization, a request's wall-clock time under
    # concurrency = its own generation time + time spent WAITING for the
    # lock behind other concurrent requests. We don't have a per-request
    # "time spent waiting" number from the HTTP client side (that would
    # need server-side instrumentation), so this is an estimate:
    # (avg latency under this concurrency level) - (avg latency with no
    # concurrency, from the warm sequential baseline). A near-zero value
    # means requests aren't meaningfully queueing; a value that grows with
    # concurrency is the queueing effect becoming visible.
    if sequential_baseline_avg_s is not None and summary["avg"] is not None:
        summary["estimated_queue_delay_s"] = summary["avg"] - sequential_baseline_avg_s
    else:
        summary["estimated_queue_delay_s"] = None

    print("Summary:", json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--questions", default=None, help="Path to a file with one question per line")
    ap.add_argument("--skip-cold-start", action="store_true",
                     help="Skip the cold-start wait (use if the service is already warm)")
    args = ap.parse_args()

    questions = DEFAULT_QUESTIONS
    if args.questions:
        with open(args.questions) as f:
            questions = [line.strip() for line in f if line.strip()]

    report = {"base_url": args.base_url, "n_questions": len(questions)}

    if not args.skip_cold_start:
        print("--- Cold start (waiting for /api/health to report ready) ---")
        print("Make sure you just (re)started the service before running this.")
        cold_start_s = wait_for_health(args.base_url)
        print(f"Cold start: {cold_start_s:.2f}s")
        report["cold_start_s"] = cold_start_s

    warm_results = run_warm_sequential(args.base_url, args.api_key, questions, n=max(10, len(questions)))
    warm_summary = summarize([r["elapsed_s"] for r in warm_results if r["ok"]])
    report["warm_sequential"] = warm_summary

    print("\n=== Concurrency sweep ===")
    print("NOTE: main.py serializes model generation behind a lock "
          "(_gen_lock) — concurrent requests queue for the model rather "
          "than running in parallel. Expect wall-clock time for N "
          "concurrent requests to scale roughly linearly with N, not stay "
          "flat. That is the direct, measurable effect of the lock this "
          "section explicitly asked to report. 'estimated_queue_delay_s' "
          "in each concurrency result below is (avg latency at this "
          "concurrency level) minus (avg latency with no concurrency) — "
          "a rough proxy for time spent waiting behind other requests.")
    report["concurrency"] = {}
    for c in (1, 2, 5):
        report["concurrency"][c] = run_concurrency(
            args.base_url, args.api_key, questions, concurrency=c,
            sequential_baseline_avg_s=warm_summary["avg"],
        )

    print("\n=== FULL REPORT (JSON) ===")
    print(json.dumps(report, indent=2, default=str))
    with open("benchmark_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\nWritten to benchmark_report.json")


if __name__ == "__main__":
    main()
