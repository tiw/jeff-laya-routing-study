#!/usr/bin/env python3
"""Batch-run the routing dataset against local jeff-serve and record results."""
import json, time, urllib.request

URL = "http://localhost:8766/v1/systemone"
QUESTION = {
    "type": "choice",
    "instructions": "You are routing an incoming user request to one of four sub-agents. Pick exactly one.",
    "criteria": {
        "researcher": "gathers information, searches, summarizes, or investigates questions",
        "coder": "writes, edits, debugs code or scripts, or implements features",
        "writer": "drafts, polishes, translates, or formats prose and documents",
        "reviewer": "critiques, audits, or checks existing work and gives feedback",
    },
}

def ask(state):
    body = json.dumps({"model": "jeff-latest", "state": state,
                       "questions": {"route": QUESTION}}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.load(r)
    dt = (time.perf_counter() - t0) * 1000
    a = out["answers"]["route"]
    return {"choice": a["choice"], "confidence": a["confidence"],
            "probabilities": a["probabilities"],
            "input_tokens": out["usage"]["input_tokens"], "latency_ms": dt}

results = []
for line in open("tests_local/routing_dataset.jsonl"):
    case = json.loads(line)
    r = ask(case["state"])
    r.update({"id": case["id"], "lang": case["lang"], "gold": case["gold"]})
    r["correct"] = r["choice"] == case["gold"]
    results.append(r)
    print(f'#{case["id"]:>2} [{case["lang"]:>5}] gold={case["gold"]:<10} '
          f'jeff={r["choice"]:<10} conf={r["confidence"]:.3f} '
          f'{"OK " if r["correct"] else "MISS"} {r["latency_ms"]:.0f}ms', flush=True)

json.dump(results, open("tests_local/jeff_results.json", "w"), indent=1)
acc = sum(r["correct"] for r in results) / len(results)
lat = sorted(r["latency_ms"] for r in results)
print(f"\nJeff accuracy: {acc:.1%}  median latency: {lat[len(lat)//2]:.0f}ms")
by_lang = {}
for r in results:
    by_lang.setdefault(r["lang"], []).append(r["correct"])
for k, v in by_lang.items():
    print(f"  {k}: {sum(v)/len(v):.1%} ({sum(v)}/{len(v)})")
