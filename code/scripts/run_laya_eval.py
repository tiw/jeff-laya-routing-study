#!/usr/bin/env python3
"""Route the dataset with laya-mlx typed-decisions checkpoint (local path, no HF)."""
import json, os, time

from laya_mlx import Router

CKPT = os.path.expanduser("~/.cache/laya/typed-decisions")
r = Router(models={"typed-decisions": CKPT}, default="typed-decisions")

QUESTION = {
    "type": "choice",
    "instructions": "You are routing an incoming user request in `request` to one of four sub-agents. Pick exactly one.",
    "criteria": {
        "researcher": "gathers information, searches, summarizes, or investigates questions",
        "coder": "writes, edits, debugs code or scripts, or implements features",
        "writer": "drafts, polishes, translates, or formats prose and documents",
        "reviewer": "critiques, audits, or checks existing work and gives feedback",
    },
}

results = []
cold = None
for line in open("tests_local/routing_dataset.jsonl"):
    case = json.loads(line)
    state = {"request": case["state"]}
    t0 = time.perf_counter()
    answers = r.predict(state, {"route": QUESTION}, model="typed-decisions")
    dt = (time.perf_counter() - t0) * 1000
    if cold is None:
        cold = dt
    a = answers["answers"]["route"]
    rec = {"id": case["id"], "lang": case["lang"], "gold": case["gold"],
           "choice": a["choice"], "confidence": a.get("confidence"),
           "probabilities": a.get("probabilities"), "latency_ms": dt,
           "input_tokens": a.get("input_tokens")}
    rec["correct"] = rec["choice"] == case["gold"]
    results.append(rec)
    print(f'#{case["id"]:>2} [{case["lang"]:>5}] gold={case["gold"]:<10} '
          f'laya={a["choice"]:<10} conf={a.get("confidence")} '
          f'{"OK " if rec["correct"] else "MISS"} {dt:.0f}ms', flush=True)

json.dump(results, open("tests_local/laya_results.json", "w"), indent=1)
acc = sum(x["correct"] for x in results) / len(results)
lat = sorted(x["latency_ms"] for x in results)
print(f"\nLaya accuracy: {acc:.1%}  first call: {cold:.0f}ms  median: {lat[len(lat)//2]:.0f}ms")
by_lang = {}
for x in results:
    by_lang.setdefault(x["lang"], []).append(x["correct"])
for k, v in by_lang.items():
    print(f"  {k}: {sum(v)/len(v):.1%} ({sum(v)}/{len(v)})")
