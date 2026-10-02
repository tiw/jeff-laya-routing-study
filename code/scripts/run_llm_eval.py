#!/usr/bin/env python3
"""Route the dataset with a local LLM (ollama qwen3.8:27b) as the strong-router baseline."""
import json, time, urllib.request

URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.8:27b-mxfp8"
OPTIONS = {"temperature": 0, "think": False}

SYSTEM = ("You are a request router for an AI assistant with four sub-agents. "
          "Reply with exactly one word: the key of the chosen sub-agent.")
CRITERIA = """- researcher: gathers information, searches, summarizes, or investigates questions
- coder: writes, edits, debugs code or scripts, or implements features
- writer: drafts, polishes, translates, or formats prose and documents
- reviewer: critiques, audits, or checks existing work and gives feedback"""

def ask(state):
    user = (f"Sub-agents:\n{CRITERIA}\n\nIncoming user request: {state}\n\n"
            "Which sub-agent should handle it first? Reply with exactly one word.")
    body = json.dumps({"model": MODEL,
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": user}],
                       "stream": False, "think": False,
                       "options": {"temperature": 0}}).encode()
    req = urllib.request.Request(URL, data=body, headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    dt = (time.perf_counter() - t0) * 1000
    text = out["message"]["content"].strip().lower()
    choice = next((k for k in ("researcher", "coder", "writer", "reviewer")
                   if k in text), "PARSE_FAIL:" + text[:30])
    return {"choice": choice, "raw": text[:40], "latency_ms": dt,
            "prompt_tokens": out.get("prompt_eval_count"),
            "output_tokens": out.get("eval_count")}

results = []
for line in open("tests_local/routing_dataset.jsonl"):
    case = json.loads(line)
    r = ask(case["state"])
    r.update({"id": case["id"], "lang": case["lang"], "gold": case["gold"]})
    r["correct"] = r["choice"] == case["gold"]
    results.append(r)
    print(f'#{case["id"]:>2} [{case["lang"]:>5}] gold={case["gold"]:<10} '
          f'llm={r["choice"]:<10} {"OK " if r["correct"] else "MISS"} '
          f'{r["latency_ms"]:.0f}ms', flush=True)

json.dump(results, open("tests_local/llm_results.json", "w"), indent=1)
acc = sum(r["correct"] for r in results) / len(results)
lat = sorted(r["latency_ms"] for r in results)
tok_in = sum(r["prompt_tokens"] or 0 for r in results) / len(results)
tok_out = sum(r["output_tokens"] or 0 for r in results) / len(results)
print(f"\nLLM accuracy: {acc:.1%}  median latency: {lat[len(lat)//2]:.0f}ms")
print(f"avg tokens per decision: prompt={tok_in:.0f} output={tok_out:.0f}")
