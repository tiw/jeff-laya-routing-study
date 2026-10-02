#!/usr/bin/env python3
"""Run all 5 new dataset files through Jeff (HTTP) + Laya (in-process), save results."""
import json, os, urllib.request

from laya_mlx import Router

CRITERIA = {
    "researcher": "gathers information, searches, summarizes, or investigates questions",
    "coder": "writes, edits, debugs code or scripts, or implements features",
    "writer": "drafts, polishes, translates, or formats prose and documents",
    "reviewer": "critiques, audits, or checks existing work and gives feedback",
}
QUESTION = {
    "type": "choice",
    "instructions": "You are routing an incoming user request in `request` to one of four sub-agents. Pick exactly one.",
    "criteria": CRITERIA,
}

r = Router(models={"typed-decisions": os.path.expanduser("~/.cache/laya/typed-decisions")},
           default="typed-decisions")

def laya_ask(state_text):
    out = r.predict({"request": state_text}, {"route": QUESTION}, model="typed-decisions")
    a = out["answers"]["route"]
    return a["choice"], a.get("probabilities")

def jeff_ask(state_text):
    body = json.dumps({"model": "jeff-latest", "state": state_text,
                       "questions": {"route": QUESTION}}).encode()
    req = urllib.request.Request("http://localhost:8766/v1/systemone",
                                 data=body, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        out = json.load(resp)
    a = out["answers"]["route"]
    return a["choice"], a["probabilities"], a["confidence"]

SETS = ["set_a_ood", "set_b_reviewer", "set_c_diagnosis", "set_d_multi_intent", "set_e_vague"]
for name in SETS:
    results = []
    for line in open(f"tests_local/{name}.jsonl"):
        case = json.loads(line)
        jc, jp, jconf = jeff_ask(case["state"])
        lc, lp = laya_ask(case["state"])
        rec = dict(case)
        rec.update(jeff_choice=jc, jeff_probs=jp, jeff_conf=jconf,
                   laya_choice=lc, laya_probs=lp,
                   jeff_correct=jc == case["gold"], laya_correct=lc == case["gold"])
        results.append(rec)
    json.dump(results, open(f"tests_local/{name}_results.json", "w"), ensure_ascii=False, indent=1)
    n = len(results)
    print(f"{name:<22} n={n:<4} jeff {sum(x['jeff_correct'] for x in results)/n:.1%}   "
          f"laya {sum(x['laya_correct'] for x in results)/n:.1%}", flush=True)
print("all done")
