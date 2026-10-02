#!/usr/bin/env python3
"""Cascade router service: Jeff + Laya + stacking meta-model; low-confidence escalates to local 27B.

POST /route  {"state": "<user request>"}
-> {"route", "confidence", "meta_probs", "jeff", "laya", "escalated", "latency_ms"}

Run (python 3.11, laya-mlx installed):
    /Users/ting/.pyenv/versions/3.11.2/bin/python cascade_router.py
Requires: jeff-serve on :8766 (MLX backend), ollama on :11434 (for escalation).
"""
import json
import os
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from laya_mlx import Router as LayaRouter

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["researcher", "coder", "writer", "reviewer"]
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
JEFF_URL = "http://localhost:8766/v1/systemone"
OLLAMA_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "qwen3.8:27b-mxfp8"
PORT = 8767

W = np.load(f"{HERE}/tests_local/meta_W.npy")
b = np.load(f"{HERE}/tests_local/meta_b.npy")
TAU = json.load(open(f"{HERE}/tests_local/meta_tau.json"))["tau"]

laya = LayaRouter(
    models={"typed-decisions": os.path.expanduser("~/.cache/laya/typed-decisions")},
    default="typed-decisions")

def ask_jeff(state):
    body = json.dumps({"model": "jeff-latest", "state": state,
                       "questions": {"route": QUESTION}}).encode()
    req = urllib.request.Request(JEFF_URL, data=body,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        a = json.load(resp)["answers"]["route"]
    return [a["probabilities"].get(c, 0.0) for c in CLASSES]

def ask_laya(state):
    out = laya.predict({"request": state}, {"route": QUESTION},
                       model="typed-decisions")
    a = out["answers"]["route"]
    p = a.get("probabilities") or {}
    return [p.get(c, 0.25) for c in CLASSES]

def ask_llm(state):
    user = ("Sub-agents:\n" + "\n".join(f"- {k}: {v}" for k, v in CRITERIA.items())
            + f"\n\nIncoming user request: {state}\n\n"
              "Which sub-agent should handle it first? Reply with exactly one word.")
    body = json.dumps({"model": LLM_MODEL,
                       "messages": [{"role": "system", "content":
                                     "You are a request router for an AI assistant with four "
                                     "sub-agents. Reply with exactly one word: the key of the "
                                     "chosen sub-agent."},
                                    {"role": "user", "content": user}],
                       "stream": False, "think": False,
                       "options": {"temperature": 0}}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        text = json.load(resp)["message"]["content"].strip().lower()
    for k in CLASSES:
        if k in text:
            return k
    return CLASSES[0]

def meta_predict(jp, lp):
    z = np.array(jp + lp) @ W + b
    z -= z.max()
    p = np.exp(z)
    return p / p.sum()

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/route":
            self.send_response(404); self.end_headers(); return
        req = json.loads(self.rfile.read(int(self.headers["content-length"])))
        state = req.get("state") or req.get("request") or ""
        t0 = time.perf_counter()
        jp, lp = ask_jeff(state), ask_laya(state)
        mp = meta_predict(jp, lp)
        route, conf = CLASSES[int(mp.argmax())], float(mp.max())
        escalated = bool(conf < TAU)
        if escalated:
            route = ask_llm(state)
        out = {"route": route, "confidence": round(conf, 4), "meta_probs": {
                   c: round(float(mp[i]), 4) for i, c in enumerate(CLASSES)},
               "jeff": {c: round(jp[i], 4) for i, c in enumerate(CLASSES)},
               "laya": {c: round(lp[i], 4) for i, c in enumerate(CLASSES)},
               "escalated": escalated, "latency_ms": round((time.perf_counter() - t0) * 1000)}
        payload = json.dumps(out, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(payload)
        print(f"{out['route']:<10} conf={out['confidence']:.3f} "
              f"esc={escalated} {out['latency_ms']:.0f}ms | {state[:40]}", flush=True)

    def log_message(self, *args):
        pass

if __name__ == "__main__":
    print(f"cascade router on :{PORT}  (tau={TAU:.4f}, llm={LLM_MODEL})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
