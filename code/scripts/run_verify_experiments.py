#!/usr/bin/env python3
"""Verify encoder/decoder behavioral claims with controlled perturbation experiments.

E1 word-order flip (active vs passive)  -> decoder should flip more
E2 topic-gradient sweep (explain why X)  -> decoder P(coder) should rise with tech-ness
E3 masking ablation (entity vs verb)     -> encoder hurt more by entity mask,
                                            decoder hurt more by verb mask
"""
import json, os, time, urllib.request

from laya_mlx import Router

# ---------------- shared routing question ----------------
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

# ---------------- laya ----------------
r = Router(models={"typed-decisions": os.path.expanduser("~/.cache/laya/typed-decisions")},
           default="typed-decisions")

def laya_ask(state_text):
    t0 = time.perf_counter()
    out = r.predict({"request": state_text}, {"route": QUESTION}, model="typed-decisions")
    a = out["answers"]["route"]
    return a["choice"], a.get("probabilities"), (time.perf_counter() - t0) * 1000

# ---------------- jeff ----------------
def jeff_ask(state_text):
    body = json.dumps({"model": "jeff-latest", "state": state_text,
                       "questions": {"route": QUESTION}}).encode()
    req = urllib.request.Request("http://localhost:8766/v1/systemone",
                                 data=body, headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        out = json.load(resp)
    a = out["answers"]["route"]
    return a["choice"], a["probabilities"], (time.perf_counter() - t0) * 1000

def both(state_text):
    jc, jp, jlat = jeff_ask(state_text)
    lc, lp, llat = laya_ask(state_text)
    return {"state": state_text, "jeff": {"choice": jc, "probs": jp},
            "laya": {"choice": lc, "probs": lp}}

# ---------------- E1: word order ----------------
E1 = [
    ("Check this SQL query for injection risks.", "This SQL query should be checked for injection risks.", "reviewer"),
    ("Review the authentication module.", "The authentication module should be reviewed.", "reviewer"),
    ("Summarize the quarterly sales report.", "The quarterly sales report should be summarized.", "researcher"),
    ("Debug the login timeout issue.", "The login timeout issue should be debugged.", "coder"),
    ("Critique my cover letter.", "My cover letter needs a critique.", "reviewer"),
    ("Proofread the proposal draft.", "The proposal draft needs proofreading.", "writer"),
    ("Compare PostgreSQL and MySQL for our use case.", "PostgreSQL and MySQL should be compared for our use case.", "researcher"),
    ("Rewrite this onboarding email.", "This onboarding email needs to be rewritten.", "writer"),
    ("Audit the Terraform configuration.", "The Terraform configuration needs an audit.", "reviewer"),
    ("Refactor the payment service.", "The payment service needs a refactor.", "coder"),
]

# ---------------- E2: topic gradient ----------------
TOPICS = [
    ("我的吉他弦总是很快生锈", 0), ("我的面包一再发酵失败", 0),
    ("我的 Excel 求和结果不对", 1), ("我的打印机连不上 Wi-Fi", 1),
    ("我的网站首页打开很慢", 2), ("我的 Docker 容器启动后立刻退出", 2),
    ("我的模型训练 loss 周期性 spike", 3), ("我的 PostgreSQL 出现慢查询", 3),
]
FRAMES = [("explain", "用户请求：解释一下为什么{x}。", "researcher"),
          ("write", "用户请求：给初学者写一篇关于{x}的一页纸介绍。", "writer")]

# ---------------- E3: masking ----------------
E3 = [
    ("Check this SQL query for injection risks.", "Check this ___ for injection risks.", "___ this SQL query for injection risks.", "reviewer"),
    ("Audit the Terraform configuration.", "Audit the ___ configuration.", "___ the Terraform configuration.", "reviewer"),
    ("Critique my cover letter.", "Critique my ___.", "___ my cover letter.", "reviewer"),
    ("Summarize the Q3 earnings report.", "Summarize the ___ report.", "___ the Q3 earnings report.", "researcher"),
    ("Compare Kafka and RabbitMQ for our workload.", "Compare ___ and ___ for our workload.", "___ Kafka and RabbitMQ for our workload.", "researcher"),
    ("Proofread the contract draft.", "Proofread the ___ draft.", "___ the contract draft.", "writer"),
    ("Evaluate this deployment pipeline.", "Evaluate this ___.", "___ this deployment pipeline.", "reviewer"),
    ("Inspect the nginx access logs.", "Inspect the ___ logs.", "___ the nginx access logs.", "reviewer"),
]

results = {"E1": [], "E2": [], "E3": []}

print("=== E1 word order ===")
j_flips = l_flips = 0
for act, pas, gold in E1:
    ra, rp = both(act), both(pas)
    jf = ra["jeff"]["choice"] != rp["jeff"]["choice"]
    lf = ra["laya"]["choice"] != rp["laya"]["choice"]
    j_flips += jf; l_flips += lf
    results["E1"].append({"gold": gold, "active": ra, "passive": rp})
    print(f"[{gold:<10}] Jeff {'FLIP' if jf else 'same'} {ra['jeff']['choice']:<10}->{rp['jeff']['choice']:<10} | "
          f"Laya {'FLIP' if lf else 'same'} {ra['laya']['choice']:<10}->{rp['laya']['choice']:<10}")
print(f"flip rate: Jeff {j_flips}/10  Laya {l_flips}/10")

print("\n=== E2 topic gradient ===")
for frame, tmpl, gold in FRAMES:
    print(f"-- frame={frame} (gold={gold}) --")
    for topic, lvl in TOPICS:
        res = both(tmpl.format(x=topic))
        jp = res["jeff"]["probs"]; lp = res["laya"]["probs"]
        results["E2"].append({"frame": frame, "topic": topic, "level": lvl, **res})
        print(f"lvl{lvl} {topic[:22]:<24} Jeff coder={jp['coder']:.3f} researcher={jp['researcher']:.3f} | "
              f"Laya coder={lp['coder']:.3f} researcher={lp['researcher']:.3f}")

print("\n=== E3 masking ===")
for name, var in (("orig", 0), ("entity_masked", 1), ("verb_masked", 2)):
    pass
stats = {}
for orig, ent, verb, gold in E3:
    for label, text in (("orig", orig), ("entity", ent), ("verb", verb)):
        res = both(text)
        for m in ("jeff", "laya"):
            k = (m, label)
            stats.setdefault(k, []).append(res[m]["choice"] == gold)
        results["E3"].append({"gold": gold, "variant": label, **res})
for m in ("jeff", "laya"):
    line = f"{m:<5}"
    for label in ("orig", "entity", "verb"):
        v = stats[(m, label)]
        line += f"  {label}: {sum(v)}/{len(v)}"
    print(line + f"   (entity drop {sum(stats[(m,'orig')])-sum(stats[(m,'entity')])}, verb drop {sum(stats[(m,'orig')])-sum(stats[(m,'verb')])})")

json.dump(results, open("tests_local/exp_results.json", "w"), ensure_ascii=False, indent=1)
print("\nsaved tests_local/exp_results.json")
