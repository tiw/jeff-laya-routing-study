#!/usr/bin/env python3
"""Stacking meta-model OOD generalization test.

Train on the 280 in-distribution items only; test on the 260 new items
(5 realistic sets). Compare against Jeff/Laya alone, prob-average, and the
hand-written 0.7 cascade at matched escalation budget.
"""
import json
import numpy as np

CLASSES = ["researcher", "coder", "writer", "reviewer"]
SETS = ["set_a_ood", "set_b_reviewer", "set_c_diagnosis", "set_d_multi_intent", "set_e_vague"]

def probs_of(d, prefix):
    p = d.get(f"{prefix}_probs") or {}
    return [p.get(c, 0.25) for c in CLASSES]

def load_rows():
    rows = []
    jeff = {r["id"]: r for r in json.load(open("tests_local/jeff_results.json"))}
    laya = {r["id"]: r for r in json.load(open("tests_local/llm_results.json"))}  # placeholder, not used
    laya = {r["id"]: r for r in json.load(open("tests_local/laya_results.json"))}
    for i, j in jeff.items():
        l = laya[i]
        rows.append({"gold": j["gold"], "src": "indist",
                     "jeff": [j["probabilities"].get(c, 0.0) for c in CLASSES],
                     "laya": [(l.get("probabilities") or {}).get(c, 0.25) for c in CLASSES],
                     "jeff_conf": j["confidence"], "valid": True})
    for d in json.load(open("tests_local/exp2_formal_results.json")):
        jp = probs_of(d, "jeff")
        rows.append({"gold": d["gold"], "src": "indist", "jeff": jp,
                     "laya": probs_of(d, "laya"), "jeff_conf": max(jp), "valid": True})
    for name in SETS:
        for d in json.load(open(f"tests_local/{name}_results.json")):
            jp = probs_of(d, "jeff")
            rows.append({"gold": d["gold"], "src": name, "jeff": jp,
                         "laya": probs_of(d, "laya"), "jeff_conf": d["jeff_conf"],
                         "valid": not d.get("needs_context", False)})
    return rows

def train(X, y, K, iters=3000, lr=0.5, lam=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    N, D = X.shape
    W = rng.normal(0, 0.01, (D, K)); b = np.zeros(K)
    Y = np.eye(K)[y]
    for _ in range(iters):
        Z = X @ W + b; Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        W -= lr * (X.T @ (P - Y) / N + lam * W)
        b -= lr * (P - Y).mean(0)
    return W, b

def predict(X, W, b):
    Z = X @ W + b; Z -= Z.max(1, keepdims=True)
    P = np.exp(Z); P /= P.sum(1, keepdims=True)
    return P

def stratified_folds(y, k=5, seed=42):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]; rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[j % k].append(i)
    return [np.array(sorted(f)) for f in folds]

rows = load_rows()
y = np.array([CLASSES.index(r["gold"]) for r in rows])
X = np.array([r["jeff"] + r["laya"] for r in rows])
src = np.array([r["src"] for r in rows])
valid = np.array([r["valid"] for r in rows])
ind = src == "indist"
ood = ~ind
ood_valid = ood & valid

print(f"total n={len(rows)}  in-dist={ind.sum()}  ood={ood.sum()} (valid={ood_valid.sum()}, "
      f"excl. {int((ood & ~valid).sum())} placeholder)")

# ---------- baselines on OOD test ----------
jeff_pred = X[:, :4].argmax(1); laya_pred = X[:, 4:].argmax(1)
avg_pred = X.reshape(len(rows), 2, 4).mean(1).argmax(1)
print("\n=== on OOD test (valid only) ===")
print(f"Jeff alone:   {(jeff_pred[ood_valid] == y[ood_valid]).mean():.1%}")
print(f"Laya alone:   {(laya_pred[ood_valid] == y[ood_valid]).mean():.1%}")
print(f"prob-average: {(avg_pred[ood_valid] == y[ood_valid]).mean():.1%}")
esc_test = np.array([rows[i]["jeff_conf"] < 0.7 for i in range(len(rows))])[ood_valid]
yt, jt = y[ood_valid], jeff_pred[ood_valid]
hand = (((jt == yt) & ~esc_test).sum() + esc_test.sum()) / len(yt)
print(f"hand cascade 0.7: acc {hand:.1%}, escalate {esc_test.mean():.1%}")

# ---------- meta trained on in-dist only, tested on OOD ----------
W, b = train(X[ind], y[ind], 4)
P = predict(X[ood_valid], W, b)
meta_pred = P.argmax(1)
print(f"\nmeta (train=in-dist 280, test=OOD): {(meta_pred == y[ood_valid]).mean():.1%}")
for s in SETS:
    m = (src == s) & valid
    if m.sum():
        print(f"  {s:<22} {(predict(X[m], W, b).argmax(1) == y[m]).mean():.1%} (n={int(m.sum())})")

# ---------- matched-budget cascade: tau from in-dist, applied to OOD ----------
Ptr = predict(X[ind], W, b).max(1)
k_esc = int(round(0.225 * ind.sum()))
tau = np.sort(Ptr)[k_esc - 1]
json.dump({"tau": float(tau), "escalate_rate_in_dist": 0.225},
          open("tests_local/meta_tau.json", "w"))
print(f"tau saved: {tau:.4f}")
esc_meta = P.max(1) < tau
meta_casc = (((meta_pred == y[ood_valid]) & ~esc_meta).sum() + esc_meta.sum()) / len(yt)
print(f"meta cascade (tau from in-dist): acc {meta_casc:.1%}, escalate {esc_meta.mean():.1%}")

# ---------- ceiling: 5-fold CV on all 540 ----------
folds = stratified_folds(y, 5)
tot = cor = 0
for te in folds:
    tr = np.setdiff1d(np.arange(len(y)), te)
    W2, b2 = train(X[tr], y[tr], 4)
    cor += (predict(X[te], W2, b2).argmax(1) == y[te]).sum(); tot += len(te)
print(f"\nmeta 5-fold CV on all 540 (ceiling, in-dist folds): {cor / tot:.1%}")

names = [f"jeff_{c}" for c in CLASSES] + [f"laya_{c}" for c in CLASSES]
print("\nmeta (in-dist trained) top coefficients:")
for k_, cls in enumerate(CLASSES):
    top = sorted(zip(names, W[:, k_]), key=lambda t: -abs(t[1]))[:3]
    print(f"  {cls:<10} " + ", ".join(f"{nm}={w:+.2f}" for nm, w in top))
