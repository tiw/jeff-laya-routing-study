#!/usr/bin/env python3
"""Stacking meta-model (softmax regression, numpy) vs hand-written 0.7 cascade.

Data: 40-case routing dataset + 240 E2 formal items, all with gold labels and
both models' full probability vectors. 5-fold stratified CV.
"""
import json
import numpy as np

CLASSES = ["researcher", "coder", "writer", "reviewer"]

def load_combined():
    rows = []
    jeff = {r["id"]: r for r in json.load(open("tests_local/jeff_results.json"))}
    laya = {r["id"]: r for r in json.load(open("tests_local/laya_results.json"))}
    for i, j in jeff.items():
        l = laya[i]
        rows.append({
            "gold": j["gold"],
            "jeff": [j["probabilities"].get(c, 0.0) for c in CLASSES],
            "laya": [(l.get("probabilities") or {}).get(c, 0.25) for c in CLASSES],
            "jeff_conf": j["confidence"],
        })
    for d in json.load(open("tests_local/exp2_formal_results.json")):
        jp = [d["jeff_probs"].get(c, 0.0) for c in CLASSES]
        lp = [(d.get("laya_probs") or {}).get(c, 0.25) for c in CLASSES]
        rows.append({"gold": d["gold"], "jeff": jp, "laya": lp, "jeff_conf": max(jp)})
    return rows

def train(X, y, K, iters=3000, lr=0.5, lam=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    N, D = X.shape
    W = rng.normal(0, 0.01, (D, K))
    b = np.zeros(K)
    Y = np.eye(K)[y]
    for _ in range(iters):
        Z = X @ W + b
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(1, keepdims=True)
        W -= lr * (X.T @ (P - Y) / N + lam * W)
        b -= lr * (P - Y).mean(0)
    return W, b

def predict(X, W, b):
    Z = X @ W + b
    Z -= Z.max(1, keepdims=True)
    P = np.exp(Z)
    P /= P.sum(1, keepdims=True)
    return P

def stratified_folds(y, k=5, seed=42):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[j % k].append(i)
    return [np.array(sorted(f)) for f in folds]

rows = load_combined()
n = len(rows)
y = np.array([CLASSES.index(r["gold"]) for r in rows])
X = np.array([r["jeff"] + r["laya"] for r in rows])
print(f"n = {n}  (40-case set + 240 E2 items)")

jeff_pred = np.array([np.argmax(r["jeff"]) for r in rows])
laya_pred = np.array([np.argmax(r["laya"]) for r in rows])
avg_pred = np.array([np.argmax((np.array(r["jeff"]) + np.array(r["laya"])) / 2) for r in rows])
print(f"Jeff alone:            {(jeff_pred == y).mean():.1%}")
print(f"Laya alone:            {(laya_pred == y).mean():.1%}")
print(f"prob-average combiner: {(avg_pred == y).mean():.1%}")

esc = np.array([r["jeff_conf"] < 0.7 for r in rows])
hand_acc = ((jeff_pred == y) & ~esc).sum() / n + esc.sum() / n  # oracle on escalated
print(f"hand cascade (conf<0.7 -> oracle): acc {hand_acc:.1%}, escalate {esc.mean():.1%}")

folds = stratified_folds(y, 5)
budget = esc.mean()
meta_pure = meta_casc = n_meta_esc = 0
for te in folds:
    tr = np.setdiff1d(np.arange(n), te)
    W, b = train(X[tr], y[tr], len(CLASSES))
    P = predict(X[te], W, b)
    meta_pure += (P.argmax(1) == y[te]).sum()
    # matched-budget abstention: tau tuned on the training fold
    maxp_tr = predict(X[tr], W, b).max(1)
    k = int(round(budget * len(tr)))
    tau = np.sort(maxp_tr)[k - 1] if k > 0 else 0.0
    esc_te = P.max(1) < tau
    meta_casc += ((P.argmax(1) == y[te]) & ~esc_te).sum() + esc_te.sum()
    n_meta_esc += esc_te.sum()

print(f"meta alone (5-fold CV):                  {meta_pure / n:.1%}")
print(f"meta cascade matched-budget (5-fold CV): {meta_casc / n:.1%}, escalate {n_meta_esc / n:.1%}")

W, b = train(X, y, len(CLASSES))
names = [f"jeff_{c}" for c in CLASSES] + [f"laya_{c}" for c in CLASSES]
print("\nmeta-model top coefficients per class (fit on all data):")
for k_, cls in enumerate(CLASSES):
    top = sorted(zip(names, W[:, k_]), key=lambda t: -abs(t[1]))[:3]
    print(f"  {cls:<10} " + ", ".join(f"{nm}={w:+.2f}" for nm, w in top))
np.save("tests_local/meta_W.npy", W)
np.save("tests_local/meta_b.npy", b)
