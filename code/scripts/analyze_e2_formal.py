#!/usr/bin/env python3
"""Analyze formal E2 and render the P(coder) gradient chart."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = json.load(open("tests_local/exp2_formal_results.json"))

frames = ["explain", "fix", "write"]
titles = {"explain": 'Frame: "Explain why X" (gold: researcher)',
          "fix": 'Frame: "Troubleshoot and fix X" (gold: coder)',
          "write": 'Frame: "Write a beginner intro about X" (gold: writer)'}
levels = [0, 1, 2, 3]
level_labels = ["non-tech\nlife", "consumer/\noffice tech", "web/dev", "ML/infra"]

def agg(frame, model_key):
    means, stds, accs = [], [], []
    for lv in levels:
        rows = [d for d in data if d["frame"] == frame and d["level"] == lv]
        ps = [r[f"{model_key}_probs"]["coder"] for r in rows]
        means.append(sum(ps) / len(ps))
        stds.append((sum((p - means[-1]) ** 2 for p in ps) / len(ps)) ** 0.5)
        accs.append(sum(r[f"{model_key}_choice"] == r["gold"] for r in rows) / len(rows))
    return means, stds, accs

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
colors = {"jeff": "#d62728", "laya": "#1f77b4"}
names = {"jeff": "Jeff-2B (decoder SFT)", "laya": "Laya-421M (encoder)"}
for ax, frame in zip(axes, frames):
    for mk in ("jeff", "laya"):
        means, stds, accs = agg(frame, mk)
        ax.plot(levels, means, "o-", color=colors[mk], label=names[mk], lw=2)
        ax.fill_between(levels, [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=colors[mk], alpha=0.15)
        for x, m, a in zip(levels, means, accs):
            ax.annotate(f"{m:.2f}", (x, m), textcoords="offset points",
                        xytext=(0, 8 if mk == "jeff" else -14),
                        ha="center", fontsize=8, color=colors[mk])
    ax.set_title(titles[frame], fontsize=10)
    ax.set_xticks(levels)
    ax.set_xticklabels(level_labels, fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
axes[0].set_ylabel("P(coder)  (mean ± std, n=20 topics)", fontsize=10)
axes[0].legend(fontsize=9, loc="upper left")
fig.suptitle("Code-topic prior under three action frames — Jeff vs Laya (240 items, M4 Pro)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("tests_local/exp2_gradient.png", dpi=160)
print("chart saved: tests_local/exp2_gradient.png")

print("\n=== mean P(coder) per level (n=20 each) ===")
for frame in frames:
    print(f"-- {frame} --")
    for mk in ("jeff", "laya"):
        means, stds, accs = agg(frame, mk)
        print(f"  {mk:<5} " + "  ".join(f"L{lv}:{m:.3f}±{s:.3f}(acc {a:.2f})"
              for lv, m, s, a in zip(levels, means, stds, accs)))
