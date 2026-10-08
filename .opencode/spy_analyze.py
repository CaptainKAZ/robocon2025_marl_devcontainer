#!/usr/bin/env python3
"""分析 py-spy raw 火焰图数据：按线程分类，统计热点叶子帧。

用法: python3 spy_analyze.py /tmp/opencode/spy_raw.txt
"""
import sys
from collections import Counter

MARK_COLLECTOR = "_worker (benchmarl/experiment/experiment.py:109)"
MARK_TRAIN = "_optimizer_loop (benchmarl/experiment/experiment.py"
MARK_EVAL = "_evaluation_worker"


def classify(stack: str) -> str:
    if MARK_EVAL in stack or "eval_worker" in stack:
        return "eval_worker"
    if MARK_COLLECTOR in stack:
        return "collector(prefetch)"
    if MARK_TRAIN in stack:
        return "main(train)"
    if "test_env" in stack or "rollout" in stack:
        return "main(rollout/other)"
    return "other"


def main(path):
    per_thread = {}
    total = 0
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            # collapsed 格式：frames...;frames N
            parts = line.rsplit(" ", 1)
            if len(parts) == 2 and parts[1].isdigit():
                stack, n = parts[0], int(parts[1])
            else:
                stack, n = line, 1
            total += n
            cls = classify(stack)
            leaves = per_thread.setdefault(cls, Counter())
            leaves[stack.split(";")[-1]] += n

    print(f"总样本 {total}")
    for cls, leaves in sorted(per_thread.items(), key=lambda kv: -sum(kv[1].values())):
        s = sum(leaves.values())
        print(f"\n===== {cls}: {s} 样本 ({100*s/total:.1f}%) =====")
        for leaf, n in leaves.most_common(12):
            leaf_short = leaf if len(leaf) < 110 else leaf[:107] + "..."
            print(f"  {100*n/s:5.1f}%  {leaf_short}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/spy_raw.txt")
