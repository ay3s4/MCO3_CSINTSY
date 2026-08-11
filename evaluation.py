"""
Evaluation script for the CatBot project. Trains the bot on each cat
scenario, then runs it greedily (no exploration) for a number of rollouts
per cat to measure how many steps it takes to catch each cat.

This does NOT modify training.py, cat_env.py, utility.py, play.py, or
bot.py -- it only imports from them.

Usage:
    python evaluation.py
    python evaluation.py --cats batmeow mittens peekaboo
    python evaluation.py --rollouts 50
    python evaluation.py --csv results.csv
"""

import argparse
import time
from typing import Dict, List, Tuple

import numpy as np

from training import train_bot
from cat_env import make_env

CATS = ["batmeow", "mittens", "paotsin", "peekaboo", "squiddyboi","trainer"]
DEFAULT_ROLLOUTS = 30
MAX_STEPS = 60  # matches the maximum steps in the project spec


def greedy_rollout(env, q_table: Dict[int, np.ndarray], max_steps: int) -> Tuple[int, bool]:
    """Play one episode using the greedy (argmax) policy, no exploration.

    Returns (steps_taken, caught). If the cat isn't caught within
    max_steps, returns (max_steps, False).
    """
    state, _ = env.reset()
    for step in range(1, max_steps + 1):
        action = int(np.argmax(q_table[state]))
        state, _, terminated, truncated, _ = env.step(action)
        if terminated:
            return step, True
        if truncated:
            return step, False
    return max_steps, False


def evaluate_cat(cat_name: str, rollouts: int, max_steps: int) -> dict:
    """Train the bot on one cat, then run it greedily `rollouts` times."""
    print(f"[{cat_name}] training...")
    t0 = time.time()
    q_table = train_bot(cat_name)
    train_time = time.time() - t0
    print(f"[{cat_name}] trained in {train_time:.2f}s, running {rollouts} greedy rollouts...")

    env = make_env(cat_type=cat_name)
    all_steps: List[int] = []
    caught_steps: List[int] = []
    catches = 0

    for _ in range(rollouts):
        steps, caught = greedy_rollout(env, q_table, max_steps)
        all_steps.append(steps)
        if caught:
            catches += 1
            caught_steps.append(steps)

    env.close()

    result = {
        "cat": cat_name,
        "train_time_s": train_time,
        "rollouts": rollouts,
        "catches": catches,
        "success_rate": catches / rollouts,
        "avg_steps_when_caught": float(np.mean(caught_steps)) if caught_steps else float("nan"),
        "std_steps_when_caught": float(np.std(caught_steps)) if caught_steps else float("nan"),
        "min_steps_when_caught": int(np.min(caught_steps)) if caught_steps else None,
        "max_steps_when_caught": int(np.max(caught_steps)) if caught_steps else None,
    }
    return result


def print_table(results: List[dict]) -> None:
    header = f"{'Cat':<12}{'Success':>10}{'Avg Steps':>12}{'Std Dev':>10}{'Min':>6}{'Max':>6}{'Train(s)':>10}"
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        success_pct = f"{r['success_rate'] * 100:.0f}%"
        avg = f"{r['avg_steps_when_caught']:.1f}" if r["catches"] else "N/A"
        std = f"{r['std_steps_when_caught']:.1f}" if r["catches"] else "N/A"
        mn = str(r["min_steps_when_caught"]) if r["catches"] else "N/A"
        mx = str(r["max_steps_when_caught"]) if r["catches"] else "N/A"
        print(f"{r['cat']:<12}{success_pct:>10}{avg:>12}{std:>10}{mn:>6}{mx:>6}{r['train_time_s']:>10.2f}")
    print()


def save_csv(results: List[dict], path: str) -> None:
    fieldnames = [
        "cat", "rollouts", "catches", "success_rate",
        "avg_steps_when_caught", "std_steps_when_caught",
        "min_steps_when_caught", "max_steps_when_caught", "train_time_s",
    ]
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"Saved results to {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CatBot on each cat scenario.")
    parser.add_argument("--cats", nargs="+", default=CATS,
                         help=f"Cat types to evaluate (default: {CATS})")
    parser.add_argument("--rollouts", type=int, default=DEFAULT_ROLLOUTS,
                         help=f"Greedy rollouts per cat (default: {DEFAULT_ROLLOUTS})")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS,
                         help=f"Max steps per rollout before counting as uncaught (default: {MAX_STEPS})")
    parser.add_argument("--csv", type=str, default=None,
                         help="Optional path to save results as CSV")
    args = parser.parse_args()

    results = []
    for cat in args.cats:
        results.append(evaluate_cat(cat, args.rollouts, args.max_steps))

    print_table(results)

    if args.csv:
        save_csv(results, args.csv)


if __name__ == "__main__":
    main()