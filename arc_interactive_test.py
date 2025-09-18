#!/usr/bin/env python
"""
ARC-AGI-v2 Interactive Tester (Solver–Evaluator–Teacher Loop)

Features:
- Relaxed schema: allows output size up to MAX_EXPAND times larger than input.
- Stronger prompt: explicitly says output can differ in size, and forbids copying input.
- Sanity checks: rejects outputs smaller than input or exact copies of input.
- Reads challenge + optional solutions JSON.
- --limit parameter controls number of puzzles (safeguarded with min(limit, len(all_ids))).
- Saves results to result/result_YYYYMMDD_HHMMSS.txt
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from openai import OpenAI
except Exception:
    print("ERROR: Install the openai package first: pip install openai>=1.30.0")
    raise

# ---------------------------
# Config
# ---------------------------

MAX_EXPAND = 5  # allow up to 5x expansion
COLOR_MIN, COLOR_MAX = 0, 9

Grid = List[List[int]]

# ---------------------------
# Helpers
# ---------------------------

def read_data(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_int_grid_flexible(g: Any, minv: int = COLOR_MIN, maxv: int = COLOR_MAX) -> Grid:
    if not isinstance(g, list) or len(g) == 0:
        raise ValueError("Grid must be non-empty list of rows")
    out: Grid = []
    width = None
    for r in g:
        if not isinstance(r, list) or len(r) == 0:
            raise ValueError("Each row must be non-empty list")
        if width is None:
            width = len(r)
        elif len(r) != width:
            raise ValueError("All rows must be same length")
        row: List[int] = []
        for v in r:
            ival = int(v)
            if ival < minv or ival > maxv:
                raise ValueError("Value out of range")
            row.append(ival)
        out.append(row)
    return out

def grid_size(g: Grid) -> Tuple[int, int]:
    return len(g), (len(g[0]) if g and isinstance(g[0], list) else 0)

def eval_grid(pred: Grid, gold: Grid) -> Dict[str, Any]:
    H1, W1 = grid_size(pred)
    H2, W2 = grid_size(gold)
    if (H1, W1) != (H2, W2):
        return {"ok": False, "reason": "size_mismatch", "accuracy": 0.0}
    total = H1 * W1
    diff = sum(pred[i][j] != gold[i][j] for i in range(H1) for j in range(W1))
    acc = 100.0 * (total - diff) / total
    return {"ok": True, "accuracy": acc, "diff": diff, "total": total}

def build_schema_relaxed(H: int, W: int, minv: int = COLOR_MIN, maxv: int = COLOR_MAX) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "grid": {
                "type": "array",
                "minItems": 1,
                "maxItems": max(1, H * MAX_EXPAND),
                "items": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": max(1, W * MAX_EXPAND),
                    "items": {"type": "integer", "minimum": minv, "maximum": maxv}
                }
            }
        },
        "required": ["grid"],
        "additionalProperties": False
    }

def build_base_prompt(train_pairs: Any, test_input: Grid) -> str:
    return (
        "You are given several training input/output grid pairs that follow a single hidden rule.\n"
        "Digits 0–9 denote colors with no numeric meaning.\n"
        "IMPORTANT:\n"
        " • The TEST output may have different size than input (e.g., scaling, tiling, framing).\n"
        " • Do NOT simply copy the input.\n"
        "Task: Apply the rule and return ONLY the output grid as JSON per schema.\n\n"
        f"TRAINING PAIRS:\n{train_pairs}\n\n"
        f"TEST INPUT:\n{test_input}"
    )

def solve_round(client: OpenAI, model: str, base_prompt: str, schema: Dict[str, Any],
                temperature: float, top_p: float, k: int):
    candidates: List[Grid] = []
    raws: List[str] = []
    for _ in range(k):
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": "You are an ARC-AGI analyst. Think silently and output only JSON."},
                {"role": "user", "content": base_prompt}
            ],
            temperature=temperature,
            top_p=top_p,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "grid_only_response",
                    "schema": schema,
                    "strict": True
                }
            }
        )
        raw = resp.output_text
        try:
            parsed = json.loads(raw)
            candidates.append(parsed["grid"])
            raws.append(raw)
        except Exception as e:
            raws.append(f"PARSE_ERROR: {e}; RAW={raw[:200]}")
    return candidates, raws

# ---------------------------
# Sanity checks & hint generator
# ---------------------------

def is_copy_of_input(pred: Grid, test_input: Grid) -> bool:
    Hp, Wp = grid_size(pred)
    Hi, Wi = grid_size(test_input)
    if (Hp, Wp) != (Hi, Wi):
        return False
    return all(pred[i][j] == test_input[i][j] for i in range(Hp) for j in range(Wp))

def too_small(pred: Grid, test_input: Grid) -> bool:
    Hp, Wp = grid_size(pred)
    Hi, Wi = grid_size(test_input)
    return Hp < Hi or Wp < Wi

def make_hint(train_pairs, test_input, pred, gold, rep):
    if pred is None or grid_size(pred) == (0, 0):
        return "Output must be a non-empty grid of integers 0–9."
    hints = []
    if too_small(pred, test_input):
        hints.append("Output should not be smaller than input. Consider tiling or scaling.")
    if is_copy_of_input(pred, test_input):
        hints.append("Do not copy the input directly. Apply transformation from training pairs.")
    if not hints:
        hints.append("Check if the rule involves repetition, geometric transforms, or color remapping.")
    return " ".join(hints)

# ---------------------------
# Main loop
# ---------------------------

def run_interactive(args: argparse.Namespace) -> None:
    challenges = read_data(args.data)
    solutions = read_data(args.solutions) if args.solutions else {}

    all_ids = list(challenges.keys())
    limit = min(args.limit, len(all_ids))
    target_ids = all_ids[:limit]

    client = OpenAI()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path("result")
    result_dir.mkdir(exist_ok=True)
    out_path = result_dir / f"result_{timestamp}.txt"

    for pid in target_ids:
        log_lines: List[str] = []
        log_lines.append(f"Puzzle ID: {pid}")
        log_lines.append(f"Model: {args.model}")
        log_lines.append("-" * 60)

        train_pairs = challenges[pid]["train"]
        test_input = challenges[pid]["test"][0]["input"]

        gold_output = None
        if pid in solutions:
            sols = solutions[pid]
            if isinstance(sols, list) and sols:
                gold_output = sols[0]

        Hi, Wi = len(test_input), len(test_input[0])
        schema = build_schema_relaxed(Hi, Wi)
        base_prompt = build_base_prompt(train_pairs, test_input)

        current_prompt = base_prompt
        best_acc = -1.0
        best_pred: Grid = None

        for r in range(1, args.rounds + 1):
            cands, raws = solve_round(client, args.model, current_prompt, schema,
                                      args.temperature, args.top_p, args.k)

            if not cands:
                log_lines.append(f"[{pid}][Round {r}] No candidates.")
                break

            scored = []
            for g in cands:
                try:
                    g_fixed = ensure_int_grid_flexible(g, COLOR_MIN, COLOR_MAX)
                except Exception as e:
                    scored.append((-1.0, None, {"ok": False, "reason": f"parse_error: {e}"}))
                    continue

                if too_small(g_fixed, test_input):
                    rep = {"ok": False, "reason": "too_small"}
                    acc = -1.0
                elif is_copy_of_input(g_fixed, test_input):
                    rep = {"ok": False, "reason": "copied_input"}
                    acc = -1.0
                else:
                    if gold_output is None:
                        rep = {"ok": True, "accuracy": None}
                        acc = -0.5
                    else:
                        rep = eval_grid(g_fixed, gold_output)
                        acc = rep.get("accuracy", 0.0) if rep.get("ok", False) else -1.0

                scored.append((acc, g_fixed, rep))

            scored.sort(key=lambda x: (x[0] is not None, x[0]), reverse=True)
            acc, pred, rep = scored[0]

            if gold_output is None:
                log_lines.append(f"[{pid}][Round {r}] Prediction accepted (no gold).")
            else:
                if rep.get("ok"):
                    log_lines.append(f"[{pid}][Round {r}] Accuracy: {acc:.2f}%")
                else:
                    log_lines.append(f"[{pid}][Round {r}] Rejected: {rep.get('reason')}")

            if gold_output and rep.get("ok") and acc >= 100.0:
                log_lines.append(f"[{pid}][Round {r}] SOLVED ✅")
                best_acc, best_pred = acc, pred
                break

            if gold_output and rep.get("ok") and acc > best_acc:
                best_acc, best_pred = acc, pred
            elif gold_output is None and rep.get("ok"):
                best_pred = pred

            hint = make_hint(train_pairs, test_input, pred, gold_output, rep)
            log_lines.append(f"[{pid}][Round {r}] HINT: {hint}")
            current_prompt = base_prompt + "\nHINT:\n" + hint

        log_lines.append("-" * 60)
        if gold_output is None:
            log_lines.append(f"[{pid}] Best prediction grid: {best_pred}")
        else:
            log_lines.append(f"[{pid}] Best accuracy: {best_acc:.2f}%")
            log_lines.append(f"[{pid}] Best prediction grid: {best_pred}")

        with out_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")

    print(f"Finished evaluation of {len(target_ids)} puzzles. Results saved to {out_path}")

# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ARC-AGI-v2 interactive tester")
    p.add_argument("--data", required=True, help="Path to arc-agi_test_challenges.json")
    p.add_argument("--solutions", required=False, help="Path to arc-agi_evaluation_solutions.json")
    p.add_argument("--model", default="gpt-4o-2024-08-06")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_interactive(args)
