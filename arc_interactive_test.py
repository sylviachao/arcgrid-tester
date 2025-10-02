#!/usr/bin/env python
"""
ARC-AGI-v2 Interactive Tester (Solver–Evaluator–Teacher Loop)

This version adds AUTO-DETECT for train/test data:
- Train pairs can be provided as numeric grids OR image filenames.
- Test input can be provided as a numeric grid OR an image filename.
- If an image is detected, it is read from --image-dir and quantized to a 0..9 grid.

I/O MODES (now act as hints; AUTO-DETECT still works regardless):
    • grid2grid       : treat inputs as grids when possible (default)
    • img2grid        : prefer image inputs for test if available (but auto-detect will handle either)
    • img2grid2img    : same as above, plus render predicted grid as PNG

STATS:
    • Per-round, per-puzzle, and overall timing (wall clock)
    • Per-round, per-puzzle, and overall token usage (supports new & legacy fields)

PRETTY PRINT:
    • Best prediction grid printed as a rectangle

PROMPTS:
    • Reads template from ./prompts/<name> and formats with {train_pairs} / {test_input}

OUTPUT:
    • Results saved to result/result_YYYYMMDD_HHMMSS.txt
    • When --io-mode img2grid2img, rendered outputs saved under result/renders_YYYYMMDD_HHMMSS/<pid>.png
"""

import os
import sys
import json
import time
import argparse
import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

# ---------- Optional image support ----------
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    import numpy as np
    NP_AVAILABLE = True
except Exception:
    NP_AVAILABLE = False

# ---------- OpenAI client ----------
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

# Fixed 10-color palette (RGB) for ARC-like grids.
PALETTE10 = None
if NP_AVAILABLE:
    PALETTE10 = np.array([
        [0,0,0], [0,0,255], [0,255,0], [255,0,0], [255,255,0],
        [255,0,255], [0,255,255], [128,128,128], [255,165,0], [255,255,255]
    ], dtype=np.uint8)

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
                raise ValueError(f"Value out of range [{minv},{maxv}]: {ival}")
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

def load_prompt_template(name: str) -> str:
    """
    Load prompt template text file from ./prompts/ directory.
    Pass --prompt arc.txt (or any filename living under ./prompts).
    Template must contain {train_pairs} and {test_input}.
    """
    path = Path("prompts") / f"{name}"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")

def build_base_prompt(dataset_template_name: str, train_pairs: Any, test_input: Any) -> str:
    """
    使用精簡 JSON（去空白）以降低 token。
    """
    template = load_prompt_template(dataset_template_name)

    # 關鍵：去掉多餘空白與換行
    train_str = json.dumps(train_pairs, ensure_ascii=False, separators=(',', ':'))
    test_str  = json.dumps(test_input,  ensure_ascii=False, separators=(',', ':'))

    has_train_ph = "{train_pairs}" in template
    has_test_ph  = "{test_input}"  in template

    if has_train_ph or has_test_ph:
        return template.format(train_pairs=train_str, test_input=test_str)

    appended = (
        template.rstrip() + "\n\n"
        "TRAINING PAIRS:\n" + train_str + "\n\n"
        "TEST INPUT:\n" + test_str
    )
    return appended

def format_grid(grid: Any) -> str:
    """Format a 2D grid (list of lists) into a neat rectangular string."""
    if grid is None:
        return "None"
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)

# ----- Timing & token helpers -----

def _fmt_secs(s: float) -> str:
    return f"{s*1000:.0f} ms" if s < 1 else f"{s:.2f} s"

def _sum_usage_from_responses(responses) -> dict:
    """
    Sum token usage from a list of raw response objects (OpenAI Responses API).
    Supports:
      - new style: usage.input_tokens / usage.output_tokens / usage.total_tokens
      - legacy style: usage.prompt_tokens / usage.completion_tokens / usage.total_tokens
    """
    total = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    if not responses:
        return total

    def _get(u, key, default=0):
        if isinstance(u, dict):
            return int(u.get(key, default) or 0)
        return int(getattr(u, key, default) or 0)

    for r in responses:
        usage = None
        if isinstance(r, dict):
            usage = r.get("usage")
        else:
            usage = getattr(r, "usage", None)
        if not usage:
            continue

        # New-style
        total["input_tokens"]      += _get(usage, "input_tokens")
        total["output_tokens"]     += _get(usage, "output_tokens")
        total["total_tokens"]      += _get(usage, "total_tokens")
        # Legacy
        total["prompt_tokens"]     += _get(usage, "prompt_tokens")
        total["completion_tokens"] += _get(usage, "completion_tokens")

        # If only legacy present and total is zero, infer
        if _get(usage, "total_tokens", 0) == 0 and (_get(usage, "prompt_tokens", 0) or _get(usage, "completion_tokens", 0)):
            total["total_tokens"] += _get(usage, "prompt_tokens", 0) + _get(usage, "completion_tokens", 0)

    return total

# ---------------------------
# Image <-> Grid (auto-detect support)
# ---------------------------

def _check_image_mode_ok():
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is required for image inputs. Install: pip install pillow")
    if not NP_AVAILABLE:
        raise RuntimeError("NumPy is required for image inputs. Install: pip install numpy")

def read_image(path: Path) -> "Image.Image":
    return Image.open(path).convert("RGB")

def image_to_grid(img: "Image.Image") -> Grid:
    """Quantize an RGB image to 10-color indices (0..9) using nearest palette color."""
    _check_image_mode_ok()
    arr = np.asarray(img).astype(np.int16)  # HxWx3
    pal = PALETTE10.astype(np.int16)        # 10x3
    dists = np.sum((arr[:, :, None, :] - pal[None, None, :, :])**2, axis=3)  # HxWx10
    idx = np.argmin(dists, axis=2)          # HxW
    return idx.tolist()

def grid_to_image(grid: Grid, scale: int = 16) -> "Image.Image":
    """Render a 0..9 grid to an RGB image via the fixed palette, scaled for visibility."""
    _check_image_mode_ok()
    g = np.array(grid, dtype=np.int32)
    rgb = PALETTE10[g]  # HxWx3
    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    if scale and scale != 1:
        img = img.resize((img.width*scale, img.height*scale), Image.NEAREST)
    return img

# --- AUTO-DETECT utilities ---

def _looks_like_grid(x: Any) -> bool:
    """Heuristic: a grid is a list of lists of ints within COLOR_MIN..COLOR_MAX."""
    if not isinstance(x, list) or not x:
        return False
    if not isinstance(x[0], list):
        return False
    try:
        _ = ensure_int_grid_flexible(x, COLOR_MIN, COLOR_MAX)
        return True
    except Exception:
        return False

def _looks_like_image_ref(x: Any) -> bool:
    """
    Heuristic: image reference can be a string filename, or dict with key 'file' or 'path'.
    Examples: "TRAIN0001_input.png", {"file": "TRAIN0001_out.png"}, {"path": "a/b.png"}
    """
    if isinstance(x, str):
        return any(x.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg"])
    if isinstance(x, dict):
        for key in ("file", "path", "image", "img"):
            v = x.get(key)
            if isinstance(v, str) and any(v.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg"]):
                return True
    return False

def _extract_image_filename(x: Union[str, Dict[str, Any]]) -> str:
    if isinstance(x, str):
        return x
    for key in ("file", "path", "image", "img"):
        v = x.get(key)
        if isinstance(v, str):
            return v
    raise ValueError(f"Cannot extract image filename from: {x}")

def _coerce_to_grid(value: Any, image_dir: Path, allow_images: bool) -> Grid:
    """
    Convert 'value' to a numeric grid (0..9).
    - If it's already a grid, validate & return.
    - If it's an image filename (detected), load from image_dir and quantize (if allow_images=True).
    """
    if _looks_like_grid(value):
        return ensure_int_grid_flexible(value, COLOR_MIN, COLOR_MAX)

    if _looks_like_image_ref(value):
        if not allow_images:
            raise ValueError("Image input not allowed here, but image reference detected.")
        if not image_dir:
            raise ValueError("Image reference detected but --image-dir is not provided.")
        fname = _extract_image_filename(value)
        img_path = image_dir / fname
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        return image_to_grid(read_image(img_path))

    raise ValueError("Value is neither a valid grid nor a recognizable image reference.")

def load_train_pairs_auto(train_pairs_raw: List[Dict[str, Any]], image_dir: Path, allow_images: bool) -> List[Dict[str, Grid]]:
    """
    For each training pair, auto-detect whether 'input'/'output' are grids or image filenames.
    Returns: list of dicts with numeric grids.
    """
    out_pairs: List[Dict[str, Grid]] = []
    for i, p in enumerate(train_pairs_raw):
        if not isinstance(p, dict) or "input" not in p or "output" not in p:
            raise ValueError(f"Malformed train pair at index {i}: {p}")
        g_in  = _coerce_to_grid(p["input"], image_dir, allow_images)
        g_out = _coerce_to_grid(p["output"], image_dir, allow_images)
        out_pairs.append({"input": g_in, "output": g_out})
    return out_pairs

def load_test_input_auto(test_list: List[Dict[str, Any]], image_dir: Path, allow_images: bool) -> Grid:
    """
    Auto-detect for test input. Uses the first test item by convention (ARC style).
    """
    if not test_list or not isinstance(test_list, list) or not isinstance(test_list[0], dict) or "input" not in test_list[0]:
        raise ValueError("Malformed test section; expected a list with dict having key 'input'.")
    return _coerce_to_grid(test_list[0]["input"], image_dir, allow_images)

# ---------------------------
# Model calls
# ---------------------------

def solve_round_messages(client: OpenAI, model: str, messages: List[Dict[str, str]],
                         schema: Dict[str, Any], temperature: float, top_p: float, k: int):
    """
    使用多訊息累積的方式呼叫 Responses API，不再每回合重複貼整段 prompt。
    Args:
        messages: 例如 [
            {"role": "system", "content": "..."},
            {"role": "user",   "content": base_prompt},
            {"role": "user",   "content": "HINT: ..."},  # 第 r 回合新增
        ]
    Returns:
        candidates: List[Grid]
        responses : List[Any]
    """
    candidates: List[Grid] = []
    responses: List[Any] = []

    for _ in range(k):
        resp = client.responses.create(
            model=model,
            input=messages,  # 核心改寫：多訊息陣列
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
        except Exception:
            pass
        responses.append(resp)
    return candidates, responses

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
# Main loop (AUTO-DETECT integrated)
# ---------------------------

def run_interactive(args: argparse.Namespace) -> None:
    # load data
    challenges = read_data(args.data)
    solutions = read_data(args.solutions) if args.solutions else {}

    prompt_template_name = args.prompt  # will be loaded later

    all_ids = list(challenges.keys())
    limit = min(args.limit, len(all_ids))
    target_ids = all_ids[:limit]

    client = OpenAI()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path("result")
    result_dir.mkdir(exist_ok=True)
    out_path = result_dir / f"result_{timestamp}.txt"

    # overall accumulators
    overall_time_sec = 0.0
    overall_rounds = 0
    overall_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

    # Prepare optional render dir for img2grid2img
    render_dir = result_dir / f"renders_{timestamp}"

    # Determine whether images are allowed/expected
    # (acts as a hint; auto-detect still works either way)
    allow_image_inputs = args.io_mode in ("img2grid", "img2grid2img") or bool(args.image_dir)
    image_dir = Path(args.image_dir) if args.image_dir else None

    for pid in target_ids:
        puzzle_t0 = time.perf_counter()

        log_lines: List[str] = []
        log_lines.append(f"Puzzle ID: {pid}")
        log_lines.append(f"Model: {args.model}")
        log_lines.append(f"IO Mode: {args.io_mode}")
        log_lines.append("-" * 60)

        # Raw from JSON
        obj = challenges[pid]
        train_pairs_raw = obj["train"]
        test_raw = obj["test"]

        # Auto-coerce training pairs (grids or images)
        if allow_image_inputs:
            if image_dir is None and any(
                _looks_like_image_ref(p.get("input")) or _looks_like_image_ref(p.get("output"))
                for p in train_pairs_raw
            ):
                raise ValueError("Train pairs reference images, but --image-dir is not provided.")
            train_pairs = load_train_pairs_auto(train_pairs_raw, image_dir or Path("."), allow_images=True)
        else:
            # Attempt to treat as grids strictly
            train_pairs = load_train_pairs_auto(train_pairs_raw, Path("."), allow_images=False)

        # Decide effective test grid (auto-detect: image or grid)
        if allow_image_inputs and any(_looks_like_image_ref(t.get("input")) for t in test_raw):
            if image_dir is None:
                raise ValueError("Test input references an image, but --image-dir is not provided.")
            test_grid = load_test_input_auto(test_raw, image_dir, allow_images=True)
        else:
            # If io-mode hints image but test is a grid, we still accept the grid.
            test_grid = load_test_input_auto(test_raw, image_dir or Path("."), allow_images=False)

        # gold (optional)
        gold_output = None
        if pid in solutions:
            sols = solutions[pid]
            if isinstance(sols, list) and sols:
                # Allow gold as grid or image filename
                try:
                    gold_output = _coerce_to_grid(sols[0], image_dir or Path("."), allow_images=allow_image_inputs)
                except Exception:
                    # fallback: maybe it's already a grid but allow_images=False raised
                    if _looks_like_grid(sols[0]):
                        gold_output = ensure_int_grid_flexible(sols[0], COLOR_MIN, COLOR_MAX)
                    else:
                        raise

        # schema & prompt
        Hi, Wi = len(test_grid), len(test_grid[0])
        schema = build_schema_relaxed(Hi, Wi)
        base_prompt = build_base_prompt(prompt_template_name, train_pairs, test_grid)

        # 多訊息累積：base_prompt 只出現一次，後續每回合只追加 HINT
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": "Output ONLY JSON that matches the schema."},
            {"role": "user",   "content": base_prompt},
        ]

        best_acc = -1.0
        best_pred: Grid = None

        # per-puzzle accumulators
        rounds_used = 0
        puzzle_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        per_round_times: List[float] = []
        per_round_token_totals: List[int] = []

        for r in range(1, args.rounds + 1):
            round_t0 = time.perf_counter()

            cands, responses = solve_round_messages(
                client, args.model, messages, schema,
                args.temperature, args.top_p, args.k
            )

            round_dt = time.perf_counter() - round_t0
            rounds_used += 1
            overall_rounds += 1
            per_round_times.append(round_dt)

            # tokens for this round
            u = _sum_usage_from_responses(responses)
            for k_ in puzzle_usage:
                puzzle_usage[k_] += u.get(k_, 0)
            per_round_token_totals.append(
                u.get("total_tokens", 0) or (u.get("prompt_tokens", 0) + u.get("completion_tokens", 0))
            )

            if not cands:
                log_lines.append(f"[{pid}][Round {r}] No candidates. "
                                 f"(time: {_fmt_secs(round_dt)}, tokens: {per_round_token_totals[-1]})")
                break

            scored = []
            for g in cands:
                try:
                    g_fixed = ensure_int_grid_flexible(g, COLOR_MIN, COLOR_MAX)
                except Exception as e:
                    scored.append((-1.0, None, {"ok": False, "reason": f"parse_error: {e}"}))
                    continue

                if too_small(g_fixed, test_grid):
                    rep = {"ok": False, "reason": "too_small"}
                    acc = -1.0
                elif is_copy_of_input(g_fixed, test_grid):
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

            # round log with time & tokens
            if gold_output is None:
                log_lines.append(f"[{pid}][Round {r}] Prediction accepted (no gold). "
                                 f"(time: {_fmt_secs(round_dt)}, tokens: {per_round_token_totals[-1]})")
            else:
                if rep.get("ok"):
                    log_lines.append(f"[{pid}][Round {r}] Accuracy: {acc:.2f}% "
                                     f"(time: {_fmt_secs(round_dt)}, tokens: {per_round_token_totals[-1]})")
                else:
                    log_lines.append(f"[{pid}][Round {r}] Rejected: {rep.get('reason')} "
                                     f"(time: {_fmt_secs(round_dt)}, tokens: {per_round_token_totals[-1]})")

            # early stop if solved perfectly
            if gold_output and rep.get("ok") and acc >= 100.0:
                log_lines.append(f"[{pid}][Round {r}] SOLVED ✅")
                best_acc, best_pred = acc, pred
                break

            if gold_output and rep.get("ok") and acc > best_acc:
                best_acc, best_pred = acc, pred
            elif gold_output is None and rep.get("ok"):
                best_pred = pred

            # 只追加「新的提示訊息」，避免 prompt 膨脹
            hint = make_hint(train_pairs, test_grid, pred, gold_output, rep)
            log_lines.append(f"[{pid}][Round {r}] HINT: {hint}")
            messages.append({"role": "user", "content": f"HINT:\n{hint}"})

        # Footer for this puzzle
        log_lines.append("-" * 60)
        if gold_output is not None:
            log_lines.append(f"[{pid}] Best accuracy: {best_acc:.2f}%")

        # Pretty-printed grid rectangle
        log_lines.append(f"[{pid}] Best prediction grid:\n{format_grid(best_pred)}")

        # If in img2grid2img, render output
        if args.io_mode == "img2grid2img" and best_pred is not None:
            render_dir.mkdir(exist_ok=True)
            out_img = grid_to_image(best_pred, scale=args.render_scale)
            out_path_img = render_dir / f"{pid}.png"
            out_img.save(out_path_img)
            log_lines.append(f"[{pid}] Render saved: {out_path_img}")

        # Per-puzzle time & tokens
        puzzle_dt = time.perf_counter() - puzzle_t0
        overall_time_sec += puzzle_dt
        for k_ in overall_usage:
            overall_usage[k_] += puzzle_usage.get(k_, 0)

        avg_round_time = (sum(per_round_times) / rounds_used) if rounds_used else 0.0
        avg_round_tokens = (sum(per_round_token_totals) / rounds_used) if rounds_used else 0.0

        log_lines.append(f"[{pid}] Rounds used: {rounds_used}/{args.rounds}")
        log_lines.append(f"[{pid}] Puzzle time: {_fmt_secs(puzzle_dt)} "
                         f"(avg/round: {_fmt_secs(avg_round_time)})")

        # Prefer new-style token fields when available
        if puzzle_usage["total_tokens"] > 0:
            log_lines.append(f"[{pid}] Tokens (input/output/total): "
                             f"{puzzle_usage['input_tokens']}/{puzzle_usage['output_tokens']}/{puzzle_usage['total_tokens']} "
                             f"(avg/round: {int(avg_round_tokens)})")
        else:
            legacy_total = puzzle_usage["prompt_tokens"] + puzzle_usage["completion_tokens"]
            log_lines.append(f"[{pid}] Tokens (prompt/completion/total): "
                             f"{puzzle_usage['prompt_tokens']}/{puzzle_usage['completion_tokens']}/{legacy_total} "
                             f"(avg/round: {int(avg_round_tokens)})")

        with out_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")

    # Overall summary
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append(f"TOTAL puzzles: {len(target_ids)}")
    summary_lines.append(f"TOTAL time: {_fmt_secs(overall_time_sec)}")
    if len(target_ids) > 0:
        summary_lines.append(f"AVG time per puzzle: {_fmt_secs(overall_time_sec / len(target_ids))}")
    summary_lines.append(f"TOTAL rounds executed: {overall_rounds}")

    if overall_usage["total_tokens"] > 0:
        summary_lines.append("TOTAL tokens (input/output/total): "
                             f"{overall_usage['input_tokens']}/"
                             f"{overall_usage['output_tokens']}/"
                             f"{overall_usage['total_tokens']}")
        if overall_rounds > 0:
            summary_lines.append("AVG tokens per round (input/output/total): "
                                 f"{overall_usage['input_tokens'] // max(overall_rounds,1)}/"
                                 f"{overall_usage['output_tokens'] // max(overall_rounds,1)}/"
                                 f"{overall_usage['total_tokens'] // max(overall_rounds,1)}")
    else:
        legacy_total = overall_usage["prompt_tokens"] + overall_usage["completion_tokens"]
        summary_lines.append("TOTAL tokens (prompt/completion/total): "
                             f"{overall_usage['prompt_tokens']}/"
                             f"{overall_usage['completion_tokens']}/"
                             f"{legacy_total}")
        if overall_rounds > 0:
            summary_lines.append("AVG tokens per round (prompt/completion/total): "
                                 f"{overall_usage['prompt_tokens'] // max(overall_rounds,1)}/"
                                 f"{overall_usage['completion_tokens'] // max(overall_rounds,1)}/"
                                 f"{legacy_total // max(overall_rounds,1)}")

    with out_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"Finished evaluation of {len(target_ids)} puzzles. Results saved to {out_path}")

# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ARC-AGI-v2 interactive tester (auto-detect grids/images)")
    p.add_argument("--data", required=True, help="Path to challenges file (JSON)")
    p.add_argument("--solutions", required=False, help="Path to solutions file (JSON)")
    p.add_argument("--prompt", required=True, help="Prompt template filename under ./prompts (e.g., arc.txt)")
    p.add_argument("--model", default="gpt-4o-2024-08-06")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--verbose", action="store_true")

    # I/O modes (hints)
    p.add_argument("--io-mode",
                   choices=["grid2grid", "img2grid", "img2grid2img"],
                   default="grid2grid",
                   help="Hint for expected inputs/outputs (auto-detect still applies).")
    p.add_argument("--image-dir", default=None,
                   help="Directory holding images referenced by JSON (train/test can be .png/.jpg).")
    p.add_argument("--render-scale", type=int, default=16,
                   help="Scale factor for rendering predicted grid to image (img2grid2img).")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_interactive(args)

