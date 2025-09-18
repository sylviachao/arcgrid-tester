# ARC-AGI-v2 Interactive Tester

This project builds an interactive solver–evaluator–teacher loop for ARC-AGI-v2 style puzzles.  
The script reads challenge tasks and (optionally) solutions, generates candidate outputs with the OpenAI API, applies sanity checks (no copying input / no downsizing), and logs per-ID results.  
It supports running a limited number of puzzles and saves outputs to a timestamped result file.  
Sample evaluation data is included in `data/evaluation`.

## Features

- **Flexible output size**: JSON schema allows outputs larger than input (configurable multiplier).  
- **Hint loop**: simple teacher provides targeted hints between rounds.  
- **Dual-file support**: reads `arc-agi_evaluation_challenges.json` + `arc-agi_evaluation_solutions.json` (optional).  
- **Batch control**: `--limit` runs only the first N puzzle IDs safely (`min(limit, total)`).  
- **Structured logs**: results saved to `result/result_YYYYMMDD_HHMMSS.txt`.

## Requirements

- Python 3.8+  
- `openai>=1.30.0`  
- An OpenAI API key (`OPENAI_API_KEY`)

## Installation

```bash
pip install "openai>=1.30.0"
```

Set API key:

- **Linux/macOS**
  ```bash
  export OPENAI_API_KEY="sk-..."
  ```
- **Windows PowerShell**
  ```powershell
  setx OPENAI_API_KEY "sk-..."
  ```
  *(Open a NEW PowerShell window so the env var takes effect.)*

## Data Formats

### Challenges (IDs at top level)

```json
{
  "00576224": {
    "train": [ { "input": [[...]], "output": [[...]] }, ... ],
    "test":  [ { "input": [[...]] } ]
  },
  "007bbfb7": { ... }
}
```

### Solutions (optional)

Common case (list of grids, first is the gold output):

```json
{
  "00576224": [ [[...], [...]] ],
  "007bbfb7": [ [[...], [...]] ]
}
```

If your solutions file wraps keys (e.g., `{ "solutions": { ... } }`) or stores a single grid directly (not in a list), adjust the loader accordingly (the provided script is easy to extend to auto-detect).

## Usage

Script: `arc_interactive_test.py`

### PowerShell example

```powershell
python arc_interactive_test.py `
  --data arctest2025/data/evaluation/arc-agi_evaluation_challenges.json `
  --solutions arctest2025/data/evaluation/arc-agi_evaluation_solutions.json `
  --model gpt-4o-2024-08-06 `
  --rounds 3 `
  --limit 5 `
  --verbose
```

### Bash example

```bash
python arc_interactive_test.py   --data arctest2025/data/evaluation/arc-agi_evaluation_challenges.json   --solutions arctest2025/data/evaluation/arc-agi_evaluation_solutions.json   --model gpt-4o-2024-08-06   --rounds 3   --limit 5   --verbose
```

## Key CLI Options

- `--data` (required): path to challenges JSON.  
- `--solutions` (optional): path to solutions JSON for scoring.  
- `--rounds`: max interactive rounds per puzzle (default 5).  
- `--k`: candidates per round (default 1). Increase for self-consistency.  
- `--temperature`, `--top_p`: sampling controls.  
- `--limit`: number of puzzle IDs to evaluate (default 10, capped by available IDs).  
- `--verbose`: print raw JSON outputs for debugging.  

## Output

Results saved under `result/result_YYYYMMDD_HHMMSS.txt`.

Each puzzle section includes:
- Model & rounds  
- Per-round status (accuracy if gold is present; otherwise “prediction accepted (no gold)”)  
- Hints used between rounds  
- Best accuracy (with gold) and the best predicted grid  

## Improving Accuracy (Tips)

- Increase `--k` (e.g., 5–10) to sample more candidates per round and pick the best.  
- Extend the teacher to check for tiling/scale factors, rotations/reflections, color remapping, border framing.  
- Add task-specific reminders to the prompt if needed.  

## Project Structure

```
arctest2025/
└─ data/
   └─ evaluation/
      ├─ arc-agi_evaluation_challenges.json
      └─ arc-agi_evaluation_solutions.json
arc_interactive_test.py
result/
README.md
```
