# ARC-AGI-v2 Interactive Tester (with Auto-Detect for Grids & Images)

This project builds an **interactive solver–evaluator–teacher loop** for ARC-AGI-v2 style puzzles.  
It now supports **both JSON grids and image-based inputs/outputs** with automatic detection, making it easier to run experiments even when your dataset mixes formats.  
The script uses the OpenAI Responses API to generate candidate outputs, applies sanity checks (no copying input / no downsizing), collects timing and token stats, and logs per-puzzle results.  
It supports running a limited number of puzzles and saves outputs to a timestamped result file.  
Optional rendering produces images of predicted grids.

---

## Features

- **Auto-detect grids vs. images**:  
  - Train input/output can be JSON grids or image filenames.  
  - Test input can be a grid or image filename.  
  - If image, the script quantizes it to 0–9 indexed colors (10 fixed-color palette).  
- **Flexible I/O modes** (`--io-mode` as hint, auto-detect still works):  
  - `grid2grid`: JSON grid → grid (default).  
  - `img2grid`: Image → grid → grid (logs grids).  
  - `img2grid2img`: Image → grid → grid → image (renders predictions to PNG).  
- **Flexible output size**: Schema allows outputs larger than input (up to `MAX_EXPAND` multiplier).  
- **Hint loop**: Simple teacher provides targeted hints between rounds.  
- **Sanity checks**: Rejects outputs smaller than input or exact copies of input.  
- **Stats collection**: Records time and tokens per round, per puzzle, and averages across all.  
- **Pretty grid printouts**: Outputs prediction grids as neat rectangles.  
- **Structured logs**: Results saved to `result/result_YYYYMMDD_HHMMSS.txt`.  
- **Optional rendering**: With `--io-mode img2grid2img`, saves predicted grids as PNG images.  

---

## Requirements

- Python 3.8+  
- Packages:  
  - `openai>=1.30.0`  
  - `pillow` (for image I/O)  
  - `numpy` (for image quantization)  
- An OpenAI API key (`OPENAI_API_KEY`)

### Install

```bash
pip install "openai>=1.30.0" pillow numpy
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

---

## Data Formats

### Challenges (IDs at top level)

Grids (classic ARC style):

```json
{
  "PUZ001": {
    "train": [ { "input": [[0,0],[1,1]], "output": [[1,1],[0,0]] } ],
    "test":  [ { "input": [[0,1],[1,0]] } ]
  }
}
```

Images (filenames instead of raw grids):

```json
{
  "PUZ001": {
    "train": [ { "input": "PUZ001_in.png", "output": "PUZ001_out.png" } ],
    "test":  [ { "input": "PUZ001_test.png" } ]
  }
}
```

### Solutions (optional)

- Grids:

```json
{
  "PUZ001": [ [[1,0],[0,1]] ]
}
```

- Or image filenames:

```json
{
  "PUZ001": [ "PUZ001_solution.png" ]
}
```

---

## Usage

Script: `interactive_tester.py`

### Example: grid → grid

```bash
python interactive_tester.py   --data data/arc_challenges.json   --solutions data/arc_solutions.json   --prompt arc.txt   --model gpt-4o-2024-08-06   --io-mode grid2grid   --rounds 3   --limit 5
```

### Example: image → grid → grid

```bash
python interactive_tester.py   --data data/arc_challenges.json   --prompt arc.txt   --io-mode img2grid   --image-dir images/   --limit 3
```

### Example: image → grid → grid → image (renders predictions)

```bash
python interactive_tester.py   --data data/arc_challenges.json   --prompt arc.txt   --io-mode img2grid2img   --image-dir images/   --render-scale 16   --limit 2
```

---

## Key CLI Options

- `--data` (required): path to challenges JSON.  
- `--solutions` (optional): path to solutions JSON for scoring.  
- `--prompt` (required): template filename under `./prompts` (e.g., `arc.txt`).  
- `--model`: OpenAI model ID (default: `gpt-4o-2024-08-06`).  
- `--rounds`: max interactive rounds per puzzle (default 5).  
- `--k`: candidates per round (default 1). Increase for self-consistency.  
- `--temperature`, `--top_p`: sampling controls.  
- `--limit`: number of puzzle IDs to evaluate (default 10).  
- `--io-mode`: preferred I/O mode (grid2grid/img2grid/img2grid2img).  
- `--image-dir`: directory containing puzzle images (if JSON references PNG/JPG).  
- `--render-scale`: pixel scale when rendering predictions to PNG (default 16).  
- `--verbose`: print extra debug logs.  

---

## Output

- Results saved to `result/result_YYYYMMDD_HHMMSS.txt`.  
- With `--io-mode img2grid2img`, rendered outputs saved under `result/renders_YYYYMMDD_HHMMSS/`.

Each puzzle section includes:
- Puzzle ID & model  
- Per-round results (accuracy, time, tokens, hints)  
- Best accuracy & best prediction grid (rectangle print)  
- Puzzle-level stats (time, tokens, rounds used)  

Final section shows totals and averages across all puzzles.

---

## Project Structure

```
project/
├─ interactive_tester.py
├─ prompts/
│  └─ arc.txt
├─ data/
│  ├─ arc_challenges.json
│  └─ arc_solutions.json
├─ images/                # if JSON references image files
└─ result/
   ├─ result_20250929_153022.txt
   └─ renders_20250929_153022/
```