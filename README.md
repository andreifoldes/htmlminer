# HTMLMiner

Agentic web domain analyzer using DSpy and LangExtract.

## Installation

### What is `uv`?
`uv` is a fast Python package and project manager from Astral. We use it to create an isolated environment and run the CLI consistently across machines.

### Install `uv` (any OS)
Pick one option for your OS, then confirm with `uv --version`.

**macOS**
```bash
brew install uv
```

**Linux / macOS (installer script)**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

**Windows (winget)**
```powershell
winget install --id Astral.UV
```

From the project root (the folder that contains `pyproject.toml`), install HTMLMiner in editable mode:
```bash
uv pip install -e .
```
This creates a local virtual environment (if needed), installs dependencies, and links the package to your working copy so changes in `src/` are picked up immediately without reinstalling.

## Usage

### Batch Processing (File)
```bash
uv run htmlminer process --file test_urls.md
```

### Single URL Processing
```bash
uv run htmlminer process --url https://deepmind.google/about/
```

### Using Firecrawl Engine
```bash
uv run htmlminer process --url https://openai.com/safety/ --engine firecrawl
```

### Controlling Summary Length
Limit the max paragraphs per dimension (Risk, Goal, Method) in the final summary (default is 3):
```bash
uv run htmlminer process --url https://anthropic.com/ --max-paragraphs 2
```

### Choosing Gemini Model Tier
Select a cheaper or more capable model for extraction and synthesis:
```bash
uv run htmlminer process --url https://anthropic.com/ --gemini-tier expensive
```

## Configuration

### `config.json`
`config.json` controls *what* the agent extracts. Each entry in `features` defines:
- `name`: the label for the dimension in results
- `description`: what the extractor should look for
- `synthesis_topic`: how the summary for that dimension should be framed

If you add or edit features, keep valid JSON and the same field names. A malformed `config.json` will stop the run with a parse error.

## Improving Extraction Quality

Try small CLI tweaks before changing code:
- **Model tier:** `--gemini-tier expensive` yields better extraction quality at higher cost.
- **Engine choice:** `--engine firecrawl` often captures richer content; `--engine trafilatura` can be cleaner for text-heavy pages.
- **Summary depth:** increase `--max-paragraphs` for more detail (or reduce it for faster, tighter outputs).
- **Input scope:** use `--file` with a curated URL list to avoid low-signal pages.

Note: there is no dedicated verbosity flag yet. For troubleshooting, check `htmlminer_logs.db` and consider adding a verbosity option if you need more console detail.

### Full CLI Options
```text
  --file TEXT             Path to markdown file containing URLs
  --url TEXT              Single URL to process
  --output TEXT           Path to output CSV file [default: results.csv]
  --engine TEXT           Engine to use: 'firecrawl' or 'trafilatura' [default: firecrawl]
  --max-paragraphs INT    Max paragraphs per dimension in agentic summary [default: 3]
  --gemini-tier TEXT      Gemini model tier: 'cheap' or 'expensive' [default: cheap]
  --help                  Show this message and exit.
```
