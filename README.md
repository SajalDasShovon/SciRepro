# SciRepro: LLM Scientific Reproduction & Code Auditing

A reusable **ReAct-based baseline** for evaluating whether an LLM can reproduce analyses from a published paper and audit original analysis code. Adapted from the supplied `audit_health_agent` implementation; retains its tool-calling loop, Hugging Face and OpenAI-compatible backends, and Python execution tool.

## Minimal workflow

**1. Install** (Python 3.10+):

```bash
pip install -e '.[pdf,analysis]'
# For local Hugging Face inference, also install a compatible PyTorch build
# and: pip install -e '.[hf]'
```

**2. Put a paper and data in a folder:**

```text
my_study/
├── paper.pdf             # searchable/text-extractable PDF, or paper.txt
├── data/
│   ├── participants.csv
│   └── measurements.csv
└── analysis/             # OPTIONAL: if supplied, audit runs too
    └── preprocessing.py
```

**3. Run:**

```bash
scirepro run --model Qwen/Qwen3-Coder-30B-A3B-Instruct --study ./my_study
```

The default `--mode auto` runs **reproduction only** without original code and **reproduction + auditing** when original analysis code is detected. No `study.txt` is required; if supplied, SciRepro uses it instead of auto-generating instructions. For more controlled experiments, use `--mode reproduce`, `--mode audit`, or `--mode both`.

**Where do results go?** By default, into `./outputs/<UTC-timestamp>_<model>/` relative to **the terminal's current directory**, not the input study folder. Set `--output /path/to/results` to change this.

```text
outputs/20261118T190000000000Z_Qwen-Qwen3-Coder-30B-A3B-Instruct/
├── manifest.json               # model, settings, input SHA256 hashes, status
├── study/                      # untouched working copy of the supplied inputs
├── reproduce/
│   ├── study/                  # agent's own writable copy, study.txt, extracted text
│   ├── transcript.json         # full agent/tool conversation
│   └── report.md               # agent's final reproduction report
├── audit/                      # present if audit was run
│   ├── study/
│   ├── transcript.json
│   └── report.md
└── summary.md                  # combined reports, NOT an accuracy score
```

The original study folder is not modified. Separate working copies for reproduction and audit prevent results from one task influencing the other.

## Inference options

Local Hugging Face (default):

```bash
scirepro run --backend hf --model Qwen/Qwen3-Coder-30B-A3B-Instruct --study ./my_study
```

OpenAI-compatible inference server (e.g., local vLLM):

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY
scirepro run --backend openai --model your-served-model-name --study ./my_study
```

The OpenAI-compatible server must support tool calls in a format accepted by the original model adapter. Arbitrary model identifiers are **not guaranteed** to support the agent's tool-calling protocol. A GPU and substantial VRAM may be needed for local Qwen models.

Optional run controls:

```bash
scirepro run --study ./my_study --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --mode both --max-turns 18 --temperature 0 --output ./results
```

The underlying agent also reads `MAX_NEW_TOKENS`, `PYTHON_TIMEOUT_SECONDS`, `SEED`, `TOP_P`, `TOP_K`, `REPETITION_PENALTY` from environment variables. Temperature 0 does not guarantee bitwise-identical outputs across environments.

## Automatic study preparation

SciRepro extracts searchable PDF text with `pypdf`, inventories data and analysis files, and generates a **transparent** `study.txt` task for the agent. It does **not** reliably infer a complete statistical analysis plan, extract every figure/table, or generate validated reference answers. A researcher should check extracted PDF text, especially tables, scanned pages, and mathematical notation. For rigorous benchmark comparisons, provide a curated `study.txt` and independently verified target results; keep reference answers **outside** the agent-visible study folder.

### Existing `study.txt`

You may include `study.txt` in your study folder to reproduce a controlled experiment. This overrides automatic instructions. For example, existing HeartSteps benchmark instructions can be reused unchanged. Keep paper/data/code accessible in the same folder.

## Important evaluation distinction

SciRepro **runs an agent and records evidence**. Its `report.md` and `summary.md` contain the **agent's claims**, not independently validated correctness scores. Automated ground-truth grading for arbitrary papers is not implemented in this initial release; this cannot be inferred reliably from paper text alone. For a research benchmark, compare executed results against independently established reference statistics and, for audits, a known defect inventory. Never interpret agreement between the original and agent-corrected code as proof of validity.



