# SciRepro: LLM-Based Scientific Reproduction & Code Auditing

**SciRepro** is an open-source, ReAct-based framework for evaluating whether Large Language Models (LLMs) can reproduce scientific analyses from published papers and audit existing analysis code.

Researchers can provide a **scientific paper, datasets, and an LLM of their choice**. SciRepro runs the agent and generates an analysis report and execution transcript.

## Quick Start

### 1. Clone and Install

Requires Python 3.10+.

```bash
git clone https://github.com/SajalDasShovon/SciRepro.git
cd SciRepro

pip install -e ".[pdf,analysis]"
```

For local Hugging Face models, also install a compatible PyTorch build and:

```bash
pip install -e ".[hf]"
```

### 2. Prepare Your Study

Organize your paper and datasets as follows:

```text
my_study/
├── paper.pdf              # Or paper.txt
├── data/
│   ├── dataset1.csv
│   └── dataset2.csv
├── analysis/              # Optional: original analysis code
│   └── preprocessing.py
└── study.txt              # Optional: custom instructions
```

**Only the paper and data are required.** If `study.txt` is not provided, SciRepro automatically generates initial reproduction instructions.

### 3. Run an LLM Agent

Using the default Qwen model:

```bash
scirepro run --study ./my_study
```

Using a different Hugging Face model:

```bash
scirepro run \
  --study ./my_study \
  --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --max-turns 5 \
  --temperature 0
```

Using an OpenAI-compatible inference endpoint:

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY

scirepro run \
  --study ./my_study \
  --backend openai \
  --model your-served-model-name
```

The selected model must be compatible with the agent's tool-calling format.

### 4. View the Results

Results are automatically saved in:

```text
outputs/<timestamp>_<model>/
├── manifest.json
├── reproduce/
│   ├── transcript.json
│   └── report.md
└── summary.md
```

If code auditing is performed, an `audit/` directory is also generated.

To view the latest reproduction report:

```bash
RUN_DIR=$(ls -td outputs/*/ | head -1)
cat "${RUN_DIR}reproduce/report.md"
```

## Additional Options

| Argument | Description |
|---|---|
| `--model` | Select an LLM |
| `--backend` | `hf` or `openai` |
| `--mode` | `auto`, `reproduce`, `audit`, `both` |
| `--max-turns` | Maximum agent turns (default: 20) |
| `--temperature` | Model sampling temperature |
| `--output` | Custom results directory |

By default, SciRepro runs reproduction and additionally performs code auditing when original analysis code is provided.

