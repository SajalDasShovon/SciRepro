"""
agent.py - the auditing agent: its tools and its loop.

The model, not hard-coded orchestration, decides which files to read, which
diagnostic Python to run, and when to stop. It works only inside the study
folder it is given and has three tools: list_files, read_file, run_python.

Environment variables (defaults in brackets): MAX_AGENT_TURNS [20],
PYTHON_TIMEOUT_SECONDS [60].

SAFETY: run_python executes model-written code with your user permissions.
Run the agent only in an isolated container or VM.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from .model import ToolCall, env

MAX_AGENT_TURNS = int(env("MAX_AGENT_TURNS", "20"))
PYTHON_TIMEOUT_SECONDS = int(env("PYTHON_TIMEOUT_SECONDS", "60"))
TOOL_OUTPUT_MAX_CHARS = 12000
MAX_LISTED_FILES = 200


# ---------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in the scientific-analysis working directory.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a text, CSV, or Python file from the working directory. "
                "Use it to inspect study information, data, or pipeline code. "
                "Large files are truncated, so prefer run_python for data summaries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative filename, for example pipeline.py.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum number of characters to return (default 12000).",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute a Python script in the scientific-analysis working directory "
                "and return its stdout/stderr. Use it for diagnostic analyses, data "
                "checks, rerunning pipeline variants, and testing hypotheses. Each call "
                "is a fresh process, so re-load data in every call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Python code to execute.",
                    },
                },
                "required": ["code"],
            },
        },
    },
]


def list_files(workdir: Path) -> str:
    """All files, including subfolders; hidden files and __pycache__ skipped."""
    files = [p.relative_to(workdir) for p in sorted(workdir.rglob("*")) if p.is_file()]
    files = [p for p in files if not any(part.startswith(".") or part == "__pycache__" for part in p.parts)]
    rows = [f"{p} ({(workdir / p).stat().st_size} bytes)" for p in files[:MAX_LISTED_FILES]]
    if len(files) > MAX_LISTED_FILES:
        rows.append(f"... and {len(files) - MAX_LISTED_FILES} more files")
    return "\n".join(rows) or "(no files)"


def read_file(workdir: Path, path: str, max_chars=TOOL_OUTPUT_MAX_CHARS) -> str:
    full_path = (workdir / path).resolve()
    if not full_path.is_relative_to(workdir.resolve()):
        return "ERROR: Path must stay inside the working directory."
    if not full_path.is_file():
        return f"ERROR: {path} does not exist or is not a regular file."

    max_chars = max(1, min(int(max_chars), 50000))
    text = full_path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n...[truncated: showing {max_chars} of {len(text)} characters]"
    return text


SECRET_ENV_HINTS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL")


def run_python(workdir: Path, code: str) -> str:
    """Execute model-written Python in workdir. Not a sandbox: use a container."""
    env_vars = {k: v for k, v in os.environ.items()
                if not any(hint in k.upper() for hint in SECRET_ENV_HINTS)}
    env_vars.update(MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")   # no plot windows
    try:
        done = subprocess.run(
            [sys.executable, "-c", code], cwd=workdir, env=env_vars,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            stdin=subprocess.DEVNULL,   # input() must not hang on the terminal
            timeout=PYTHON_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: Python execution timed out after {PYTHON_TIMEOUT_SECONDS} s."

    text = ""
    if done.stdout:
        text += "STDOUT:\n" + done.stdout + "\n"
    if done.stderr:
        text += "STDERR:\n" + done.stderr + "\n"
    text += f"(exit code {done.returncode})"

    # Keep the head and the tail (tracebacks live at the end).
    if len(text) > TOOL_OUTPUT_MAX_CHARS:
        head = TOOL_OUTPUT_MAX_CHARS * 2 // 3
        tail = TOOL_OUTPUT_MAX_CHARS - head
        text = (text[:head] + f"\n...[{len(text) - TOOL_OUTPUT_MAX_CHARS} characters truncated]...\n"
                + text[-tail:])
    return text


def execute_tool(workdir: Path, call: ToolCall) -> str:
    """Run one tool. Errors are returned to the model, never raised."""
    args = call.arguments
    try:
        if call.name == "list_files":
            return list_files(workdir)
        if call.name == "read_file":
            if "path" not in args:
                return "ERROR: read_file requires a 'path' argument."
            return read_file(workdir, str(args["path"]), args.get("max_chars", TOOL_OUTPUT_MAX_CHARS))
        if call.name == "run_python":
            code = args.get("code")
            if not isinstance(code, str) or not code.strip():
                return "ERROR: run_python requires a non-empty 'code' string."
            return run_python(workdir, code)
        return f"ERROR: Unknown tool '{call.name}'. Available tools: list_files, read_file, run_python."
    except Exception as exc:  # report any tool failure to the model
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------

def system_prompt() -> str:
    libs = [name for name in ("numpy", "pandas", "scipy", "statsmodels", "matplotlib")
            if importlib.util.find_spec(name) is not None]
    lib_line = (
        f"Python libraries available to run_python: {', '.join(libs)} plus the standard library."
        if libs else
        "run_python has only the Python standard library (no numpy/pandas)."
    )
    return f"""
You are an autonomous scientific-analysis auditor.

Your job is to determine whether a reported result is trustworthy.

You have tools to list files, read files, and execute Python in the study's
working directory. Use them actively. Do not merely speculate.
{lib_line}

Audit the full path from raw data to the reported finding. Problems can arise
in data quality, preprocessing, how data sources are combined, implementation
details, analysis assumptions, or statistics.

Rules:
1. Inspect the study description and the pipeline code.
2. Reproduce the current result.
3. Form diagnostic hypotheses based on evidence.
4. Run tests that could support or reject each hypothesis.
5. Do not assume in advance that any particular bug exists.
6. Do not finish after only reading files. Run diagnostic analyses.
7. Keep each Python call focused and print compact summaries, not whole tables.
8. When the evidence is sufficient, stop calling tools and write a concise
   final report with:
      - whether a problem was found,
      - the evidence,
      - the likely cause,
      - a proposed correction and the corrected result, if you computed it,
      - whether the scientific conclusion changes.

Do not claim certainty beyond the evidence.
""".strip()


def banner(title: str, char: str = "=") -> None:
    print("\n" + char * 72 + "\n" + title + "\n" + char * 72)


def reproduction_prompt() -> str:
    return """You are an autonomous scientific reproduction agent. Use the supplied paper, datasets, and study instructions to independently reproduce reported analyses. Inspect inputs, determine the unit of analysis, implement preprocessing and computations in Python, and verify the results with executed evidence. Distinguish raw data counts from final analysis counts. Do not invent results or claim validation without running it. Report targets, computed results, discrepancies, code and evidence, and unresolved limitations."""


def run_agent(backend, workdir: Path, messages: list, mode: str = "audit") -> str:
    """
    The LLM decides which files to inspect, which hypotheses to test, which
    Python to run, and when to stop. Returns the final report. `messages` is
    filled in place so the caller can save it even on Ctrl-C.
    """
    messages.append({"role": "system", "content": reproduction_prompt() if mode == "reproduce" else system_prompt()})
    messages.append({"role": "user", "content": (workdir / "study.txt").read_text()
                     + "\n\nBegin the " + ("reproduction" if mode == "reproduce" else "audit") + ". Use the available tools."})

    def push_back(content, instruction):
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": instruction})

    python_runs = 0
    for turn_number in range(1, MAX_AGENT_TURNS + 1):
        banner(f"AGENT TURN {turn_number}/{MAX_AGENT_TURNS}")
        turn = backend.generate(messages, TOOLS)
        if turn.content:
            print(turn.content)

        if turn.parse_errors:
            # Malformed or cut-off tool call: report it, never run partial code.
            errors = "; ".join(turn.parse_errors)
            print("\nPARSE ERROR:", errors)
            push_back(turn.content or "(malformed tool call)",
                      f"Your last tool call could not be executed: {errors}. Re-issue it "
                      "in the correct format. If the code was long, split it into smaller steps.")

        elif turn.tool_calls:
            messages.append({"role": "assistant", "content": turn.content, "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in turn.tool_calls
            ]})
            for call in turn.tool_calls:
                print(f"\nTOOL REQUEST: {call.name}")
                if call.name == "run_python":
                    print(call.arguments.get("code", ""))
                elif call.arguments:
                    print(json.dumps(call.arguments, indent=2))
                observation = execute_tool(workdir, call)
                print("\nTOOL OBSERVATION:\n" + observation)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": observation})
                python_runs += call.name == "run_python"

        elif turn.truncated:
            push_back(turn.content, "Your response hit the length limit. "
                                    "Write the final report again, more concisely.")

        elif python_runs == 0:
            push_back(turn.content, "You have not run any diagnostic analysis yet. Use "
                                    "run_python to reproduce the result and test at least "
                                    "one hypothesis before concluding.")

        else:
            messages.append({"role": "assistant", "content": turn.content})
            return turn.content

    # Turn budget exhausted: ask for a report, and offer no tools so the
    # model cannot answer with another tool call instead of a report.
    banner("TURN BUDGET EXHAUSTED: REQUESTING FINAL REPORT")
    messages.append({"role": "user", "content": (
        "You have used your tool budget. Do not call any more tools. Write the "
        "final report now, based only on the evidence you have gathered."
    )})
    turn = backend.generate(messages, tools=None)
    messages.append({"role": "assistant", "content": turn.content})
    return turn.content or "(The agent produced no final report.)"
