"""
model.py - the LLM behind the audit agent.

Default model: Qwen/Qwen3-Coder-30B-A3B-Instruct (30.5B MoE, 3.3B active).
Other choices (set MODEL_NAME):
    Qwen/Qwen3-30B-A3B-Instruct-2507     same size, general instruct model
    Qwen/Qwen3-235B-A22B-Instruct-2507   much larger, use vLLM on several GPUs
    Qwen/Qwen3-4B-Instruct-2507          small, for quick smoke tests
Both Qwen tool-call formats are parsed: JSON (Qwen3 / 2507 instruct models)
and XML (Qwen3-Coder models).

Backends (set BACKEND):
    hf      Hugging Face transformers, fully local (default). The 30B models
            need ~62 GB of GPU memory in bf16; device_map="auto" splits the
            model over all visible GPUs.
    openai  any OpenAI-compatible server, e.g. vLLM (much faster for MoE):
                vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \\
                    --enable-auto-tool-choice --tool-call-parser qwen3_coder \\
                    --max-model-len 65536
            Use --tool-call-parser hermes for the *-Instruct-2507 models.
            The client then needs no torch.

Environment variables (defaults in brackets): BACKEND [hf], MODEL_NAME,
OPENAI_BASE_URL [http://localhost:8000/v1], OPENAI_API_KEY [EMPTY],
MAX_NEW_TOKENS [4096], TEMPERATURE [0.7, use 0 for greedy], TOP_P [0.8],
TOP_K [20], REPETITION_PENALTY [1.05], SEED [0]

Chat history format (OpenAI style, but tool-call arguments are dicts):
    {"role": "system" | "user", "content": str}
    {"role": "assistant", "content": str,
     "tool_calls": [{"id": str, "type": "function",
                     "function": {"name": str, "arguments": dict}}]}
    {"role": "tool", "tool_call_id": str, "content": str}
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass


def env(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


BACKEND = env("BACKEND", "hf").lower()
MODEL_NAME = env("MODEL_NAME", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
OPENAI_BASE_URL = env("OPENAI_BASE_URL", "http://localhost:8000/v1").rstrip("/")
OPENAI_API_KEY = env("OPENAI_API_KEY", "EMPTY")
MAX_NEW_TOKENS = int(env("MAX_NEW_TOKENS", "4096"))

# Qwen's recommended sampling settings for its 2507 and Coder instruct models.
TEMPERATURE = float(env("TEMPERATURE", "0.7"))
TOP_P = float(env("TOP_P", "0.8"))
TOP_K = int(env("TOP_K", "20"))
REPETITION_PENALTY = float(env("REPETITION_PENALTY", "1.05"))
SEED = int(env("SEED", "0"))


@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str


@dataclass
class AssistantTurn:
    content: str          # visible text, tool-call markup removed
    tool_calls: list      # [ToolCall]
    parse_errors: list    # [str]
    truncated: bool       # generation hit MAX_NEW_TOKENS


# ---------------------------------------------------------------------
# Tool-call parsing
# ---------------------------------------------------------------------

SPECIAL_TOKEN_RE = re.compile(r"<\|[^|<>]*\|>")          # <|im_end|>, <|endoftext|>, ...
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
XML_FUNCTION_RE = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.DOTALL)
XML_PARAMETER_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Drop <think> blocks emitted by hybrid-thinking models."""
    return text.split("</think>")[-1].replace("<think>", "")


def make_call(name, arguments, call_id=None) -> ToolCall:
    if isinstance(arguments, str):
        # strict=False tolerates raw newlines inside strings, common in code.
        arguments = json.loads(arguments.strip() or "{}", strict=False)
    if not isinstance(arguments, dict):
        raise ValueError("tool-call arguments must be a JSON object")
    return ToolCall(str(name), arguments, call_id or "call_" + uuid.uuid4().hex[:12])


def parse_call(body: str) -> ToolCall:
    body = body.strip()
    if body.startswith("{"):
        # Qwen3 / Hermes: {"name": NAME, "arguments": {...}}
        payload = json.loads(body, strict=False)
        if not isinstance(payload, dict) or "name" not in payload:
            raise ValueError('expected {"name": ..., "arguments": {...}}')
        return make_call(payload["name"], payload.get("arguments", payload.get("parameters", {})))

    # Qwen3-Coder: <function=NAME><parameter=KEY>\nVALUE\n</parameter></function>
    # Values stay strings; the tools convert them where needed.
    match = XML_FUNCTION_RE.search(body)
    if not match:
        raise ValueError("unrecognized tool-call format")
    params = {key.strip(): value for key, value in XML_PARAMETER_RE.findall(match.group(2))}
    return make_call(match.group(1).strip(), params)


def parse_model_text(text: str):
    """Split raw model text into (visible content, tool calls, parse errors)."""
    text = strip_reasoning(SPECIAL_TOKEN_RE.sub("", text))
    bodies = TOOL_CALL_RE.findall(text)
    content = TOOL_CALL_RE.sub("", text)
    errors = []

    # An unclosed block means generation was cut off mid-call.
    # Never execute partial code.
    if "<tool_call>" in content:
        content = content.split("<tool_call>")[0]
        errors.append("a <tool_call> block was not closed; the output was "
                      "probably cut off by the token limit")

    # Qwen3-Coder occasionally omits the <tool_call> wrapper.
    if not bodies and "<function=" in content:
        content, _, rest = content.partition("<function=")
        bodies = ["<function=" + rest]

    calls = []
    for body in bodies:
        try:
            calls.append(parse_call(body))
        except ValueError as exc:   # json.JSONDecodeError is a ValueError
            errors.append(f"could not parse tool call: {exc}")
    return content.strip(), calls, errors


# ---------------------------------------------------------------------
# Backend 1: local Hugging Face transformers
# ---------------------------------------------------------------------

class HFBackend:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            print("WARNING: PyTorch sees no CUDA GPU. A 30B model on CPU needs "
                  "~64 GB RAM and is extremely slow.")
        print(f"Loading {MODEL_NAME}")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, device_map="auto", dtype="auto")
        torch.manual_seed(SEED)

    def generate(self, messages, tools) -> AssistantTurn:
        inputs = self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        prompt_len = inputs["input_ids"].shape[-1]

        if TEMPERATURE > 0:
            sampling = dict(do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
                            top_k=TOP_K, repetition_penalty=REPETITION_PENALTY)
        else:
            # None silences "temperature is set but do_sample=False" warnings.
            sampling = dict(do_sample=False, temperature=None, top_p=None, top_k=None)

        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, **sampling)

        new_tokens = output[0, prompt_len:].tolist()
        print(f"[prompt {prompt_len} tokens, generated {len(new_tokens)} tokens]")
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        content, calls, errors = parse_model_text(raw)
        return AssistantTurn(content, calls, errors, truncated=len(new_tokens) >= MAX_NEW_TOKENS)


# ---------------------------------------------------------------------
# Backend 2: OpenAI-compatible server (vLLM, SGLang, llama.cpp, ...)
# ---------------------------------------------------------------------

def to_openai_messages(messages):
    """The OpenAI API wants tool-call arguments as JSON strings, not dicts."""
    out = []
    for m in messages:
        if m.get("tool_calls"):
            m = {**m, "tool_calls": [
                {**c, "function": {"name": c["function"]["name"],
                                   "arguments": json.dumps(c["function"]["arguments"])}}
                for c in m["tool_calls"]
            ]}
        out.append(m)
    return out


class OpenAIBackend:
    def __init__(self):
        self.url = f"{OPENAI_BASE_URL}/chat/completions"
        print(f"Using OpenAI-compatible server at {OPENAI_BASE_URL}, model {MODEL_NAME}")

    def generate(self, messages, tools) -> AssistantTurn:
        payload = {
            "model": MODEL_NAME,
            "messages": to_openai_messages(messages),
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "seed": SEED,
        }
        if tools:
            payload["tools"] = tools
        if TEMPERATURE > 0:
            # top_k / repetition_penalty are vLLM/SGLang extensions.
            payload.update(top_p=TOP_P, top_k=TOP_K, repetition_penalty=REPETITION_PENALTY)

        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {OPENAI_API_KEY}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=1800) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Server returned HTTP {exc.code}: {body[:2000]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot reach {self.url} ({exc.reason}). "
                               "Is the vLLM server running?") from exc

        choice = data["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        server_calls = message.get("tool_calls") or []
        usage = data.get("usage") or {}
        print(f"[prompt {usage.get('prompt_tokens')} tokens, "
              f"generated {usage.get('completion_tokens')} tokens]")

        if server_calls:
            calls, errors = [], []
            for tc in server_calls:
                fn = tc.get("function", {})
                try:
                    calls.append(make_call(fn.get("name", ""), fn.get("arguments") or "{}", tc.get("id")))
                except ValueError as exc:
                    errors.append(f"could not parse tool-call arguments: {exc}")
            content = strip_reasoning(content).strip()
        else:
            # Without a server-side tool parser, calls arrive as raw text.
            content, calls, errors = parse_model_text(content)

        return AssistantTurn(content, calls, errors, truncated=choice.get("finish_reason") == "length")


def make_backend():
    if BACKEND == "hf":
        return HFBackend()
    if BACKEND == "openai":
        return OpenAIBackend()
    sys.exit(f"Unknown BACKEND={BACKEND!r}; use 'hf' or 'openai'.")
