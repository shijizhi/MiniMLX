#!/usr/bin/env python3
"""Run a small deterministic Python code-generation pass@1 suite safely."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

SAFE_EXECUTOR = r'''
import ast, json, sys

payload = json.load(sys.stdin)
allowed_nodes = {
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Expr,
    ast.Constant, ast.Name, ast.Load, ast.Store, ast.If, ast.For, ast.While,
    ast.Break, ast.Continue, ast.Pass, ast.Assign, ast.AugAssign, ast.BinOp,
    ast.UnaryOp, ast.Compare, ast.BoolOp, ast.And, ast.Or, ast.Not, ast.USub,
    ast.UAdd, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.Pow, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In,
    ast.NotIn, ast.Is, ast.IsNot, ast.Call, ast.List, ast.Tuple, ast.Dict,
    ast.Set, ast.Subscript, ast.Slice, ast.IfExp, ast.ListComp, ast.SetComp,
    ast.DictComp, ast.GeneratorExp, ast.comprehension, ast.Lambda, ast.JoinedStr,
    ast.FormattedValue, ast.Attribute, ast.keyword,
}
safe_builtins = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "int": int, "len": len, "list": list,
    "max": max, "min": min, "range": range, "reversed": reversed,
    "round": round, "set": set, "sorted": sorted, "str": str,
    "sum": sum, "tuple": tuple, "zip": zip,
}
safe_methods = {
    "append", "pop", "lower", "isalnum", "upper", "join", "sort",
    "values", "startswith",
}
source = payload["source"]
tree = ast.parse(source)
if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
    raise ValueError("output must contain exactly one top-level function")
function = tree.body[0]
if function.name != payload["name"] or function.decorator_list:
    raise ValueError("function name mismatch or decorators are not allowed")
for node in ast.walk(tree):
    if type(node) not in allowed_nodes:
        raise ValueError("disallowed syntax: " + type(node).__name__)
    if isinstance(node, ast.Name) and node.id.startswith("__"):
        raise ValueError("dunder names are not allowed")
    if isinstance(node, ast.Attribute) and node.attr not in safe_methods:
        raise ValueError("disallowed attribute: " + node.attr)
    if isinstance(node, ast.Call):
        fn = node.func
        if isinstance(fn, ast.Name):
            if fn.id not in safe_builtins and fn.id != payload["name"]:
                raise ValueError("disallowed call: " + fn.id)
        elif isinstance(fn, ast.Attribute):
            if fn.attr not in safe_methods:
                raise ValueError("disallowed method call: " + fn.attr)
        else:
            raise ValueError("disallowed call expression")

namespace = {"__builtins__": safe_builtins}
exec(compile(tree, "<generated-answer>", "exec"), namespace, namespace)
fn = namespace[payload["name"]]
results = []
for case in payload["tests"]:
    actual = fn(*case["args"], **case.get("kwargs", {}))
    results.append({"actual": actual, "passed": actual == case["expected"]})
json.dump(results, sys.stdout)
'''


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n?(.*?)```", text, flags=re.I | re.S)
    return (blocks[0] if blocks else text).strip()


def run_sandboxed(name: str, source: str, tests: list[dict[str, object]]) -> list[dict[str, object]]:
    payload = json.dumps({"name": name, "source": source, "tests": tests})
    result = subprocess.run(
        [sys.executable, "-I", "-c", SAFE_EXECUTOR],
        input=payload,
        text=True,
        capture_output=True,
        timeout=4,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"sandbox exited {result.returncode}")
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "Benchmarks" / "qwen25_coder_cases.json",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    from mlx_lm import load, stream_generate
    from minimlx_runtime.runtime import Runtime

    suite = json.loads(args.cases.read_text())
    runtime = Runtime()
    runtime.load(args.model_dir)
    reference_model, reference_tokenizer = load(str(args.model_dir))
    task_results = []
    for task in suite["tasks"]:
        prompt = (
            f"{task['prompt']}\n"
            f"The required function signature is {task['signature']}. "
            "Return only one Python function, with no imports and no explanation."
        )
        generated = runtime.generate(prompt, max_tokens=args.max_tokens)
        formatted = reference_tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        reference_prompt_tokens = list(
            reference_tokenizer.encode(formatted, add_special_tokens=False)
        )
        reference_responses = stream_generate(
            reference_model,
            reference_tokenizer,
            reference_prompt_tokens,
            max_tokens=args.max_tokens,
            prefill_step_size=1024,
        )
        reference_tokens = [int(response.token) for response in reference_responses]
        token_match = (
            generated.prompt_tokens == reference_prompt_tokens
            and generated.generated_tokens == reference_tokens
        )
        source = extract_code(generated.text)
        try:
            tests = run_sandboxed(task["name"], source, task["tests"])
            passed = sum(test["passed"] for test in tests)
            task_results.append({
                "name": task["name"],
                "passed": passed == len(tests),
                "tests_passed": passed,
                "tests_total": len(tests),
                "generated_tokens": len(generated.generated_tokens),
                "ended_with_eos_or_stop": len(generated.generated_tokens) < args.max_tokens,
                "reference_prompt_and_generation_tokens_match": token_match,
                "reference_generated_tokens": len(reference_tokens),
                "source": source,
                "tests": tests,
            })
        except Exception as exc:
            task_results.append({
                "name": task["name"],
                "passed": False,
                "tests_passed": 0,
                "tests_total": len(task["tests"]),
                "generated_tokens": len(generated.generated_tokens),
                "reference_prompt_and_generation_tokens_match": token_match,
                "reference_generated_tokens": len(reference_tokens),
                "error": f"{type(exc).__name__}: {exc}",
                "source": source,
            })
    total_passed = sum(task["tests_passed"] for task in task_results)
    total_tests = sum(task["tests_total"] for task in task_results)
    report = {
        "suite": suite["suite"],
        "model": str(args.model_dir.resolve()),
        "sampling": "greedy (MiniMLX default)",
        "pass_at_1_tasks": sum(task["passed"] for task in task_results),
        "task_count": len(task_results),
        "reference_token_match_tasks": sum(
            task["reference_prompt_and_generation_tokens_match"]
            for task in task_results
        ),
        "unit_tests_passed": total_passed,
        "unit_tests_total": total_tests,
        "tasks": task_results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if (
        report["pass_at_1_tasks"] != report["task_count"]
        or report["reference_token_match_tasks"] != report["task_count"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
