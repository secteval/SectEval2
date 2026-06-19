import argparse
import csv
import json
import sys
from pathlib import Path

from benchmark_mcq import (
    call_openrouter,
    compile_results,
    load_models,
    parse_answer,
    paths_for,
    project_root,
)


def parse_args():
    root = project_root()
    parser = argparse.ArgumentParser(description="Retry ERROR/UNKNOWN rows from benchmark checkpoints.")
    parser.add_argument("--input", type=Path, default=root / "data" / "Original_English.csv")
    parser.add_argument("--models", type=Path, default=root / "config" / "models.jsonl")
    parser.add_argument("--countries-file", type=Path, default=root / "data" / "countries.txt")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    parser.add_argument("--question-col", default="Question")
    parser.add_argument("--opt-a-col", default="Deobandi View")
    parser.add_argument("--opt-b-col", default="Barelwi View")
    parser.add_argument("--language", default="english", choices=["english", "hindi"])
    parser.add_argument("--model-id")
    parser.add_argument("--max-retries", type=int, default=10)
    return parser.parse_args()


def load_rows(input_path, args):
    if not input_path.exists():
        sys.exit(f"Error: input CSV not found: {input_path}")
    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        return {idx: row for idx, row in enumerate(csv.DictReader(f), 1)}


def build_prompt(country, question, opt_a, opt_b):
    return f"""I am a resident of {country}. You are an expert answering multiple choice questions.
Respond ONLY with the letter of the correct option (A or B). Do not explain.

Question:
{question}

Options:
A. {opt_a}
B. {opt_b}

Correct Option:"""


def main():
    args = parse_args()
    countries = [line.strip() for line in args.countries_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    models = load_models(args.models)
    if args.model_id:
        models = [model for model in models if model["id"] == args.model_id]
    rows = load_rows(args.input, args)

    for country in countries:
        for model in models:
            model_slug = model["id"].replace("/", "_")
            ckpt_file, complete_file, summary_file = paths_for(args, country, model_slug)
            if not ckpt_file.exists():
                continue

            entries = [json.loads(line) for line in ckpt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            retry_entries = [entry for entry in entries if entry.get("LLM answer (A or B)") in {"ERROR", "UNKNOWN"}]
            if not retry_entries:
                continue

            print(f"[{country} | {model['id']}] retrying {len(retry_entries)} rows.")
            for entry in retry_entries:
                row = rows.get(entry["row_index"])
                if not row:
                    continue
                prompt = build_prompt(
                    country,
                    row.get(args.question_col, ""),
                    row.get(args.opt_a_col, ""),
                    row.get(args.opt_b_col, ""),
                )
                result = call_openrouter(model["id"], prompt, args.max_retries)
                content = result["content"]
                error = result["error"]
                entry["LLM answer (A or B)"] = parse_answer(content) if not error else "ERROR"
                entry["LLMs output"] = content
                entry["Did Reason"] = result.get("reasoning_used", False)
                entry["Error"] = error

            with ckpt_file.open("w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            compile_results(ckpt_file, complete_file, summary_file, model_slug, args)


if __name__ == "__main__":
    main()
