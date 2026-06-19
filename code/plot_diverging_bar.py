import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_NAMES = {
    "anthropic_claude-opus-4-6": "Claude Opus 4.6",
    "anthropic_claude-sonnet-4-6": "Claude Sonnet 4.6",
    "deepseek_deepseek-v4-pro": "DeepSeek v4 Pro",
    "google_gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "google_gemma-4-31b-it": "Gemma 4 31B",
    "openai_gpt-oss-120b": "GPT-OSS 120B",
    "mistralai_mistral-large-2512": "Mistral Large 2512",
    "x-ai_grok-4.3": "Grok 4.3",
    "qwen_qwen3.7-max": "Qwen 3.7 Max",
    "microsoft_phi-4": "Phi-4",
    "moonshotai_kimi-k2.6": "Kimi K2.6",
    "meta-llama_llama-4-maverick": "Llama 4 Maverick",
    "meta-llama_llama-4-scout": "Llama 4 Scout",
    "nvidia_nemotron-3-super-120b-a12b": "Nemotron 3 Super",
}


def format_model_name(name):
    return MODEL_NAMES.get(name, name.replace("_", " ").replace("-", " ").title())


def generate_diverging_plot(summary_dir: Path, output_name: Path, title: str):
    data_list = []
    for file_path in sorted(summary_dir.glob("*_summary.json")):
        data = json.loads(file_path.read_text(encoding="utf-8"))
        model_raw = data.get("Model", file_path.name.split("_summary")[0])
        a_data = data.get("A-Deobandi View", {})
        b_data = data.get("B-Barelwi View", {})
        a_pct = float(str(a_data.get("Percentage", "0%")).replace("%", ""))
        b_pct = float(str(b_data.get("Percentage", "0%")).replace("%", ""))
        data_list.append(
            {
                "model_name": format_model_name(model_raw),
                "a_pct": a_pct,
                "a_count": int(a_data.get("Count", 0)),
                "b_pct": b_pct,
                "b_count": int(b_data.get("Count", 0)),
                "margin": b_pct - a_pct,
            }
        )

    if not data_list:
        print(f"No summary data found in {summary_dir}")
        return

    data_list = sorted(data_list, key=lambda x: x["margin"])
    fig, ax = plt.subplots(figsize=(12, max(6, len(data_list) * 0.6)), dpi=300)
    y_pos = np.arange(len(data_list))

    ax.barh(y_pos, [-x["a_pct"] for x in data_list], color="#C44E52", height=0.6, label="Option A - Deobandi View")
    ax.barh(y_pos, [x["b_pct"] for x in data_list], color="#4C72B0", height=0.6, label="Option B - Barelwi View")
    ax.axvline(0, color="#2d3748", linewidth=1.2)

    for idx, item in enumerate(data_list):
        if item["a_pct"] > 0:
            ax.text(-item["a_pct"] - 1.5, idx, f"{item['a_pct']:.1f}% ({item['a_count']})", ha="right", va="center", fontsize=8.5)
        if item["b_pct"] > 0:
            ax.text(item["b_pct"] + 1.5, idx, f"{item['b_pct']:.1f}% ({item['b_count']})", ha="left", va="center", fontsize=8.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([x["model_name"] for x in data_list], fontsize=10, weight="bold")
    ax.set_xticks(np.arange(-100, 101, 20))
    ax.set_xticklabels([f"{abs(x)}%" for x in np.arange(-100, 101, 20)])
    ax.set_xlim(-115, 115)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.set_title(title, fontsize=14, weight="bold", pad=40)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    output_name.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_name, dpi=300, facecolor="white", edgecolor="none")
    plt.close()
    print(f"Wrote {output_name}")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Create diverging alignment bar plots.")
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="SectEval")
    args = parser.parse_args()
    summary_dir = args.summary_dir if args.summary_dir.is_absolute() else root / args.summary_dir
    output = args.output if args.output.is_absolute() else root / args.output
    generate_diverging_plot(summary_dir, output, args.title)


if __name__ == "__main__":
    main()
