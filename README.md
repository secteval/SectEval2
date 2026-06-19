# SectEval2 Benchmark

SectEval is a binary multiple-choice benchmark for measuring sectarian bias in language model answers. This release evaluates how language models choose between Deobandi and Barelwi positions, following the same benchmark design previously used for Shia and Sunni sectarian views.

Each question presents two possible answers to the model. Option A represents the Deobandi view and Option B represents the Barelwi view, but the prompt does not tell the model which sect is associated with either option. The model is only asked to choose the option it considers correct, allowing the benchmark to measure answer preferences without explicitly naming the sectarian source of each option in the prompt.

The benchmark includes English and Hindi question sets, country-conditioned prompting, retained model results, checkpoints, and generated plots.


## Repository Layout

```text
organised/
├── code/
│   ├── benchmark_mcq.py        # OpenRouter benchmark runner
│   ├── benchmark_retry.py      # retries ERROR/UNKNOWN checkpoint rows
│   └── plot_diverging_bar.py   # summary JSON to diverging bar chart
├── config/
│   └── models.jsonl            # benchmark model list
├── data/
│   ├── Original_English.csv    # 80 questions
│   ├── Original_Hindi.csv      # 80 questions
│   └── countries.txt           # country contexts used in prompts
├── outputs/
│   ├── checkpoints/            # resumable JSONL checkpoints
│   ├── results/                # complete and summary JSON results
│   └── figures/                # generated plots and report CSVs
└── requirements.txt
```

## Data Format

Both CSV files contain these columns:

- `ID`
- `Section`
- `Question`
- `Deobandi View`
- `Barelwi View`
- `Deobandi References & Links`
- `Barelwi References & Links`

The benchmark prompt asks the model to choose only `A` or `B`:

- `A`: `Deobandi View`
- `B`: `Barelwi View`

These labels are used in the dataset and output summaries. They are not disclosed to the model during the binary choice prompt; the model sees only the question text and the two answer options.

Country-conditioned runs use the countries listed in `data/countries.txt`.

## Setup

Use Python 3.10+.

```bash
cd organised
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY="your_openrouter_key"
```

Optional OpenRouter metadata:

```bash
export OPENROUTER_SITE_URL="https://github.com/your-org/your-repo"
export OPENROUTER_APP_NAME="sect-eval"
```

## Run The Benchmark

Run the English benchmark with default paths:

```bash
python3 code/benchmark_mcq.py \
    --language english \
    --input data/Original_English.csv \
    --countries-file data/countries.txt \
    --models config/models.jsonl \
    --output-dir outputs
```

Run the Hindi benchmark:

```bash
python3 code/benchmark_mcq.py \
    --language hindi \
    --input data/Original_Hindi.csv \
    --countries-file data/countries.txt \
    --models config/models.jsonl \
    --output-dir outputs
```

Run a quick smoke test on 10 sampled questions:

```bash
python3 code/benchmark_mcq.py \
    --test-run \
    --model-id openai/gpt-oss-120b
```

Useful controls:

- `--model-id`: run one model from `config/models.jsonl`
- `--workers`: parallel question calls per model
- `--model-workers`: parallel models per country
- `--max-retries`: API retries per question

## Retry Failed Rows

The runner writes checkpoints before compiling final JSON. Retry any rows marked `ERROR` or `UNKNOWN`:

```bash
python3 code/benchmark_retry.py \
    --language english \
    --input data/Original_English.csv \
    --countries-file data/countries.txt \
    --models config/models.jsonl \
    --output-dir outputs
```

## Generate Plots

Create a country diverging bar plot from summary JSON:

```bash
python3 code/plot_diverging_bar.py \
    --summary-dir outputs/results/english/India/summary \
    --output outputs/figures/India_diverging_regenerated.png \
    --title "India"
```

Existing generated figures and report CSVs are stored in `outputs/figures/`.

## Output Files

For a country-conditioned English run, outputs are written to:

```bash
outputs/checkpoints/english/<country>/<model_slug>.jsonl
outputs/results/english/<country>/complete/<model_slug>.json
outputs/results/english/<country>/summary/<model_slug>_summary.json
```

For Hindi runs, the same structure is used under `outputs/checkpoints/hindi/` and `outputs/results/hindi/` for new runs. Previously generated Hindi aggregate outputs are also kept under `outputs/results/hindi/complete/` and `outputs/results/hindi/summary/`.

Each complete result JSON contains question text, options, parsed answer, raw model output, reasoning flag, and error field. Each summary JSON contains total counts and percentages for Option A, Option B, `UNKNOWN`, and `ERROR`.
