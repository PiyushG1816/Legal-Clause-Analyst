# Legal Clause Analyst

Fine-tuning Llama 3.1 8B for structured legal clause risk analysis using QLoRA and knowledge distillation.

The model takes a legal clause as input and returns a structured JSON analysis with clause type, risk level (Low / Medium / High), summary, risk justification, and negotiation suggestion.

---

## Results

Evaluated on 400 held-out test examples comparing base Llama 3.1 8B Instruct vs. the fine-tuned model:

| Metric | Base Model | Fine-tuned | Delta |
|---|---|---|---|
| Mean Loss | 2.1641 | 0.8697 | -1.2944 |
| JSON Parse Rate | 97.25% | 100.00% | +2.75% |
| Type Exact Match | 1.25% | 53.25% | **+52.00pp** |
| Risk Exact Match | 41.75% | 59.75% | **+18.00pp** |

The base model has near-zero knowledge of LEDGAR/CUAD clause taxonomy (1.25% type accuracy). Fine-tuning teaches the model the domain-specific label vocabulary and enforces consistent structured output.

---

## How It Works

### 1. Data Pipeline

Legal clauses were sourced from two public datasets:

- **LEDGAR** — ~2,000 clauses extracted from SEC filings
- **CUAD** — ~2,000 clauses from the Contract Understanding Atticus Dataset (41 clause types)

Since neither dataset provides structured risk labels in the required format, **Gemini API was used as a teacher model** to generate JSON annotations for each clause — a knowledge distillation approach. The final compiled dataset contains ~4,000 training examples.

```
raw clauses (LEDGAR + CUAD)
        ↓
Gemini API → generates type, summary, risk, reason, suggestion
        ↓
compiled_dataset.jsonl (~4k train) + val.jsonl + test.jsonl
        ↓
QLoRA fine-tuning on Llama 3.1 8B
```

### 2. Fine-tuning

- **Base model:** `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **Method:** QLoRA — 4-bit NF4 quantization + rank-16 LoRA adapters on all 7 projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Trainable parameters:** ~0.7% of total (≈42M of 8B)
- **Hardware:** Google Colab T4 GPU (15GB VRAM)
- **Optimizer:** 8-bit AdamW, cosine LR schedule, lr=2e-4
- **Effective batch size:** 8 (batch size 2 × gradient accumulation 4)
- **Gradient checkpointing:** enabled to fit within T4 memory
- **Best checkpoint:** step 1000 (val loss ~0.87, selected via `load_best_model_at_end=True`)

### 3. Evaluation

Three-layer evaluation comparing base vs. fine-tuned model:

1. **Cross-entropy loss** — full-sequence CE on gold chat, directly comparable to training val loss
2. **Discrete metrics** — JSON parse rate, clause type exact match, risk level exact match
3. **LLM-as-judge** — pairwise comparison via Gemini 1.5 Flash on free-text fields (summary, reason, suggestion) with randomized position assignment to control for position bias

### 4. Gradio App

Interactive demo with side-by-side comparison of base vs. fine-tuned model output. Upload a PDF or TXT contract — the app segments it into clauses using regex pattern matching and runs each clause through both models using PEFT's `disable_adapter()` context manager (one model load, two inference modes, no extra VRAM cost).

---

## Project Structure

```
├── notebooks/
│   ├── CUAD/                          # CUAD dataset exploration
│   ├── LEDGAR/                        # LEDGAR dataset exploration
│   ├── finetune_llama3_qlora.ipynb    # QLoRA training pipeline
│   ├── evaluate_base_vs_finetuned.ipynb  # Evaluation & metrics
│   └── legal_clause_analyst_app.ipynb    # Gradio demo app
│
├── scripts/
│   ├── 1_ingest_ledgar.py             # Extract clauses from LEDGAR
│   ├── 2_distill_gemini.py            # Generate labels via Gemini (LEDGAR)
│   ├── 3_ingest_cuad.py               # Extract clauses from CUAD
│   ├── 4_distill_cuad_gemini.py       # Generate labels via Gemini (CUAD)
│   ├── 5_merge_datasets.py            # Merge LEDGAR + CUAD
│   └── 8_compile_eval_data.py         # Compile train / val / test splits
│
├── .env.example                       # Environment variable template
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.10+
- Google Colab with T4 GPU (for training and inference)
- Gemini API key (for data distillation scripts only)
- HuggingFace account with Llama 3.1 access

### Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```
GEMINI_API_KEY=your_gemini_api_key
HF_TOKEN=your_huggingface_token
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Reproduce the Pipeline

Run scripts in order:

```bash
python scripts/1_ingest_ledgar.py
python scripts/2_distill_gemini.py
python scripts/3_ingest_cuad.py
python scripts/4_distill_cuad_gemini.py
python scripts/5_merge_datasets.py
python scripts/6_compile_eval_data.py
```

Then open `notebooks/finetune_llama3_qlora.ipynb` in Colab and run all cells.

---

## Model Output Format

```json
{
  "type": "Anti-Assignment",
  "summary": "Restricts the borrower from transferring rights or obligations to third parties without lender consent.",
  "risk": "Medium",
  "reason": "Limits operational flexibility but is a standard lender protection clause.",
  "suggestion": "Negotiate a carve-out for assignments to affiliates or subsidiaries without requiring prior consent."
}
```

---

## Stack

`HuggingFace Transformers` · `PEFT` · `BitsAndBytes` · `TRL` · `Gradio` · `pdfplumber` · `Google Gemini API` · `Unsloth`

---

## Datasets

- [LEDGAR](https://huggingface.co/datasets/lex_glue) — via LexGLUE benchmark
- [CUAD](https://huggingface.co/datasets/cuad) — Contract Understanding Atticus Dataset