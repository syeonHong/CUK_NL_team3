# CUK_NL_team3

2025-2 Natural Language Processing term project (Team 3, Catholic University of Korea)

---

## 1. Project Overview

This project investigates how a GPT-2 language model internalizes and generalizes a simple word-order rule under three different learning settings:

- **Explicit** fine-tuning (with rule descriptions in prompts)  
- **Implicit** fine-tuning (no explicit rule, only sentence exposure)  
- **In-context learning (ICL)** with 0–4 shot examples

We compare:

- English (SVO) vs. an artificial SOV language (ArLa)  
- Explicit vs. implicit learning conditions  
- Fine-tuning vs. ICL performance

Main experiments:

- **E1 — Fine-tuning Efficiency**  
  - dev PPL curves under explicit / implicit / prompt variations
- **E2 — Grammaticality Judgment**  
  - BLiMP-style PLL ranking, 5-choice grammaticality, surprisal analysis
- **E3 — In-context Learning (0–4 shot)**  
  - few-shot ICL without parameter updates on ArLa

---

## 2. Repository Structure

```bash
main/
│
├── config/                 # YAML configs for experiments
├── data/                   # Dataset zips and split scripts
├── src/
│   ├── artlang_generator.py   # Artificial language (ArLa, SOV) generator
│   ├── build_datasets.py      # Build ENG/SOV explicit/implicit datasets
│   ├── create_pairs.py        # Create OK/Violation minimal pairs
│   ├── dataloader.py          # PyTorch Dataset & DataLoader
│   ├── model.py               # GPT-2 wrapper model & loss
│   ├── prompts.py             # Explicit / implicit prompt templates
│   ├── run_ft.py              # E1: fine-tuning pipeline
│   ├── run_eval_e2.py         # E2: BLiMP, 5-choice, surprisal eval
│   ├── run_icl.py             # E3: 0/1/2/4-shot ICL
│   └── utils.py               # Shared utility functions
│
├── scripts/                # Plotting & result aggregation scripts
└── utils/                  # Metrics & helpers
```

## 3. File Description

### 3.1 Dataset & Preprocessing

- `build_datasets.py`  
  - Builds English (SVO) and ArLa (SOV) datasets.  
  - Creates explicit / implicit versions by attaching or omitting rule prompts.

- `create_pairs.py`  
  - Generates OK vs. Violation minimal pairs for grammaticality judgment.

- `split.py`  
  - Splits data into train / dev / test (and OOD, if used).

- `artlang_generator.py`  
  - Generator for the artificial SOV language (ArLa).  
  - Controls word order templates, vocabulary, and noise level.


### 3.2 Model & DataLoader

- `model.py`  
  - GPT-2 language model wrapper.  
  - Implements forward pass and LM loss for fine-tuning / evaluation.

- `prompts.py`  
  - Explicit-card, explicit-explanation, implicit prompt templates.  
  - Central place to modify rule descriptions and input formats.

- `dataloader.py`  
  - Tokenization and PyTorch `Dataset` / `DataLoader` implementation.  
  - Handles explicit / implicit formatting at batch level.


### 3.3 Experiment Pipelines

- `run_ft.py`  
  - E1: fine-tuning under explicit / implicit and ENG / ArLa conditions.  
  - Saves checkpoints and training logs (loss / PPL).

- `run_eval_e2.py`  
  - E2: grammaticality judgment evaluation.  
  - BLiMP-style PLL ranking, 5-choice evaluation, surprisal extraction.

- `run_icl.py`  
  - E3: 0/1/2/4-shot ICL experiments on ArLa.  
  - Uses PMI-style scoring over natural-language labels.


### 3.4 Visualization & Utilities

- `plot_learning_curves.py`  
  - Plots train / dev PPL and loss curves for E1.

- `plot_surprisal.py`  
  - Visualizes token-level surprisal peaks for selected sentences.

- `scripts/`  
  - Additional plotting and result-aggregation scripts (tables, figures).

- `utils/`  
  - Metrics (e.g., AUC, ECE, ΔPLL) and general helper functions.

---

## 4. Experiment Structure

### E1 — Fine-tuning Efficiency

- Goal: compare dev PPL and convergence across
  - explicit vs. implicit prompts,
  - English (SVO) vs. ArLa (SOV),
  - different explicit prompt styles.
- Output:
  - loss / PPL logs,
  - learning curves.

### E2 — Grammaticality Judgment

- Subtasks:
  - BLiMP-style PLL ranking (OK vs. Violation),
  - 5-choice grammaticality on ArLa,
  - surprisal peak analysis.
- Output:
  - accuracy, AUC, ΔPLL,
  - calibration metrics (e.g., ECE),
  - surprisal plots.

### E3 — In-Context Learning (0–4 shot, ArLa)

- Goal: test whether GPT-2 can infer the SOV rule from a few in-context examples without parameter updates.
- Models:
  - base GPT-2,
  - ArLa explicit fine-tuned model,
  - ArLa implicit fine-tuned model.
- Output:
  - shot-wise accuracy curves,
  - error patterns across ICL prompts.

---

## 5. Component Flow

1. Dataset construction  
   - `artlang_generator.py` → `build_datasets.py` → `create_pairs.py` → `split.py`
2. Model & DataLoader  
   - `model.py` + `prompts.py` + `dataloader.py`
3. Experiments  
   - E1: `run_ft.py`  
   - E2: `run_eval_e2.py`  
   - E3: `run_icl.py`
4. Analysis & visualization  
   - `scripts/`, `plot_learning_curves.py`, `plot_surprisal.py`

---

## 6. Team Members (Team 3, CUK)

- Dataset / ArLa generation: 류재형  
- E1 / E2 (ArLa) : 이유진  
- E1 / E2 (English): 최한종  
- E3 (ArLa ICL): 장주은  
- Integration / documentation / repo structure: 홍승연

