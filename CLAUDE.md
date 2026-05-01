# Project Notes — Myopia Progression Prediction

> Guidance for Claude Code (or any AI assistant) working on this repository. Read this before making changes.

## Project Identity

This is a research codebase supporting the MPhil thesis of **Syed Ahmad Hassan** (2024-MPhil-OP-037). Outputs will be used in academic publications and a thesis chapter, so quality, reproducibility, and traceability matter more than feature velocity.

## Critical Constraints

1. **Never commit patient data**. The CSV files in `data/raw/` are anonymised but should not be pushed to public remotes.
2. **Never break reproducibility**. `RANDOM_STATE = 42` in `src/config.py` controls every stochastic operation. Do not reset, override, or randomise it without changing the README and re-running the pipeline.
3. **Never apply SMOTE before splitting**. SMOTE must run only on the training set, after the stratified split, otherwise synthetic samples leak into validation/test.
4. **Never fit the scaler on the full dataset**. `StandardScaler.fit_transform(X_train)` only; transform val/test with the fitted scaler.
5. **Best-model artifacts must be regenerated together.** `best_model.joblib`, `scaler.joblib`, and `feature_columns.json` in `app/` must come from the same pipeline run, otherwise inference is silently incorrect.

## How the Pipeline Hangs Together

```
data/raw/combined_clinical_data_and_labels.csv
        │
        │  src/data/loader.py            (validate schema)
        ▼
        │  src/data/feature_engineering.py (encode, derive 32 features)
        ▼
        │  src/data/preprocessor.py       (clean, split, scale)
        ▼
        │  src/data/augmentation.py        (SMOTE on train only)
        ▼
        │  src/models/trainer.py           (5-fold CV + final fit + test eval)
        ▼
        │  src/models/evaluator.py         (9 evaluation plots)
        │  src/models/explainer.py         (SHAP explanations)
        ▼
        outputs/{models, figures, reports}/
        app/{best_model.joblib, scaler.joblib, ...}   (Streamlit consumes these)
```

The orchestrator is `run_pipeline.py` at the repository root.

## Running the Pipeline

| Command | Purpose | Time |
|---|---|---|
| `python run_pipeline.py --skip-shap --skip-diagrams` | Standard run with all training and evaluation | ~3 min |
| `python run_pipeline.py` | Full pipeline including SHAP and diagrams | ~6 min |
| `python -m src.utils.generate_research_document` | Generate Word document | ~5 s |
| `streamlit run app/streamlit_app.py` | Launch the clinical app | instant |

## Coding Conventions

- **PEP 8** with type annotations on all public functions
- **Pathlib** for every file path (no string concatenation)
- **Docstrings** on every module and public function
- **No print debugging** in committed code; remove or convert to `logging`
- **Centralised configuration** — all paths, random states, and styling live in `src/config.py`
- **Publication-quality fonts** — defined once in `src.config.PUB_RC`. Increase here, not in individual plot functions.

## Visualisation Standards

All matplotlib/seaborn plots **must** use `plt.rc_context(PUB_RC)`. The current settings produce:

- Title: 17 pt, bold
- Axis labels: 15 pt, bold
- Tick labels: 13 pt
- Legend: 12 pt
- Suptitle: 19 pt, bold
- Serif font (Times New Roman / DejaVu Serif fallback)

Streamlit plots use Plotly with the same colour palette (`COLOR_NEG = #1565C0`, `COLOR_POS = #E65100`).

## Streamlit Theme Safety

The Streamlit app **must** remain readable in both light and dark modes. The current implementation:

1. Uses Streamlit's CSS variables (e.g. `[data-testid="stMetricValue"]`) where possible
2. Wraps theme-dependent colours behind `@media (prefers-color-scheme)` rules
3. Forces white text only inside coloured-background result boxes (always readable on solid colour)
4. Never hard-codes white text on a transparent background

If you change the CSS, test in **both** light and dark mode before pushing.

## Tests Worth Running After Any Change

```bash
# 1. Module imports clean
python -c "import sys; sys.path.insert(0, '.'); from src.config import *; from src.data.loader import load_raw; print('imports OK')"

# 2. Smoke test the data pipeline only
python -c "
import sys; sys.path.insert(0, '.')
from src.data.loader import load_raw
from src.data.feature_engineering import engineer_all_features
from src.data.preprocessor import full_preprocessing_pipeline
from src.config import RAW_NUMERIC_COLS, ALL_FEATURE_COLS
df = engineer_all_features(load_raw())
prep = full_preprocessing_pipeline(df, RAW_NUMERIC_COLS, ALL_FEATURE_COLS)
print('shape:', df.shape, 'train:', prep['X_train_scaled'].shape)
"

# 3. Verify Streamlit artifacts load
python -c "
import joblib, json
m = joblib.load('app/best_model.joblib')
s = joblib.load('app/scaler.joblib')
fc = json.load(open('app/feature_columns.json'))
print('Model:', type(m).__name__, '| Features:', len(fc))
"
```

## Common Pitfalls Encountered

| Symptom | Cause | Fix |
|---|---|---|
| `AttributeError: 'NoneType' object has no attribute 'memmap'` during stacking ensemble training | joblib + Windows multiprocessing edge case | `n_jobs=1` in `StackingClassifier`. Already fixed in `src/models/definitions.py`. |
| `ValueError: The palette dictionary is missing keys: {'0', '1'}` | Newer seaborn coerces palette dict keys to strings | Pass a list (`palette=COLORS`) instead of `palette=PALETTE` for `x="label"` plots |
| `MemoryError: bad allocation` when saving large EDA grids at 300 DPI | Figure exceeds RAM at full publication DPI | Use `DPI_SCREEN` (150) for grid plots; reserve 300 DPI only for `outputs/figures/publication/` |
| `feature_columns.json` and saved `best_model.joblib` mismatch | Manual partial pipeline run | Always re-run `python run_pipeline.py` end-to-end after any feature change |

## Updating the Research Document

The Word document at `outputs/document/Myopia_Progression_Research_Document.docx` is **regenerated from the latest pipeline outputs** every time you run `python -m src.utils.generate_research_document`. Edits to the `.docx` directly will be lost. To change wording, edit `src/utils/generate_research_document.py`.

## What "Done" Means for a Change

A code change is complete only when:

- [ ] `python run_pipeline.py --skip-shap --skip-diagrams` runs end-to-end without errors
- [ ] `outputs/reports/model_results_summary.csv` is updated
- [ ] `app/` artifacts are consistent with the new `outputs/models/` artifacts
- [ ] Streamlit app launches without exceptions and the gauge updates on prediction
- [ ] If figures changed, the README headline numbers and the `.docx` document are regenerated

## Useful Files at a Glance

| File | Purpose |
|---|---|
| `src/config.py` | Single source of truth for paths, constants, fonts |
| `run_pipeline.py` | 12-phase orchestrator |
| `app/streamlit_app.py` | Theme-safe clinical UI |
| `src/utils/generate_research_document.py` | Builds the .docx from latest results |
| `outputs/reports/model_results_summary.csv` | Authoritative results table |
| `README.md` | Public documentation |
| `CLAUDE.md` | This file |
