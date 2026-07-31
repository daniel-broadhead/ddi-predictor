# DDI Predictor

Predicts the specific type of interaction between two drugs — not just whether one exists — using molecular structure and gradient-boosted trees, with per-prediction explainability.

![Confusion matrix, top 15 interaction types](models/confusion_matrix_top15.png)

## What it does

Given two drug names, the model predicts which of 54 clinically distinct interaction types is most likely between them (e.g. *"may increase the anticoagulant activities of"*), along with the specific molecular features that drove that prediction.

Try it: [live demo link — add once deployed to Hugging Face Spaces]

## Why multiclass, not binary

Most DDI classifier projects predict a simple yes/no interaction flag. This one predicts the interaction *mechanism* instead — a more clinically useful signal, and a better showcase of both ML engineering and domain understanding. The dataset (TDC's DrugBank benchmark, 191,808 pairs) is natively labeled with 86 distinct interaction types; after merging classes with fewer than 100 samples into a single `Other` bucket, this became a 54-class problem.

## Architecture
Drug names -> SMILES Lookup -> Morgan fingerprints(RDKit) -> XGBoost(54 classes) -> native SHAP-style explanation -> Result

- **Features**: Morgan fingerprints (radius 2, 2048 bits per drug, concatenated to 4096 dimensions)
- **Model**: XGBoost, `multi:softprob`, `tree_method='hist'`, early stopping, balanced sample weighting
- **Explainability**: XGBoost's native `pred_contribs` (Tree SHAP under the hood) rather than the `shap` library's `TreeExplainer` — at 54 classes, `TreeExplainer` was too slow for real-time serving; native contributions are mathematically equivalent but fast enough for a live app
- **Serving**: FastAPI backend + Streamlit frontend, sharing one prediction module (`src/predict.py`) so there's no duplicated logic between the two interfaces

## Results


| Metric | Score |
|---|---|
| Macro F1 | 0.8685022100635987 |
| Micro F1 | 0.8492659848326723 |

Macro F1 is the headline metric — it weights every class equally regardless of size, which matters given the dataset's long tail (the three largest classes account for ~62% of all pairs). Full per-class breakdown and confusion matrix in `models/confusion_matrix.png`.

## Running locally

```bash
git clone https://github.com/daniel-broadhead/ddi-predictor
cd ddi-predictor
conda create -n ddi-env python=3.10
conda activate ddi-env
conda install -c conda-forge rdkit
pip install -r requirements.txt
```

```bash
# FastAPI backend
uvicorn app.api:app --reload

# Streamlit frontend (separate terminal)
streamlit run app/streamlit_app.py
```

**Note:** `requirements.txt` pins `numpy<2` (RDKit's conda build predates NumPy 2.x's breaking ABI change) and `PyTDC==0.4.1` (newer releases pull in an unrelated single-cell genomics dependency, `tiledbsoma`, that fails to build on Windows).

## Project structure
```
DDI_Project/
├── app/
│   ├── api.py            FastAPI backend
│   └── streamlit_app.py  Streamlit frontend
├── src/
│   ├── features.py       SMILES → Morgan fingerprint conversion
│   ├── model.py          Training and evaluation functions
│   ├── explain.py        Per-prediction explanation
│   └── predict.py        Shared prediction pipeline
├── models/                Trained model, label encoder, translations
└── notebooks/             Exploration, training, evaluation
```

## Limitations & future work

- **The `Other` class isn't a real pharmacological category.** It's a statistical necessity — 33 interaction types with fewer than 100 training samples each, merged so the model and evaluation would be viable. Some of the individually-rare types folded into `Other` may be clinically significant despite being data-poor. **This tool does not replace professional medical or pharmacist guidance, especially for any interaction the model places in the `Other` category.**
- **Drug name lookup is currently limited to 9 well-known drugs**, resolved via PubChem as a temporary stand-in. A full 1,706-drug lookup (joining DrugBank's vocabulary export against this dataset's SMILES) is in progress, pending DrugBank's academic download access coming back online.
- Future direction: map top contributing fingerprint bits back to actual molecular substructures (partially built in `notebooks/`, not yet wired into the live app).

## License

MIT — see `LICENSE`.