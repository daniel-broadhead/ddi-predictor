import os
import json
import numpy as np
import joblib

from src.features import smiles_to_fp
from src.explain import explain_prediction

# Resolve paths relative to THIS FILE's location, not the caller's cwd.
# predict.py gets imported from the notebook, from app/api.py (uvicorn),
# and from app/streamlit_app.py (each of which sets cwd differently) —
# __file__-based paths make this work correctly regardless of caller.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
_MODEL_PATH = os.path.join(_PROJECT_ROOT, 'models', 'ddi_xgb.joblib')
_LABEL_ENCODER_PATH = os.path.join(_PROJECT_ROOT, 'models', 'label_encoder.joblib')
_DRUG_LOOKUP_PATH = os.path.join(_SRC_DIR, 'drug_lookup.json')
_TRANSLATIONS_PATH = os.path.join(_PROJECT_ROOT, 'models', 'interaction_translations.json')

_model = None
_label_encoder = None
_drug_lookup = None
_translations = None


def _load_artifacts():
    """Load model, label encoder, drug lookup, and interaction-type
    translations once, then reuse — avoids re-reading from disk on
    every prediction call once the app is running."""
    global _model, _label_encoder, _drug_lookup, _translations
    if _model is None:
        _model = joblib.load(_MODEL_PATH)
    if _label_encoder is None:
        _label_encoder = joblib.load(_LABEL_ENCODER_PATH)
    if _drug_lookup is None:
        with open(_DRUG_LOOKUP_PATH) as f:
            _drug_lookup = json.load(f)
    if _translations is None:
        with open(_TRANSLATIONS_PATH) as f:
            _translations = json.load(f)
    return _model, _label_encoder, _drug_lookup, _translations


def predict_from_smiles(smiles1, smiles2):
    """Given two SMILES strings, return the translated interaction
    label, model confidence, and top contributing fingerprint features."""
    model, le, _, translations = _load_artifacts()
    fp1 = smiles_to_fp(smiles1)
    fp2 = smiles_to_fp(smiles2)
    raw_label, top_features = explain_prediction(fp1, fp2, model, le)

    pair = np.concatenate([fp1, fp2]).reshape(1, -1).astype(np.float32)
    confidence = float(model.predict_proba(pair)[0].max())

    translated = translations.get(raw_label, f"Unknown interaction type ({raw_label})")
    return translated, confidence, top_features


def predict_from_names(drug1_name, drug2_name):
    """Given two drug names, resolve them to SMILES via the local
    lookup, predict, and substitute the real names into the
    translated interaction sentence (which contains #Drug1/#Drug2
    placeholders from the source label text). Raises ValueError if
    either name isn't found in the lookup — the caller (API/app)
    decides how to surface that to the user."""
    _, _, drug_lookup, _ = _load_artifacts()

    smiles1 = drug_lookup.get(drug1_name.lower())
    smiles2 = drug_lookup.get(drug2_name.lower())

    if smiles1 is None:
        raise ValueError(f"'{drug1_name}' not found in drug lookup")
    if smiles2 is None:
        raise ValueError(f"'{drug2_name}' not found in drug lookup")

    label, confidence, top_features = predict_from_smiles(smiles1, smiles2)
    label = label.replace('#Drug1', drug1_name).replace('#Drug2', drug2_name)
    return label, confidence, top_features
