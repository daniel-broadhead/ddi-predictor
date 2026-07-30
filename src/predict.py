import os
import json
import joblib
import numpy as np

from src.features import smiles_to_fp
from src.explain import explain_prediction

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
_MODEL_PATH = os.path.join(_PROJECT_ROOT, 'models', 'ddi_xgb.joblib')
_LABEL_ENCODER_PATH = os.path.join(_PROJECT_ROOT, 'models', 'label_encoder.joblib')
_DRUG_LOOKUP_PATH = os.path.join(_SRC_DIR, 'drug_lookup.json')

_model = None
_label_encoder = None
_drug_lookup = None


def _load_artifacts():
    global _model, _label_encoder, _drug_lookup
    if _model is None:
        _model = joblib.load(_MODEL_PATH)
    if _label_encoder is None:
        _label_encoder = joblib.load(_LABEL_ENCODER_PATH)
    if _drug_lookup is None:
        with open(_DRUG_LOOKUP_PATH) as f:
            _drug_lookup = json.load(f)
    return _model, _label_encoder, _drug_lookup


def predict_from_smiles(smiles1, smiles2):
    """Given two SMILES strings, return the predicted interaction
    label, its confidence, and the top contributing fingerprint features."""
    model, le, _ = _load_artifacts()
    fp1 = smiles_to_fp(smiles1)
    fp2 = smiles_to_fp(smiles2)
    label, top_features = explain_prediction(fp1, fp2, model, le)

    pair = np.concatenate([fp1, fp2]).reshape(1, -1).astype(np.float32)
    confidence = float(model.predict_proba(pair)[0].max())

    return label, confidence, top_features


def predict_from_names(drug1_name, drug2_name):
    """Given two drug names, resolve them to SMILES via the local
    lookup, then predict. Raises ValueError if either name isn't
    found — the caller (API/app) decides how to surface that."""
    _, _, drug_lookup = _load_artifacts()

    smiles1 = drug_lookup.get(drug1_name.lower())
    smiles2 = drug_lookup.get(drug2_name.lower())

    if smiles1 is None:
        raise ValueError(f"'{drug1_name}' not found in drug lookup")
    if smiles2 is None:
        raise ValueError(f"'{drug2_name}' not found in drug lookup")

    return predict_from_smiles(smiles1, smiles2)

