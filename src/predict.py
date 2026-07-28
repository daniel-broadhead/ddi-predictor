import os
import joblib

from src.features import smiles_to_fp
from src.explain import explain_prediction


_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
_MODEL_PATH = os.path.join(_PROJECT_ROOT, 'models', 'ddi_xgb.joblib')
_LABEL_ENCODER_PATH = os.path.join(_PROJECT_ROOT, 'models', 'label_encoder.joblib')

_model = None
_label_encoder = None


def _load_artifacts():
    """Load model + label encoder once, then reuse — avoids re-reading
    from disk on every prediction call once the app is running."""
    global _model, _label_encoder
    if _model is None:
        _model = joblib.load(_MODEL_PATH)
    if _label_encoder is None:
        _label_encoder = joblib.load(_LABEL_ENCODER_PATH)
    return _model, _label_encoder


def predict_from_smiles(smiles1, smiles2):
    """Given two SMILES strings, return the predicted interaction
    label and its top contributing fingerprint features."""
    model, le = _load_artifacts()
    fp1 = smiles_to_fp(smiles1)
    fp2 = smiles_to_fp(smiles2)
    label, top_features = explain_prediction(fp1, fp2, model, le)
    return label, top_features