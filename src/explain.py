import numpy as np
import xgboost as xgb


def explain_prediction(drug1_fp, drug2_fp, model, le):
    """Given two Morgan fingerprints, a trained XGBoost model, and
    its LabelEncoder, return the predicted interaction label and
    the top 10 fingerprint bits driving that prediction.

    Uses XGBoost's native pred_contribs (Tree SHAP under the hood)
    rather than shap.TreeExplainer — mathematically equivalent, but
    fast enough for real-time serving at 54 classes. See README
    "Why native contributions" for the full reasoning.
    """
    pair = np.concatenate([drug1_fp, drug2_fp]).reshape(1, -1).astype(np.float32)
    dmat = xgb.DMatrix(pair)

    pred_class = int(model.predict(pair)[0])
    contribs = model.get_booster().predict(dmat, pred_contribs=True)
    sv = contribs[0, pred_class, :-1]

    top_idx = np.argsort(np.abs(sv))[-10:][::-1]
    predicted_label = le.inverse_transform([pred_class])[0]
    return predicted_label, [(int(i), float(sv[i])) for i in top_idx]
