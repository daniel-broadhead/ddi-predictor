# Streamlit-only image — mirrors what's deployed on Hugging Face Spaces.
# Not used for the actual deployment (Spaces builds its own container via
# the Streamlit SDK), but included to demonstrate a portable, reproducible
# build path independent of any specific hosting platform.

FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (separately from app code) so Docker can
# cache this layer — rebuilds after code-only changes skip reinstalling
# packages, which matters given rdkit/scikit-learn/xgboost aren't tiny.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# App code
COPY src/ ./src/
COPY app/streamlit_app.py ./app/streamlit_app.py

# Model artifacts — only the 3 files predict.py actually loads at runtime,
# not the evaluation charts (confusion matrices, SHAP summary) that live
# alongside them in models/ but are dev/README artifacts, not serving deps.
COPY models/ddi_xgb.joblib ./models/ddi_xgb.joblib
COPY models/label_encoder.joblib ./models/label_encoder.joblib
COPY models/interaction_translations.json ./models/interaction_translations.json

EXPOSE 8501

# --server.address=0.0.0.0 is required here even though local dev never
# needed it — Streamlit defaults to binding localhost only, which is
# unreachable from outside the container even with `docker run -p`.
# --server.headless=true skips the browser auto-launch (no display exists
# inside a container anyway, so this avoids the same failure mode that
# hit local Windows dev, for an unrelated reason).
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]