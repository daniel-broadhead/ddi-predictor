import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.predict import predict_from_names

st.title('Drug-Drug Interaction Predictor')
st.markdown('Enter two drug names to predict the type of interaction between them.')

col1, col2 = st.columns(2)
drug1 = col1.text_input('Drug 1', placeholder='e.g. Warfarin')
drug2 = col2.text_input('Drug 2', placeholder='e.g. Aspirin')

if st.button('Check Interaction') and drug1 and drug2:
    try:
        label, confidence, top_features = predict_from_names(drug1, drug2)
    except ValueError as e:
        st.error(str(e))
    else:
        st.markdown(f'### Predicted interaction')
        st.write(label.replace('#Drug1', drug1).replace('#Drug2', drug2))
        st.metric('Model confidence', f'{confidence:.1%}')

        with st.expander('Top contributing molecular features'):
            for idx, value in top_features:
                st.write(f'Bit {idx}: SHAP value {value:.4f}')