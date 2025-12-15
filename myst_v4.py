# This file must be run in the command line with: streamlit run myst_v4.py
# Previously, Streamlit has to be installed with pip install streamlit

import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# must copy custom estimator class here so model pipeline functions correctly
class PdaysTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.median_pdays = None
    
    def fit(self, X, y=None):
        known_pdays = X.loc[X['pdays'] != -1, 'pdays']
        self.median_pdays = known_pdays.median() if len(known_pdays) > 0 else 0
        return self
    
    def transform(self, X):
        X = X.copy()
        X['prev_contacted'] = (X['pdays'] != -1).astype(int)
        X['pdays_duration'] = X['pdays'].replace(-1, self.median_pdays)
        X.drop('pdays', axis=1, inplace=True)
        return X

st.set_page_config(page_title="Bank Product Subscription Prediction", page_icon="🏦", layout="centered")
st.title("Making Predictions on New Data by the Deployed Model on the Baking Dataset")

@st.cache_resource
def load_pack():
    return load("pack_for_streamlit.joblib")

pack = load_pack()
model = pack["model"]
num_cols = pack["num_cols"]
cat_cols = pack["cat_cols"]
num_summary = pack["num_summary"]
cat_options = pack["cat_options"]
classes_ = pack["classes_"]
acc = pack.get("accuracy", None)

if acc is not None:
    st.caption(f"Accuracy en test (entrenado): **{acc:.3f}**")

st.markdown("Input values for the features and press **Predict**:")

with st.form("form"):
    st.subheader("Numerical features")

    num_inputs = {}
    for c in num_cols:
        p1 = num_summary[c]["p1"]
        p99 = num_summary[c]["p99"]
        med = num_summary[c]["median"]

        # If empty in numerical features, it is considered a missing value
        raw = st.text_input(
            c,
            value=str(med),
            placeholder="(leave empty for missing value)"
        )

        # Convert empty to missing (np.nan)
        if raw.strip() == "":
            num_inputs[c] = np.nan
        else:
            try:
                num_inputs[c] = float(raw)
            except:
                num_inputs[c] = np.nan  # in case the value is non-numerical

    st.subheader("Categorical features")

    cat_inputs = {}
    for c in cat_cols:
        opts = cat_options[c]
        default = opts[0] if opts else ""
        cat_inputs[c] = st.selectbox(c, opts, index=0)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Construct the data point with values given for each feature
    data = {**num_inputs, **cat_inputs}
    X_one = pd.DataFrame([data], columns=num_cols + cat_cols)

    try:
        proba = model.predict_proba(X_one)[0]
        y_pred = classes_[int(np.argmax(proba))]
    except Exception as e:
        st.error(f"Error the prediction: {e}")
        st.stop()

    st.success(f"Prediction: **{y_pred}**")

    st.write("Probabilities per class:")
    st.write({str(cls): f"{100*p:.1f}%" for cls, p in zip(classes_, proba)})
