import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

with open("\model\hmm_model.pkl","rb") as f:
    hmm_model = pickle.load(f)

with open("model\regime_map.pkl","rb") as f:
    regime_map = pickle.load(f)

## App Title

st.set_page_config(page_title="Market Regime Detection",layout="wide")
st .title("Market Volatility Detection using HMM")

st.markdown(
    """
This application detects hidden market volatility regimes using a 
    **Hidden Markov Model (HMM)** trained on financial return data.
"""
)

uploaded_file=st.file_uploader("Upload a CSV file containing Data and Closing Price",
                               type=["csv"]
                               )

def preprocess_data(df):
    df=df.copy()
    df["date"]=pd.to_datetime(df["date"])
    df=df.sort_values("date")
    df["log_returns"]=np.log(df["close"]/df["close"].shift(1))
    df.dropna(inplace=True)
    return df

##Main logic

if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)

    if not {"date","close"}.issubset(df.columns):
        st.error("CSV must contain 'date' and 'close' columns")
    else:
        df=preprocess_data(df)

        X=df["log_returns"].values.reshape(-1,1)

        df["regime"]=hmm_model.predict(X)

        df["regime_label"]=df["regime"].map(regime_map)


        ##Plot Regimes
        st.subheader("📈 Price Chart with Detected Regimes")
        fig, ax = plt.subplots(figsize=(14, 6))

        for regime,label in regime_map.items():
            subset=df[df["regime"]==regime]
            ax.scatter(
                subset["date"],
                subset["close"],
                label=label,
                s=10
            )

        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.set_title("Market Regimes detected using HMM")
        ax.legend()

        st.pyplot(fig)

        stats=(
            df.groupby("regime_label")["log_returns"]
            .agg(["mean","std","count"])
            .reset_index()
        )
        st.dataframe(stats)

else:
    st.info("Please upload a CSV file to begin")


