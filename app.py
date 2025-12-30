import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotly.express as px

with open("model/hmm_model.pkl","rb") as f:
    hmm_model = pickle.load(f)

with open("model/regime_map.pkl","rb") as f:
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
        df.columns=df.columns.str.lower().str.strip()
        column_map={
            "adj close":"close",
            "closing price":"close",
            "close price":"close"
        }
        df.rename(columns=column_map,inplace=True)
        df["date"]=pd.to_datetime(df["date"])
        df=df.sort_values("date").reset_index(drop=True)

        #Log returns
        df["log_returns"]=np.log(df["close"]/df["close"].shift(1))

        #Absolute Returns
        df["abs_returns"]=df["log_returns"].abs()

        #Rolling Volatility
        df["vol_7"]=df["log_returns"].rolling(window=7).std()
        df["vol_14"]=df["log_returns"].rolling(window=14).std()
        df["vol_30"]=df["log_returns"].rolling(window=30).std()


        #Dropna values
        df=df.dropna().reset_index(drop=True)

        FEATURE_COLS=["log_returns","vol_14"]


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


    state_volatility = {}
    for state in range(hmm_model.n_components):
        state_volatility[state]=df.loc[
            df["regime"]==state,"vol_14"
        ].mean()

    #Sort States by Volatility
    sorted_states=sorted(state_volatility,key=state_volatility.get)

    regime_map={
        sorted_states[0]:"Low volatility (Stable Markets)",
        sorted_states[1]:"Medium volatility (Transition Phase)",
        sorted_states[2]:"High volatility Market (Market Stress)",
    }
    df["regime_label"]=df["regime"].map(regime_map)

    fig=px.scatter(
        df,
        x="date",
        y="close",
        color="regime_label",
        title="Market Regimes Detected using Hidden Markov Model",
        color_discrete_map={
            "Low Volatility (Stable Market)": "green",
            "Low Volatility Sideways(Transition Phase)": "orange",
            "High Volatility (Market Stress)": "red"
        }
    )

    df["regime_label"]=df["regime"].map(regime_map)

    st.markdown("""
                ### 🔍 Interpretation
                # - **Green (Low Volatility)**: Stable market conditions with predictable price movements.
                # - **Orange (Low Volatility Sideways)**: Transitional phases where market sentiment shifts.
                # - **Red (High Volatility)**: Stress regimes associated with uncertainty and large price swings.
                The Hidden Markov Model identifies these regimes without supervision, learning latent market states directly from return dynamics.
                """)
    

    fig,ax=plt.subplots(figsize=(14,6))
    colors={
        "Low volatility (Stable Markets)":"tab:blue",
        "Medium volatility (Transition Phase)":"tab:orange",
        "High volatility Market (Market Stress)":"tab:green"
    }
    regime_map = {
        0: "Low volatility (Stable Markets)",
        1: "Medium volatility (Transition Phase)",
        2: "High volatility Market (Market Stress)"
    }
    st.write("Regime counts:", df["regime"].value_counts())
    st.write("Regime counts:", df["regime_label"].value_counts())
    for regime,color in colors.items():
        subset=df[df["regime_label"]==regime]
        ax.scatter(
            subset["date"],
            subset["close"],
            label=regime,
            s=12,
            color=color
        )
        ax.set_title("Volatility Regimes Detected using HMM")
        ax.set_xlabel("Date")
        ax.set_ylabel("Index level")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)



else:
    st.info("Please upload a CSV file to begin")


