# Hidden Markov Model for Market Regime Detection

This project implements a **Hidden Markov Model (HMM)** to identify and analyze **latent market volatility regimes** in financial time series data. The model captures unobserved market states and their transition dynamics, enabling deeper insights into market behavior across stable and stressed periods.

---

## 📌 Project Overview

Financial markets often switch between different volatility regimes that are not directly observable. This project models such latent states using a **Gaussian Hidden Markov Model**, allowing us to probabilistically infer market regimes based on observed return and volatility features.

The model identifies three hidden market regimes:

- **Low Volatility Bull Market**
- **Low Volatility Sideways Market**
- **High Volatility Stress Market**

By estimating regime transition probabilities and expected durations, the model helps quantify market stability, persistence, and risk.

---

## 📊 Methodology

1. Financial time series data is preprocessed to compute return and volatility-based features.
2. A **Gaussian Hidden Markov Model** is trained on the observed data.
3. The model learns:
   - Hidden market states (regimes)
   - State transition probabilities
   - Regime-specific statistical characteristics
4. Each time step is assigned a most probable hidden regime using the Viterbi algorithm.

---

## 🔁 Regime Transition Analysis

The **transition probability matrix** quantifies the likelihood of moving from one market regime to another. This matrix provides insights into:

- Persistence of low-volatility regimes
- Probability of transitioning into high-stress periods
- Mean-reverting versus unstable market behavior

High diagonal values indicate regime stability, while off-diagonal values represent regime-switching dynamics.

---

## ⏱ Regime Duration Analysis

Expected regime durations are computed using the transition probabilities of the HMM.  
This helps assess how long the market typically remains in a particular regime before transitioning.

Empirical regime durations provide additional validation of model stability and market structure.

---

## 🛠 Tech Stack

- **Python**
- **NumPy**
- **Pandas**
- **hmmlearn**
- **scikit-learn**
- **Matplotlib / Seaborn**

---

## ⚠ Limitations

- The HMM relies on the **Markov assumption**, where future states depend only on the current state.
- The number of hidden regimes must be specified in advance.
- Gaussian emission assumptions may not fully capture heavy-tailed financial returns.
- Model performance can be sensitive to initialization and hyperparameter selection.

---

## 🚀 Future Scope

- Incorporate **Student-t or Mixture Gaussian emissions** for fat-tailed distributions.
- Extend the model to **multivariate HMMs** across multiple asset classes.
- Integrate **macroeconomic indicators** such as VIX, interest rates, or inflation.
- Implement **Bayesian HMMs** for uncertainty quantification.
- Deploy a **real-time regime detection dashboard**.

---

## 🌐 Application & Deployment

A web-based application using **Streamlit or Flask** can be developed to:

- Upload financial datasets
- Train HMM models interactively
- Visualize regime-colored price charts
- Display transition matrices and regime durations

The application can be deployed on **PythonAnywhere** for public access.

---

## 📈 Use Cases

- Market regime classification
- Risk management and portfolio allocation
- Financial research and academic projects
- Quantitative finance and data science portfolios

---

## 📄 License

This project is intended for educational and research purposes.
