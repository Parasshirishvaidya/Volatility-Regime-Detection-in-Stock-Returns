## Limitations

1. The Hidden Markov Model assumes that market states follow a first-order Markov process, which may oversimplify real-world financial dynamics.
2. The number of regimes is fixed in advance and may not always align with true market structure.
3. Gaussian emission assumptions may not fully capture extreme market movements and fat-tailed distributions.
4. Model performance is sensitive to feature selection and initialization.
5. The model is trained on historical data and may not generalize perfectly to unseen market conditions.

---

## Future Scope

1. Use Student-t or Mixture Gaussian emissions to better model heavy-tailed returns.
2. Extend to multivariate HMMs across multiple assets or indices.
3. Integrate macroeconomic indicators such as VIX, interest rates, or inflation.
4. Implement online / rolling HMMs for real-time regime detection.
5. Develop a web-based interactive dashboard for regime visualization and monitoring.
