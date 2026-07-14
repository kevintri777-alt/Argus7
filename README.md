# ARGUS — AI-Powered Options Research

An AI-driven quantitative research platform for American-style equity options.
Built by Kevin Trivedi & Vivan Jhaveri.

**Live demo:** deployed on Streamlit Community Cloud.

## What it does

ARGUS forecasts each stock's next-month realized volatility, compares that
forecast to what the option market is charging (implied vol), detects
dislocations in the volatility surface (term structure and skew), and turns
them into explained, regime-sized trade tickets.

Direction is unpredictable (measured R² ≈ 0.00), so ARGUS never bets on it.
It bets on bent volatility relationships mean-reverting — traded as calendar
spreads (TERM) and risk reversals (SKEW).

## The four layers

- **L0 Data** — option chains wrangled through a 9-stage, gate-checked pipeline.
- **L1 Pricing** — binomial-tree American pricer + implied-vol solver, validated
  against Black–Scholes.
- **L2 Forecast** — HAR-RV-IV model, chosen by tournament over Transformer /
  LightGBM / Ridge / SHAR, retrained per ticker.
- **L3 Decision** — transparent rules: dislocation z-scores, regime gate,
  conviction sizing, quantile confidence bands.

## Honesty note

Signals are statistically validated (win rate 79%, n=43, t≈5, replicated on 4
tickers) but **pre-cost**: a cost-aware backtest shows bid-ask spreads absorb
the edge at retail execution. ARGUS is a research and education platform in
paper-trading stage — **not investment advice.**

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app auto-discovers tickers from `data/features/*_features.parquet`.
