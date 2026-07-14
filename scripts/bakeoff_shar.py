"""Honest walk-forward bake-off: HAR-RV-IV incumbent vs SHAR / log-vol variants.

Promote SHAR to production ONLY if it beats HAR-RV-IV out of sample.
Metrics: walk-forward correlation (same TimeSeriesSplit(4) the app uses),
RMSE, and QLIKE (the volatility-specific loss).
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit

HAR = ["rv_5", "rv_21", "rv_63", "atm_iv"]
SHAR = ["rsv_neg_5", "rsv_pos_5", "rv_21", "rv_63", "atm_iv"]


def add_shar(full):
    """Downside/upside realized semivariance over a 5-day window from daily ret.
    Convention matches rv_*: annualized vol = sqrt(252 * mean(ret^2))."""
    r = full["ret"]
    neg = (r.clip(upper=0) ** 2)
    pos = (r.clip(lower=0) ** 2)
    full["rsv_neg_5"] = np.sqrt(252 * neg.rolling(5).mean())
    full["rsv_pos_5"] = np.sqrt(252 * pos.rolling(5).mean())
    return full


def qlike(actual, pred):
    """QLIKE on variance (vol^2). Lower is better. Robust to vol-of-vol."""
    a = np.clip(actual, 1e-6, None) ** 2
    p = np.clip(pred, 1e-6, None) ** 2
    return float(np.mean(a / p - np.log(a / p) - 1))


def wf_eval(df, feats, log_target=False, model="ols"):
    X = df[feats].values
    y = df["fwd_rv_21"].values
    corrs, preds, acts = [], [], []
    for tr, te in TimeSeriesSplit(4).split(X):
        yt = np.log(y[tr]) if log_target else y[tr]
        if model == "ridge":
            m = RidgeCV(alphas=np.logspace(-3, 2, 20)).fit(X[tr], yt)
        else:
            m = LinearRegression().fit(X[tr], yt)
        p = m.predict(X[te])
        if log_target:
            p = np.exp(p)
        corrs.append(pearsonr(p, y[te])[0])
        preds.append(p)
        acts.append(y[te])
    p_all = np.concatenate(preds)
    a_all = np.concatenate(acts)
    return {
        "corr": float(np.mean(corrs)),
        "rmse": float(np.sqrt(np.mean((p_all - a_all) ** 2))),
        "qlike": qlike(a_all, p_all),
    }


def main():
    files = sorted(glob.glob("data/features/*_features.parquet"))
    configs = [
        ("HAR-RV-IV  (incumbent)", HAR, False, "ols"),
        ("SHAR-RV-IV (level)     ", SHAR, False, "ols"),
        ("SHAR-RV-IV (log)       ", SHAR, True, "ols"),
        ("SHAR-RV-IV (log+ridge) ", SHAR, True, "ridge"),
    ]
    for f in files:
        tk = Path(f).stem.split("_")[0].upper()
        full = add_shar(pd.read_parquet(f).sort_values("date").reset_index(drop=True))
        df = full[full.has_label].dropna(subset=SHAR + ["fwd_rv_21"]).reset_index(drop=True)
        # sanity: semivariance reconstructs total variance
        recon = np.corrcoef(df.rsv_neg_5**2 + df.rsv_pos_5**2, df.rv_5**2)[0, 1]
        print(f"\n=== {tk}  (n={len(df)})   semivar->rv_5^2 recon corr={recon:.3f} ===")
        print(f"{'model':<24}  {'wf_corr':>8}  {'rmse':>7}  {'qlike':>7}")
        base = None
        for name, feats, logt, mdl in configs:
            r = wf_eval(df, feats, logt, mdl)
            if base is None:
                base = r
            flag = ""
            if name.strip() != "HAR-RV-IV  (incumbent)":
                better = r["corr"] > base["corr"] and r["qlike"] <= base["qlike"] * 1.02
                flag = "  <-- beats incumbent" if better else ""
            print(f"{name:<24}  {r['corr']:>+8.3f}  {r['rmse']:>7.4f}  "
                  f"{r['qlike']:>7.4f}{flag}")


if __name__ == "__main__":
    main()
