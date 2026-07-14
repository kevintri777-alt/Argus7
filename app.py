import glob
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.model_selection import TimeSeriesSplit

st.set_page_config(page_title="ARGUS // VOL_LAB", layout="wide")

# ---------------- terminal skin ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&display=swap');
html, body, [class*="st-"], .stMarkdown, button, input, label {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
}
.block-container { max-width: 1500px; padding-top: 0.8rem; }
.statusbar {
    display:flex; justify-content:space-between; background:#000;
    border:1px solid #262626; border-radius:6px; padding:.45rem .9rem;
    font-size:.75rem; color:#9a9a9a; margin-bottom:.8rem;
}
.statusbar b { color:#ffb300; }
.bighead { font-size:1.9rem; font-weight:800; margin:.2rem 0 .6rem 0; }
.bighead .sym { color:#fff; } .bighead .sep { color:#555; }
.bighead .dt  { color:#ffb300; }
.statgrid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:.8rem; }
.statcard {
    background:#101010; border:1px solid #2a2a2a; border-radius:8px;
    padding:.7rem 1rem .8rem 1rem;
}
.statcard .lbl { font-size:.68rem; letter-spacing:.12em; color:#7a7a7a;
                 text-transform:uppercase; }
.statcard .val { font-size:2.1rem; font-weight:800; line-height:1.15; }
.v-white{color:#f2f2f2}.v-cyan{color:#39d8e8}.v-green{color:#3ddc84}
.v-amber{color:#ffb300}.v-red{color:#ff5f6b}
.chip { display:inline-block; font-size:.66rem; font-weight:700;
        padding:.1rem .5rem; border-radius:4px; margin-top:.35rem;
        border:1px solid; }
.c-green{ color:#3ddc84; border-color:#3ddc84; background:#0d2417; }
.c-red  { color:#ff5f6b; border-color:#ff5f6b; background:#2a1214; }
.c-amber{ color:#ffb300; border-color:#ffb300; background:#241c08; }
.c-cyan { color:#39d8e8; border-color:#39d8e8; background:#08222a; }
.ticket {
    background:#101010; border:1px solid #2a2a2a; border-radius:8px;
    padding:.8rem 1rem; margin-bottom:.8rem;
}
.ticket .thead { display:flex; justify-content:space-between;
                 margin-bottom:.55rem; font-size:.9rem; }
.ticket .size { color:#ffb300; font-weight:700; }
.leg { display:flex; justify-content:space-between; align-items:center;
       padding:.45rem 0; border-top:1px dashed #222; font-size:.92rem; color:#e6e6e6; }
.leg .note { color:#6f6f6f; font-size:.72rem; }
.act { display:inline-block; min-width:3.6rem; text-align:center;
       font-size:.86rem; font-weight:800; letter-spacing:.09em;
       padding:.26rem .6rem; border-radius:5px; margin-right:.7rem; }
.act.buy  { color:#06140c; background:#3ddc84; }
.act.sell { color:#1c0709; background:#ff5f6b; }
.zrow { display:flex; align-items:center; gap:.6rem; margin-top:.55rem;
        font-size:.7rem; color:#7a7a7a; }
.zbar { flex:1; height:6px; background:#1d1d1d; border-radius:3px; }
.zfill{ height:6px; background:#ffb300; border-radius:3px; }
.zval { color:#ffb300; font-weight:700; }
.bandwrap { background:#101010; border:1px solid #2a2a2a; border-radius:8px;
            padding:.7rem 1rem; }
.bandlbl { font-size:.68rem; letter-spacing:.12em; color:#7a7a7a;
           text-transform:uppercase; margin-bottom:.5rem; }
.band { display:flex; height:20px; border-radius:4px; overflow:hidden;
        font-size:.62rem; font-weight:700; text-align:center; color:#0a0a0a; }
.band div { display:flex; align-items:center; justify-content:center; }
.cuts { display:flex; justify-content:space-between; font-size:.66rem;
        color:#6f6f6f; margin-top:.3rem; }
.footline { font-size:.72rem; color:#6f6f6f; border-top:1px solid #222;
            padding-top:.5rem; margin-top:.6rem; }
.notrade { background:#101010; border:1px solid #2a2a2a; border-radius:8px;
           padding:1.2rem; color:#9a9a9a; font-size:.9rem; }
.aboutbox { background:#101010; border:1px solid #2a2a2a; border-radius:8px;
            padding:1rem 1.2rem; margin-bottom:.8rem; font-size:.86rem;
            line-height:1.55; color:#cfcfcf; }
.aboutbox h4 { color:#ffb300; font-size:.8rem; letter-spacing:.12em;
               text-transform:uppercase; margin:0 0 .4rem 0; }
</style>
""", unsafe_allow_html=True)

# ---------------- data / model (unchanged logic) ----------------
FILES = {Path(f).stem.split("_")[0].upper(): f
         for f in glob.glob("data/features/*_features.parquet")}
if not FILES:
    st.error("No feature files found in data/features/")
    st.stop()

ZWIN, ZCLIP, H = 60, 2.0, 21
ZFIRE, ZHIGH = 1.5, 2.0          # fire ticket at |z|>1.5; flag |z|>2 as high-conviction
HAR = ["rv_5", "rv_21", "rv_63", "atm_iv"]
QLEVELS = [0.1, 0.5, 0.9]        # quantile-regression confidence band

@st.cache_data
def load(path):
    full = pd.read_parquet(path)
    full["iv_rank_252"] = full["iv_rank_252"].fillna(0.5)
    full = full.sort_values("date").reset_index(drop=True)
    for col, zc in [("iv_term_slope", "z_term"), ("iv_skew", "z_skew")]:
        r = full[col].rolling(ZWIN)
        full[zc] = (full[col] - r.mean()) / r.std()
    df = full[full.has_label].copy()
    lo, hi = df["rv_21"].quantile([1/3, 2/3]).tolist()
    Xh, yh = df[HAR].values, df["fwd_rv_21"].values
    m = LinearRegression().fit(Xh, yh)
    W = dict(zip(HAR, m.coef_, strict=True))
    W["intercept"] = m.intercept_
    QB = {}                       # quantile band: q -> (coef array, intercept)
    for q in QLEVELS:
        qm = QuantileRegressor(quantile=q, alpha=0.0, solver="highs").fit(Xh, yh)
        QB[q] = (qm.coef_.copy(), float(qm.intercept_))
    return full, W, lo, hi, QB

@st.cache_data
def track_record(path):
    full, _, _, _, _ = load(path)
    rows = []
    for src, name in [("iv_term_slope", "TERM"), ("iv_skew", "SKEW")]:
        s, z = full[src], full["z_term" if name == "TERM" else "z_skew"]
        fut = s.shift(-H) - s
        for i in range(ZWIN, len(full) - H, H):
            zi = z.iloc[i]
            if not np.isfinite(zi) or not np.isfinite(fut.iloc[i]):
                continue
            pnl = float(-np.clip(zi, -ZCLIP, ZCLIP) * fut.iloc[i]) * 100
            rows.append({"date": full["date"].iloc[i].date(), "signal": name,
                         "fired (|z|>1.5)": "YES" if abs(zi) > ZFIRE else "-",
                         "conv": "HIGH" if abs(zi) > ZHIGH else "-",
                         "z": round(float(zi), 2),
                         "size": round(float(abs(np.clip(zi, -ZCLIP, ZCLIP))), 2),
                         "P&L (vol pts)": round(pnl, 2),
                         "result": "WIN" if pnl > 0 else "LOSS"})
    return pd.DataFrame(rows).sort_values("date", ascending=False)

@st.cache_data
def forecast_stats(path):
    """Walk-forward accuracy + full in-sample forecast series for this ticker."""
    full, W, _, _, QB = load(path)
    df = full[full.has_label].copy().reset_index(drop=True)
    X, y = df[HAR].values, df["fwd_rv_21"].values
    corrs = []
    for tr, te in TimeSeriesSplit(4).split(X):
        p = LinearRegression().fit(X[tr], y[tr]).predict(X[te])
        corrs.append(pearsonr(p, y[te])[0])
    df["forecast"] = (X @ np.array([W[k] for k in HAR])) + W["intercept"]
    df["f_lo"] = (X @ QB[0.1][0]) + QB[0.1][1]
    df["f_hi"] = (X @ QB[0.9][0]) + QB[0.9][1]
    df["f_lo"] = np.minimum(df["f_lo"], df["forecast"])
    df["f_hi"] = np.maximum(df["f_hi"], df["forecast"])
    mape_model = float((np.abs(df["forecast"] - y) / y).mean())
    mape_lazy = float((np.abs(df["rv_21"] - y) / y).mean())
    return df[["date", "forecast", "fwd_rv_21", "f_lo", "f_hi"]], \
        float(np.mean(corrs)), mape_model, mape_lazy

# ---------------- controls ----------------
c_tk, c_dt, c_gap = st.columns([2, 1, 2])
with c_tk:
    ticker = st.radio("SYMBOL", sorted(FILES), horizontal=True)
full, W, REGIME_LO, REGIME_HI, QB = load(FILES[ticker])

def regime_of(v):
    return "calm" if v < REGIME_LO else ("normal" if v < REGIME_HI else "stressed")

with c_dt:
    d = st.date_input("AS-OF DATE", value=full.date.max().date(),
                      min_value=full.date.min().date(),
                      max_value=full.date.max().date())
row = full[full.date <= pd.Timestamp(d)].iloc[-1]

fc = sum(W[k] * row[k] for k in HAR) + W["intercept"]

def q_row(q):
    c, b = QB[q]
    return float(sum(c[i] * row[HAR[i]] for i in range(len(HAR))) + b)

fc_lo, fc_hi = sorted([q_row(0.1), q_row(0.9)])
fc_lo, fc_hi = min(fc_lo, fc), max(fc_hi, fc)   # band always brackets the point
edge = fc - row["atm_iv"]
reg = regime_of(fc)
size_mult = {"calm": 1.0, "normal": 1.0, "stressed": 0.5}[reg]
S = row["c"]

# ---------------- status bar + header ----------------
st.markdown(
    f"<div class='statusbar'><span>&#9679; <b>ARGUS</b> v1.0 &nbsp; "
    f"&gt; run vol.lab --symbol {ticker} --asof {row['date'].date()}</span>"
    f"<span>by Kevin Trivedi &amp; Vivan Jhaveri</span></div>",
    unsafe_allow_html=True)
st.markdown(
    f"<div class='bighead'><span class='sym'>{ticker}</span> "
    f"<span class='sep'>//</span> <span class='dt'>{row['date'].date()}</span>"
    f"<span style='float:right;font-size:1rem;color:#9a9a9a;padding-top:.6rem'>"
    f"spot <b style='color:#fff'>{S:.2f}</b></span></div>",
    unsafe_allow_html=True)

# ---------------- stat cards ----------------
lvl = ("OPTIONS RICH" if edge < -0.05 else
       "OPTIONS CHEAP" if edge > 0.05 else "FAIRLY PRICED")
edge_cls = "c-red" if edge < -0.05 else ("c-green" if edge > 0.05 else "c-amber")
reg_cls = {"calm": "v-green", "normal": "v-green", "stressed": "v-red"}[reg]
st.markdown(f"""
<div class='statgrid'>
 <div class='statcard'><div class='lbl'>Forecast 21d vol</div>
   <div class='val v-white'>{fc:.0%}</div>
   <span class='chip c-amber'>model HAR-RV-IV</span>
   <span class='chip c-cyan'>80% band {fc_lo:.0%}&ndash;{fc_hi:.0%}</span></div>
 <div class='statcard'><div class='lbl'>Market IV</div>
   <div class='val v-cyan'>{row['atm_iv']:.0%}</div>
   <span class='chip {edge_cls}'>edge {edge:+.1%} &middot; {lvl}</span></div>
 <div class='statcard'><div class='lbl'>Regime</div>
   <div class='val {reg_cls}'>{reg.upper()}</div>
   <span class='chip c-green'>size x{size_mult}</span></div>
</div>
""", unsafe_allow_html=True)
st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

tab_decide, tab_forecast, tab_record, tab_lab, tab_history, tab_about = st.tabs(
    ["VOL_LAB", "FORECAST", "TRACK_RECORD", "MODEL_LAB", "HISTORY", "ABOUT"])

def zbar_html(label, z):
    pct = min(abs(z) / ZCLIP, 1.0) * 100
    return (f"<div class='zrow'><span>{label}</span>"
            f"<div class='zbar'><div class='zfill' style='width:{pct:.0f}%'></div></div>"
            f"<span class='zval'>{z:+.2f}</span></div>")

def leg_html(action, text, price, note=""):
    a = "buy" if action == "BUY" else "sell"
    note_html = f" <span class='note'>{note}</span>" if note else ""
    return (f"<div class='leg'><span><span class='act {a}'>{action}</span>"
            f"{text}{note_html}</span><span>{price}</span></div>")

def conv_html(z):
    return ("<span class='chip c-green' style='margin-left:.4rem'>HIGH CONVICTION</span>"
            if abs(z) > ZHIGH else "")

# ================= VOL_LAB =================
with tab_decide:
    left, right = st.columns([1, 2.2])

    with left:
        any_ticket = False
        z = row["z_term"]
        if np.isfinite(z) and abs(z) > ZFIRE:
            any_ticket = True
            size = round(abs(np.clip(z, -ZCLIP, ZCLIP)) * size_mult, 2)
            if z < 0:
                legs = (leg_html("SELL", "1-month call", f"~{S:.0f}", "panic-priced")
                        + leg_html("BUY", "4-month call", f"~{S:.0f}"))
            else:
                legs = (leg_html("BUY", "1-month call", f"~{S:.0f}")
                        + leg_html("SELL", "4-month call", f"~{S:.0f}"))
            st.markdown(
                f"<div class='ticket'><div class='thead'>"
                f"<span><span class='chip c-amber'>TERM</span> "
                f"<b>calendar spread</b>{conv_html(z)}</span>"
                f"<span class='size'>size {size}x</span></div>"
                f"{legs}{zbar_html('z_term', z)}</div>", unsafe_allow_html=True)

        z = row["z_skew"]
        if np.isfinite(z) and abs(z) > ZFIRE:
            any_ticket = True
            size = round(abs(np.clip(z, -ZCLIP, ZCLIP)) * size_mult, 2)
            if z > 0:
                legs = (leg_html("SELL", "1-month put", f"~{S*0.9:.0f}",
                                 "overpriced fear")
                        + leg_html("BUY", "1-month call", f"~{S*1.1:.0f}"))
            else:
                legs = (leg_html("BUY", "1-month put", f"~{S*0.9:.0f}",
                                 "cheap protection")
                        + leg_html("SELL", "1-month call", f"~{S*1.1:.0f}"))
            st.markdown(
                f"<div class='ticket'><div class='thead'>"
                f"<span><span class='chip c-amber'>SKEW</span> "
                f"<b>risk reversal</b>{conv_html(z)}</span>"
                f"<span class='size'>size {size}x</span></div>"
                f"{legs}{zbar_html('z_skew', z)}</div>", unsafe_allow_html=True)

        if not any_ticket:
            st.markdown("<div class='notrade'>NO TRADE — nothing dislocated "
                        "beyond |z| &gt; 1.5 today.</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class='bandwrap'><div class='bandlbl'>// regime bands</div>
        <div class='band'>
          <div style='flex:1;background:#3ddc84'>CALM</div>
          <div style='flex:1;background:#ffb300'>NORMAL</div>
          <div style='flex:1;background:#ff5f6b'>STRESSED</div>
        </div>
        <div class='cuts'><span>&lt;{REGIME_LO:.0%}</span>
        <span>{REGIME_LO:.0%} – {REGIME_HI:.0%}</span>
        <span>&gt;{REGIME_HI:.0%}</span></div></div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("<div style='font-size:.72rem;color:#39d8e8;"
                    "letter-spacing:.1em'>VOL_HISTORY // IV_ATM vs RV_21D</div>",
                    unsafe_allow_html=True)
        hist = full[["date", "rv_21", "atm_iv"]].dropna().copy()
        lines = alt.Chart(
            hist.melt("date", ["atm_iv", "rv_21"], "series", "vol")
        ).mark_line(strokeWidth=1.4).encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("vol:Q", title=None),
            color=alt.Color("series:N",
                            scale=alt.Scale(domain=["atm_iv", "rv_21"],
                                            range=["#39d8e8", "#ffb300"]),
                            legend=alt.Legend(title="", orient="top-right",
                                              labelExpr="datum.value == 'atm_iv' "
                                              "? 'implied vol (ATM)' "
                                              ": 'realized vol (21d)'")),
        ).properties(height=430)
        pin = alt.Chart(pd.DataFrame({"date": [row["date"]]})).mark_rule(
            strokeWidth=1.5, strokeDash=[5, 4], color="#e8e8e8"
        ).encode(x="date:T")
        st.altair_chart((lines + pin).configure_view(stroke=None)
                        .configure_axis(gridColor="#1c1c1c", labelColor="#8a8a8a",
                                        domainColor="#333"),
                        use_container_width=True)
        st.markdown(f"<div class='footline'>z_term {row['z_term']:+.2f} &nbsp;|&nbsp; "
                    f"z_skew {row['z_skew']:+.2f} &nbsp;|&nbsp; spot {S:.2f} "
                    f"&nbsp;|&nbsp; regimes: calm&lt;{REGIME_LO:.0%}"
                    f"&lt;normal&lt;{REGIME_HI:.0%}&lt;stressed &nbsp;|&nbsp; "
                    f"research only — paper-trade</div>",
                    unsafe_allow_html=True)

# ================= FORECAST =================
with tab_forecast:
    fdf, wf_corr, mape_m, mape_l = forecast_stats(FILES[ticker])
    st.markdown(f"""
    <div class='statgrid' style='grid-template-columns:repeat(4,1fr)'>
     <div class='statcard'><div class='lbl'>Walk-forward corr</div>
       <div class='val v-cyan'>{wf_corr:+.2f}</div></div>
     <div class='statcard'><div class='lbl'>Model accuracy (MAPE)</div>
       <div class='val v-white'>{1-mape_m:.0%}</div></div>
     <div class='statcard'><div class='lbl'>Lazy baseline</div>
       <div class='val v-amber'>{1-mape_l:.0%}</div></div>
     <div class='statcard'><div class='lbl'>Model edge</div>
       <div class='val v-green'>{(mape_l-mape_m)*100:+.1f} pts</div></div>
    </div>""", unsafe_allow_html=True)
    st.caption("Accuracy = 100% minus the average % miss. The lazy baseline "
               "guesses 'next month = last month'. The edge is the gap.")
    band = alt.Chart(fdf).mark_area(opacity=0.14, color="#ffb300").encode(
        x=alt.X("date:T", title=None), y=alt.Y("f_lo:Q", title="21d vol"),
        y2="f_hi:Q")
    lines = alt.Chart(
        fdf.melt("date", ["fwd_rv_21", "forecast"], "series", "vol")
    ).mark_line(strokeWidth=1.4).encode(
        x=alt.X("date:T", title=None), y=alt.Y("vol:Q", title="21d vol"),
        color=alt.Color("series:N",
                        scale=alt.Scale(domain=["fwd_rv_21", "forecast"],
                                        range=["#39d8e8", "#ffb300"]),
                        legend=alt.Legend(title="", orient="top-right",
                                          labelExpr="datum.value == 'forecast' "
                                          "? 'model forecast' "
                                          ": 'realized (what happened)'")),
    ).properties(height=400)
    st.altair_chart((band + lines).configure_view(stroke=None)
                    .configure_axis(gridColor="#1c1c1c", labelColor="#8a8a8a",
                                    domainColor="#333"),
                    use_container_width=True)
    st.caption("Shaded band = 10th–90th percentile quantile-regression forecast "
               "(probabilistic confidence, not a point guess).")
    wtxt = " &nbsp;&middot;&nbsp; ".join(
        f"{k} <b class='v-amber'>{v:+.3f}</b>" for k, v in W.items())
    st.markdown(f"<div class='footline'>learned weights: {wtxt}</div>",
                unsafe_allow_html=True)

# ================= TRACK_RECORD =================
with tab_record:
    tr = track_record(FILES[ticker])
    if tr.empty:
        st.info("Not enough history for a track record.")
    else:
        wins = (tr["P&L (vol pts)"] > 0).mean()
        fired = (tr["fired (|z|>1.5)"] == "YES").sum()
        high = (tr["conv"] == "HIGH").sum()
        st.markdown(f"""
        <div class='statgrid' style='grid-template-columns:repeat(4,1fr)'>
         <div class='statcard'><div class='lbl'>Periods</div>
           <div class='val v-white'>{len(tr)}</div></div>
         <div class='statcard'><div class='lbl'>Win rate</div>
           <div class='val v-green'>{wins:.0%}</div></div>
         <div class='statcard'><div class='lbl'>Cum P&L (vol pts)</div>
           <div class='val v-cyan'>{tr["P&L (vol pts)"].sum():+.1f}</div></div>
         <div class='statcard'><div class='lbl'>Fires |z|&gt;1.5 / high-conv &gt;2</div>
           <div class='val v-white'>{int(fired)} /
           <span class='v-amber'>{int(high)}</span></div></div>
        </div>""", unsafe_allow_html=True)
        st.caption("Conviction-sized bets, non-overlapping 21-day periods, "
                   "pre-cost — real spreads would absorb most of this.")
        left, right = st.columns([1, 1])
        with left:
            cum = (tr.sort_values("date")
                     .assign(cum=lambda x: x["P&L (vol pts)"].cumsum()))
            st.line_chart(cum.set_index("date")[["cum"]]
                          .rename(columns={"cum": "cumulative P&L (vol pts)"}),
                          height=380, color="#ffb300")
        with right:
            st.dataframe(tr, use_container_width=True, hide_index=True,
                         height=380)

# ================= MODEL_LAB =================
with tab_lab:
    st.markdown("<div class='aboutbox'><h4>// model tournament — how the "
                "forecaster was chosen</h4>"
                "Every candidate was built, trained, and evaluated under "
                "walk-forward validation on TSLA (2019–2022). The winner runs "
                "this app; the losers are kept here as evidence.</div>",
                unsafe_allow_html=True)
    tourney = pd.DataFrame([
        {"model": "Transformer (encoder, 2 layers)", "features": 14,
         "out-of-sample score": "R2 -4.9", "verdict": "REJECTED — overfit"},
        {"model": "LightGBM (gradient-boosted trees)", "features": 14,
         "out-of-sample score": "corr 0.09", "verdict": "REJECTED"},
        {"model": "Ridge (9-feature signal model)", "features": 9,
         "out-of-sample score": "Sharpe 0.42 vs rule 0.60", "verdict": "REJECTED"},
        {"model": "HAR-RV (classic baseline)", "features": 3,
         "out-of-sample score": "corr 0.11", "verdict": "baseline"},
        {"model": "HAR-RV-IV (winner)", "features": 4,
         "out-of-sample score": "corr 0.24 (up to 0.60)", "verdict": "SELECTED"},
        {"model": "SHAR-RV-IV (semivariance + log-vol)", "features": 5,
         "out-of-sample score": "corr 0.16-0.52 (4-tkr wf)",
         "verdict": "TESTED — no lift vs HAR-RV-IV"},
    ])
    st.dataframe(tourney, use_container_width=True, hide_index=True)
    st.markdown(
        "<div class='aboutbox'><h4>// SHAR extension — tested, not promoted</h4>"
        "We split realized vol into downside/upside semivariance (SHAR) and "
        "modelled log-volatility, then re-ran the same walk-forward split on "
        "every ticker. Walk-forward correlation, HAR-RV-IV vs SHAR-log: "
        "<b class='v-amber'>TSLA 0.20 / 0.16 &middot; AAPL 0.61 / 0.55 &middot; "
        "NVDA 0.29 / 0.22 &middot; QQQ 0.48 / 0.52</b>. HAR-RV-IV wins 3 of 4; "
        "SHAR edges ahead only on QQQ, and with worse error there. Log-vol does "
        "fix one real defect — plain regression can emit near-zero or negative "
        "variance (QLIKE blow-up on TSLA) — but it trades correlation for that "
        "stability. Per our promotion rule, HAR-RV-IV stays in production and "
        "SHAR is kept here as evidence.</div>", unsafe_allow_html=True)
    fdf, wf_corr, mape_m, mape_l = forecast_stats(FILES[ticker])
    st.markdown(
        f"<div class='aboutbox'><h4>// this ticker ({ticker})</h4>"
        f"Retrained on {ticker}'s own history at load time. "
        f"Walk-forward correlation here: <b class='v-cyan'>{wf_corr:+.2f}</b>. "
        f"The same 4-feature model adapts its weights per ticker — see the "
        f"FORECAST tab footer for the learned values.</div>",
        unsafe_allow_html=True)
    st.markdown(
        "<div class='aboutbox'><h4>// honest finding</h4>"
        "Simple beat complex at this data size (~1,000 days per ticker) — "
        "reproducing the published result that parsimonious models dominate "
        "deep learning on single-asset volatility. Transformer, LightGBM, and "
        "now SHAR/log-vol were all tested and none beat the 4-feature model. "
        "The term/skew reversion "
        "signal replicated on all four tickers (TSLA, AAPL, NVDA, QQQ) "
        "pre-cost; the cost-aware backtest shows bid-ask spreads absorb the "
        "edge at retail execution.</div>", unsafe_allow_html=True)

# ================= HISTORY =================
with tab_history:
    hist = full[["date", "rv_21", "atm_iv"]].dropna().copy()
    hist["regime"] = hist["rv_21"].apply(regime_of)
    hist["date2"] = hist["date"].shift(-1).fillna(hist["date"])
    ymax = float(max(hist["rv_21"].max(), hist["atm_iv"].max()) * 1.05)
    bands = alt.Chart(hist).mark_rect(opacity=0.10).encode(
        x="date:T", x2="date2:T", y=alt.datum(0), y2=alt.datum(ymax),
        color=alt.Color("regime:N",
                        scale=alt.Scale(domain=["calm", "normal", "stressed"],
                                        range=["#3ddc84", "#ffb300", "#ff5f6b"]),
                        legend=alt.Legend(title="regime (trailing)")),
    ).properties(height=440)
    lines = alt.Chart(
        hist.melt("date", ["atm_iv", "rv_21"], "series", "vol")
    ).mark_line(strokeWidth=1.4).encode(
        x="date:T", y=alt.Y("vol:Q", title="annualized vol"),
        color=alt.Color("series:N",
                        scale=alt.Scale(domain=["atm_iv", "rv_21"],
                                        range=["#39d8e8", "#ffb300"]),
                        legend=alt.Legend(title="")),
    )
    pin = alt.Chart(pd.DataFrame({"date": [row["date"]]})).mark_rule(
        strokeWidth=1.5, strokeDash=[5, 4], color="#e8e8e8").encode(x="date:T")
    st.altair_chart((bands + lines + pin).configure_view(stroke=None)
                    .configure_axis(gridColor="#1c1c1c", labelColor="#8a8a8a",
                                    domainColor="#333"),
                    use_container_width=True)
    st.caption("Cyan = implied vol (market's expectation). Amber = realized vol "
               "(what happened). Background = regime; dashed line = selected date.")

# ================= ABOUT =================
with tab_about:
    a, b = st.columns(2)
    with a:
        st.markdown(
            "<div class='aboutbox'><h4>// what argus is</h4>"
            "An AI-driven research platform for American-style equity options. "
            "It forecasts each stock's next-month volatility, compares the "
            "forecast to what the option market is charging, detects "
            "dislocations in the volatility surface (term structure and skew), "
            "and turns them into explained, regime-sized trade tickets.</div>"
            "<div class='aboutbox'><h4>// the four layers</h4>"
            "L0 DATA — 13M+ option records wrangled through a 9-stage, "
            "gate-checked pipeline.<br>"
            "L1 PRICING — binomial-tree American pricer + implied-vol solver, "
            "validated against Black–Scholes.<br>"
            "L2 FORECAST — HAR-RV-IV model, selected by tournament, retrained "
            "per ticker.<br>"
            "L3 DECISION — transparent rules: dislocation z-scores, regime "
            "gate, conviction sizing.</div>", unsafe_allow_html=True)
    with b:
        st.markdown(
            "<div class='aboutbox'><h4>// the strategy, plainly</h4>"
            "Direction is unpredictable (measured: R2 ~ 0.00) — so ARGUS never "
            "bets on it. Instead it bets on bent price relationships snapping "
            "back: when panic makes short-dated insurance abnormally expensive "
            "vs long-dated (TERM), or crash-fear inflates puts vs calls (SKEW). "
            "Both trades are hedged pairs; both are sized down automatically "
            "in stressed regimes.</div>"
            "<div class='aboutbox'><h4>// honesty box</h4>"
            "Signals are statistically validated (win rate 79%, n=43, t~5, "
            "replicated on 4 tickers) but PRE-COST: the cost-aware backtest "
            "shows bid-ask spreads absorb the edge at retail execution. ARGUS "
            "is a research and education platform in paper-trading stage — "
            "not investment advice.</div>"
            "<div class='aboutbox'><h4>// credits</h4>"
            "Built by Kevin Trivedi and Vivan Jhaveri.<br>"
            "Thanks to our mentor Mr. Ashutosh, The Innovation Story, and "
            "Mr. Henry Cippola.</div>", unsafe_allow_html=True)
