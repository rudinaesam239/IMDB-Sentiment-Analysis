import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ───────────────────────── Page config ─────────────────────────
st.set_page_config(page_title="CineSense · IMDB Sentiment AI", page_icon="🎬",
                   layout="wide", initial_sidebar_state="expanded")

POS, NEG, ACC, ACC2, MUTED = "#22c55e", "#ef4444", "#8b5cf6", "#06b6d4", "#94a3b8"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.block-container {{ padding-top: 1.6rem; max-width: 1300px; }}
#MainMenu, footer {{ visibility: hidden; }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg,#0f1424 0%,#0b0f1a 100%);
    border-right: 1px solid rgba(255,255,255,.06); }}

.hero {{ position:relative; overflow:hidden; padding: 2.2rem 2.4rem; border-radius: 22px;
    background: radial-gradient(1200px 400px at 0% 0%, rgba(139,92,246,.35), transparent 60%),
                radial-gradient(900px 400px at 100% 100%, rgba(6,182,212,.30), transparent 60%),
                #121830;
    border: 1px solid rgba(255,255,255,.08); margin-bottom: 1.4rem; }}
.hero h1 {{ margin:0; font-size: 2.3rem; font-weight: 800; letter-spacing:-.5px;
    background: linear-gradient(90deg,#fff,#c4b5fd 55%,#67e8f9); -webkit-background-clip:text;
    -webkit-text-fill-color: transparent; }}
.hero p {{ margin:.5rem 0 0; color:#b6bdd3; font-size:1.02rem; }}
.chips span {{ display:inline-block; margin:.9rem .4rem 0 0; padding:.28rem .75rem; border-radius:99px;
    background: rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.1); font-size:.78rem; color:#d6dbee; }}

.kpi {{ background: linear-gradient(145deg,#161d33,#10162a); border:1px solid rgba(255,255,255,.07);
    border-radius:18px; padding:1.1rem 1.2rem; height:100%; transition: transform .2s, border-color .2s; }}
.kpi:hover {{ transform: translateY(-3px); border-color: rgba(139,92,246,.6); }}
.kpi .l {{ color:{MUTED}; font-size:.74rem; text-transform:uppercase; letter-spacing:.09em; font-weight:600; }}
.kpi .v {{ font-size:1.95rem; font-weight:800; margin:.15rem 0; }}
.kpi .s {{ color:#7c86a2; font-size:.78rem; }}

.card {{ background:#121930; border:1px solid rgba(255,255,255,.07); border-radius:18px; padding:1.2rem 1.4rem; }}
.sec {{ font-size:1.25rem; font-weight:700; margin: 1.2rem 0 .2rem; }}
.sub {{ color:{MUTED}; font-size:.9rem; margin-bottom:.8rem; }}

.verdict {{ border-radius:20px; padding:1.4rem; text-align:center; border:1px solid; }}
.verdict .e {{ font-size:3rem; line-height:1; }}
.verdict .t {{ font-size:1.7rem; font-weight:800; margin-top:.3rem; }}
.verdict .c {{ color:#cbd2e6; font-size:.9rem; }}
.rev {{ line-height:2; font-size:1.02rem; padding:1rem 1.2rem; border-radius:14px;
    background:#0e1428; border:1px solid rgba(255,255,255,.06); }}
.rev mark {{ padding:.1rem .3rem; border-radius:6px; color:#fff; }}
.err {{ background:#121930; border-left:4px solid; border-radius:12px; padding:.9rem 1.1rem; margin-bottom:.7rem;
    font-size:.92rem; color:#d3d9ec; }}
.pill {{ display:inline-block; padding:.1rem .6rem; border-radius:99px; font-size:.75rem; font-weight:700; margin-bottom:.4rem; }}
div[data-testid="stTextArea"] textarea {{ border-radius:14px; font-size:1rem; }}
.stButton>button {{ border-radius:12px; border:1px solid rgba(255,255,255,.12); font-weight:600; }}
.stButton>button:hover {{ border-color:{ACC}; color:#fff; }}
</style>
""", unsafe_allow_html=True)


# ───────────────────────── Data / model ─────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load():
    p = Path(__file__).parent / "artifacts.joblib"
    if not p.exists():
        return None
    return joblib.load(p)


A = load()
if A is None:
    st.error("`artifacts.joblib` not found. Run `python train_model.py IMDB_Dataset.csv` first.")
    st.stop()

vec, model = A["vectorizer"], A["model"]
coef, bias = model.coef_[0], float(model.intercept_[0])
vocab = vec.vocabulary_
analyzer = vec.build_analyzer()
M = A["metrics"]
sigmoid = lambda z: 1 / (1 + np.exp(-z))


def predict(text, thr=0.5):
    x = vec.transform([text])
    z = float(x.dot(coef)[0] + bias)
    p = float(sigmoid(z))
    return dict(z=z, p=p, label="Positive" if p >= thr else "Negative", x=x)


def contributions(text):
    cnt = Counter(analyzer(text))
    rows = [(w, c, coef[vocab[w]], c * coef[vocab[w]]) for w, c in cnt.items() if w in vocab]
    return pd.DataFrame(rows, columns=["word", "count", "weight", "contribution"])


def highlight(text):
    text = re.sub(r"<br\s*/?>", " ", text)
    out, mx = [], max(abs(coef).max(), 1e-9)
    for tok in re.findall(r"\w+|\W+", text):
        key = tok.lower()
        if key in vocab:
            w = coef[vocab[key]]
            a = min(0.15 + 0.85 * abs(w) / (mx * 0.55), 0.95)
            col = f"rgba(34,197,94,{a:.2f})" if w > 0 else f"rgba(239,68,68,{a:.2f})"
            out.append(f'<mark style="background:{col}" title="weight {w:+.2f}">{tok}</mark>')
        else:
            out.append(tok.replace("<", "&lt;"))
    return "".join(out)


def style(fig, h=380, legend=True):
    fig.update_layout(template="plotly_dark", height=h, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(family="Inter", size=13), showlegend=legend,
                      legend=dict(orientation="h", y=1.12, x=0))
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    return fig


def kpi(label, value, sub="", color="#fff"):
    return f'<div class="kpi"><div class="l">{label}</div><div class="v" style="color:{color}">{value}</div><div class="s">{sub}</div></div>'


def section(title, sub=""):
    st.markdown(f'<div class="sec">{title}</div><div class="sub">{sub}</div>', unsafe_allow_html=True)


def hero(title, desc, chips):
    c = "".join(f"<span>{x}</span>" for x in chips)
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{desc}</p><div class="chips">{c}</div></div>',
                unsafe_allow_html=True)


# ───────────────────────── Sidebar ─────────────────────────
with st.sidebar:
    st.markdown("## 🎬 CineSense")
    st.caption("IMDB sentiment · Logistic Regression")
    page = st.radio("Navigate", [
        "🎯 Live Predictor", "📊 Model Performance", "🎚️ Threshold Lab",
        "🔍 Explainability", "🧮 BCE Lab", "🐞 Error Analysis", "🗂️ Dataset Explorer"],
        label_visibility="collapsed")
    st.divider()
    st.markdown(f"""
**Model**  
Logistic Regression (L2, C={A['best_C']})  
**Features**  
BoW · {len(vocab):,} words  
**Data split**  
{A['split']['train']:,} train / {A['split']['test']:,} test  
**Seed** · {A['seed']}
""")
    st.divider()
    st.metric("Test accuracy", f"{M['accuracy']*100:.2f}%")
    st.metric("ROC-AUC", f"{M['auc']:.4f}")


# ═════════════════════════ PAGES ═════════════════════════
if page == "🎯 Live Predictor":
    hero("Is this review a rave or a rant?",
         "Type any movie review and watch the model compute the logit, probability, and the exact words that drove its decision.",
         ["Bag-of-Words", "Logistic Regression", "Binary Cross-Entropy", "Explainable"])

    EXAMPLES = {
        "🌟 Triumph": "An absolute triumph of cinema with breathtaking visuals and stellar acting.",
        "😏 Sarcasm": "I thoroughly enjoyed wasting two hours of my life staring at a completely incoherent plot.",
        "🤔 Negation": "The cinematography was not bad, but the pacing was not good either.",
        "💔 Disaster": "Worst movie ever. Terrible script, awful acting, a complete waste of time and money.",
    }
    if "txt" not in st.session_state:
        st.session_state.txt = EXAMPLES["🌟 Triumph"]

    cols = st.columns(len(EXAMPLES))
    for c, (k, v) in zip(cols, EXAMPLES.items()):
        if c.button(k, width="stretch"):
            st.session_state.txt = v

    left, right = st.columns([1.6, 1])
    with left:
        text = st.text_area("Your review", key="txt", height=170, label_visibility="collapsed",
                            placeholder="Write a movie review…")
        thr = st.slider("Decision threshold τ", 0.05, 0.95, 0.50, 0.05,
                        help="Predict Positive when P(positive) ≥ τ")
    if not text.strip():
        st.info("Write a review to get a prediction.")
        st.stop()

    r = predict(text, thr)
    pos = r["label"] == "Positive"
    col = POS if pos else NEG
    with right:
        st.markdown(f"""<div class="verdict" style="border-color:{col}55;background:linear-gradient(145deg,{col}22,transparent)">
            <div class="e">{'😍' if pos else '😡'}</div>
            <div class="t" style="color:{col}">{r['label']}</div>
            <div class="c">P(positive) = <b>{r['p']:.1%}</b> &nbsp;·&nbsp; logit z = <b>{r['z']:+.3f}</b></div></div>""",
                    unsafe_allow_html=True)

    g1, g2 = st.columns([1, 1.2])
    with g1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=r["p"] * 100, number=dict(suffix="%", font=dict(size=40)),
            title=dict(text="Probability of Positive", font=dict(size=14, color=MUTED)),
            gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=col, thickness=0.28),
                       bgcolor="rgba(255,255,255,.04)", borderwidth=0,
                       steps=[dict(range=[0, thr * 100], color="rgba(239,68,68,.15)"),
                              dict(range=[thr * 100, 100], color="rgba(34,197,94,.15)")],
                       threshold=dict(line=dict(color="#fff", width=3), thickness=.85, value=thr * 100))))
        st.plotly_chart(style(fig, 300, False), width="stretch")
    with g2:
        df = contributions(text)
        if df.empty:
            st.warning("No words from the model's vocabulary were found in this review.")
        else:
            top = df.reindex(df.contribution.abs().sort_values(ascending=False).index).head(10).iloc[::-1]
            fig = go.Figure(go.Bar(x=top.contribution, y=top.word, orientation="h",
                                   marker_color=[POS if v > 0 else NEG for v in top.contribution],
                                   text=[f"{v:+.2f}" for v in top.contribution], textposition="outside"))
            fig.update_layout(title="Words pushing the decision (w × count)")
            st.plotly_chart(style(fig, 300, False), width="stretch")

    section("Word-level highlighting", "🟩 pushes toward Positive · 🟥 pushes toward Negative · intensity = learned weight. Hover for the exact value.")
    st.markdown(f'<div class="rev">{highlight(text)}</div>', unsafe_allow_html=True)

    with st.expander("🧮 Show the math for this prediction"):
        s = df.contribution.sum() if not df.empty else 0.0
        st.latex(r"z = \mathbf{w}^\top \mathbf{x} + b")
        st.markdown(f"Σ(wⱼ·xⱼ) = **{s:+.4f}** &nbsp; + &nbsp; b = **{bias:+.4f}** &nbsp; ⇒ &nbsp; **z = {r['z']:+.4f}**")
        st.latex(rf"\hat y = \sigma(z) = \frac{{1}}{{1+e^{{-({r['z']:.4f})}}}} = {r['p']:.4f}")
        st.caption("BoW ignores word order and negation — try the Negation and Sarcasm examples to see it fail.")


elif page == "📊 Model Performance":
    hero("Model Performance", "Everything evaluated on the held-out 10,000-review test set (τ = 0.5).",
         ["Stratified 80/20", f"C = {A['best_C']}", "300 L-BFGS iterations"])
    items = [("Accuracy", f"{M['accuracy']:.2%}", "overall correct", ACC),
             ("Precision", f"{M['precision']:.2%}", "of predicted positive", ACC2),
             ("Recall", f"{M['recall']:.2%}", "of actual positive", POS),
             ("F1-Score", f"{M['f1']:.2%}", "harmonic mean", "#f59e0b"),
             ("ROC-AUC", f"{M['auc']:.4f}", "ranking quality", "#ec4899"),
             ("Test BCE", f"{M['bce']:.4f}", "log loss", "#fff")]
    for c, (l, v, s, colr) in zip(st.columns(6), items):
        c.markdown(kpi(l, v, s, colr), unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        cm = A["cm"]
        lab = ["Negative", "Positive"]
        fig = go.Figure(go.Heatmap(z=cm, x=lab, y=lab, colorscale=[[0, "#131a2b"], [1, ACC]],
                                   showscale=False, text=[[f"{v:,}" for v in row] for row in cm],
                                   texttemplate="%{text}", textfont=dict(size=22)))
        fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual",
                          yaxis_autorange="reversed")
        st.plotly_chart(style(fig, 400, False), width="stretch")
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"TN {tn:,} · FP {fp:,} · FN {fn:,} · TP {tp:,}")
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=A["roc"]["fpr"], y=A["roc"]["tpr"], mode="lines", fill="tozeroy",
                        line=dict(color=ACC2, width=3), fillcolor="rgba(6,182,212,.12)",
                        name=f"AUC = {M['auc']:.4f}")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, dash="dash"), name="Random")
        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(style(fig, 400), width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        L = A["train_losses"]
        fig = go.Figure(go.Scatter(x=list(range(1, len(L) + 1)), y=L, mode="lines",
                                   line=dict(color=ACC, width=3), fill="tozeroy",
                                   fillcolor="rgba(139,92,246,.12)"))
        fig.update_layout(title="Training BCE Loss per L-BFGS iteration", xaxis_title="Iteration", yaxis_title="BCE")
        st.plotly_chart(style(fig, 360, False), width="stretch")
        st.caption(f"Loss fell from **{L[0]:.4f}** (iter 1) to **{L[-1]:.4f}** (iter {len(L)}) — smooth, monotone convergence.")
    with c4:
        g = A["grid"]
        best = int(np.argmax(g["acc"]))
        fig = go.Figure(go.Bar(x=[f"C={c}" for c in g["C"]], y=g["acc"],
                               marker_color=[POS if i == best else "#334155" for i in range(len(g["C"]))],
                               text=[f"{a:.4f}" for a in g["acc"]], textposition="outside"))
        fig.update_layout(title="5-fold CV accuracy by regularisation C",
                          yaxis_range=[min(g["acc"]) - 0.02, max(g["acc"]) + 0.01])
        st.plotly_chart(style(fig, 360, False), width="stretch")


elif page == "🎚️ Threshold Lab":
    hero("Threshold Lab", "Slide τ to see the precision ↔ recall trade-off on the test set.",
         ["τ ∈ [0.05, 0.95]", "Precision", "Recall", "F1"])
    T = pd.DataFrame(A["thresholds"])
    tau = st.select_slider("Decision threshold τ", options=T.tau.tolist(), value=0.5)
    row = T[T.tau == tau].iloc[0]
    cols = st.columns(4)
    for c, (l, k, colr) in zip(cols, [("Precision", "precision", ACC2), ("Recall", "recall", POS),
                                     ("F1-Score", "f1", "#f59e0b"), ("Accuracy", "accuracy", ACC)]):
        c.markdown(kpi(l, f"{row[k]:.2%}", f"at τ = {tau}", colr), unsafe_allow_html=True)

    fig = go.Figure()
    for k, colr in [("precision", ACC2), ("recall", POS), ("f1", "#f59e0b")]:
        fig.add_scatter(x=T.tau, y=T[k], mode="lines+markers", name=k.capitalize(), line=dict(color=colr, width=3))
    fig.add_vline(x=tau, line_color="#fff", line_dash="dash")
    fig.update_layout(title="Precision / Recall / F1 vs τ", xaxis_title="τ", yaxis_title="Score")
    st.plotly_chart(style(fig, 420), width="stretch")

    st.markdown(f"""<div class="card"><b>📌 When would you pick τ = 0.3?</b><br>
    Lower τ whenever a <b>missed positive (false negative)</b> costs more than a false alarm — e.g. a hospital
    no-show predictor: an extra reminder call is cheap, an empty appointment slot is not. You gain recall and accept lower precision.
    <br><br><b>Reading the table:</b> raising τ makes the model conservative (precision ↑, recall ↓); lowering τ makes it permissive (recall ↑, precision ↓).</div>""",
                unsafe_allow_html=True)
    st.write("")
    st.dataframe(T.set_index("tau").style.format("{:.4f}").background_gradient(cmap="Purples"), width="stretch")


elif page == "🔍 Explainability":
    hero("What did the model learn?", "Each of the 10,000 words has one learned weight. Positive weight → pushes toward Positive.",
         ["Top-10 words", "Word lookup", "Weight distribution"])
    names = np.array(vec.get_feature_names_out())
    n = st.slider("How many words per side", 5, 30, 10)
    ip, ine = np.argsort(coef)[::-1][:n], np.argsort(coef)[:n]
    c1, c2 = st.columns(2)
    for c, idx, colr, t in [(c1, ip, POS, f"Top {n} Positive words"), (c2, ine, NEG, f"Top {n} Negative words")]:
        fig = go.Figure(go.Bar(x=coef[idx][::-1], y=names[idx][::-1], orientation="h", marker_color=colr,
                               text=[f"{v:+.2f}" for v in coef[idx][::-1]], textposition="outside"))
        fig.update_layout(title=t, xaxis_title="Learned weight w")
        c.plotly_chart(style(fig, 140 + 28 * n, False), width="stretch")

    c3, c4 = st.columns([1, 1.4])
    with c3:
        section("🔎 Word lookup", "Check the weight of any word in the vocabulary.")
        w = st.text_input("Word", "masterpiece").strip().lower()
        if w in vocab:
            wt = coef[vocab[w]]
            rank = int((coef > wt).sum()) + 1
            st.markdown(kpi(f"“{w}”", f"{wt:+.3f}", f"rank {rank:,} of {len(coef):,}", POS if wt > 0 else NEG),
                        unsafe_allow_html=True)
        elif w:
            st.warning("Not in the 10,000-word vocabulary (or filtered as a stop-word).")
    with c4:
        fig = go.Figure(go.Histogram(x=coef, nbinsx=60, marker_color=ACC))
        fig.update_layout(title="Distribution of all learned weights", xaxis_title="w", yaxis_title="# words")
        st.plotly_chart(style(fig, 320, False), width="stretch")
    st.caption(f"Bias term b = {bias:+.4f}. Sensible sentiment words at both extremes confirm the model learned a real lexical signal.")


elif page == "🧮 BCE Lab":
    hero("Binary Cross-Entropy Lab", "Interactive version of Task 2 — tweak weights and see logit, probability and loss update live.",
         ["Sigmoid", "BCE vs MSE", "Vanishing gradients"])
    c1, c2 = st.columns([1, 1.2])
    with c1:
        section("Weights", "Defaults match the assignment: masterpiece, boring, acting.")
        w1 = st.slider("w · masterpiece", -4.0, 4.0, 1.8, 0.1)
        w2 = st.slider("w · boring", -4.0, 4.0, -2.1, 0.1)
        w3 = st.slider("w · acting", -4.0, 4.0, 0.5, 0.1)
        b = st.slider("bias b", -3.0, 3.0, -0.2, 0.1)
        on = st.multiselect("Words present in the review", ["masterpiece", "boring", "acting"],
                            default=["masterpiece", "boring", "acting"])
    z = b + sum(w for n, w in zip(["masterpiece", "boring", "acting"], [w1, w2, w3]) if n in on)
    yh = float(sigmoid(z))
    eps = 1e-12
    l1, l0 = -np.log(yh + eps), -np.log(1 - yh + eps)
    with c2:
        k = st.columns(2)
        k[0].markdown(kpi("Logit z", f"{z:+.3f}", "w·x + b", ACC), unsafe_allow_html=True)
        k[1].markdown(kpi("ŷ = σ(z)", f"{yh:.4f}", "P(positive)", ACC2), unsafe_allow_html=True)
        st.write("")
        k = st.columns(2)
        k[0].markdown(kpi("BCE if y = 1", f"{l1:.4f}", "−log(ŷ)", POS), unsafe_allow_html=True)
        k[1].markdown(kpi("BCE if y = 0", f"{l0:.4f}", "−log(1−ŷ)", NEG), unsafe_allow_html=True)
        ratio = max(l1, l0) / max(min(l1, l0), 1e-9)
        st.info(f"Wrong-vs-right penalty ratio: **{ratio:,.1f}×** — confident mistakes are punished hardest.")

    zs = np.linspace(-8, 8, 400)
    ys = sigmoid(zs)
    fig = go.Figure()
    fig.add_scatter(x=zs, y=np.abs(ys - 1), name="BCE gradient |ŷ − y|", line=dict(color=POS, width=3))
    fig.add_scatter(x=zs, y=np.abs(ys - 1) * ys * (1 - ys), name="MSE gradient |ŷ − y|·ŷ(1−ŷ)",
                    line=dict(color=NEG, width=3))
    fig.add_vline(x=z, line_dash="dash", line_color="#fff")
    fig.update_layout(title="Gradient w.r.t. z when true label y = 1", xaxis_title="logit z", yaxis_title="|∂J/∂z|")
    st.plotly_chart(style(fig, 380), width="stretch")
    st.markdown("""<div class="card"><b>Why BCE beats MSE for logistic regression</b><br>
    • <b>Vanishing gradients:</b> MSE carries the extra factor ŷ(1−ŷ), which collapses when the sigmoid saturates (left side of the red curve) — a confidently <i>wrong</i> model barely learns.
    BCE's gradient is just (ŷ − y), so it stays large exactly when the model is most wrong.<br>
    • <b>Convexity:</b> BCE is the Bernoulli negative log-likelihood — convex in (w, b), so every local minimum is global. MSE through a sigmoid is non-convex.</div>""",
                unsafe_allow_html=True)


elif page == "🐞 Error Analysis":
    hero("Error Analysis", "The most confident mistakes the model made on the test set — and why BoW fails on them.",
         ["False Positives", "False Negatives", "Negation · Sarcasm · Mixed"])
    k = st.slider("Examples per type", 1, 6, 3)
    c1, c2 = st.columns(2)
    for c, key, colr, title, desc in [
        (c1, "fp", NEG, "False Positives", "true = Negative · predicted = Positive"),
        (c2, "fn", POS, "False Negatives", "true = Positive · predicted = Negative")]:
        with c:
            section(title, desc)
            for e in A["errors"][key][:k]:
                txt = e["text"][:520] + ("…" if len(e["text"]) > 520 else "")
                st.markdown(f'<div class="err" style="border-color:{colr}"><span class="pill" style="background:{colr}33;color:{colr}">'
                            f'P(positive) = {e["p"]:.3f}</span><br>{txt}</div>', unsafe_allow_html=True)
    st.markdown("""<div class="card"><b>Typical failure patterns</b><br>
    <b>1 · Mixed reviews</b> — praise and criticism in the same text; BoW just sums word counts.<br>
    <b>2 · Negation</b> — “not bad”, “far from great” flip meaning, but each word keeps its own weight.<br>
    <b>3 · Sarcasm</b> — literal words are positive while the intent is negative.</div>""", unsafe_allow_html=True)


else:
    D = A["data"]
    hero("Dataset Explorer", f"{D['n']:,} IMDB movie reviews, perfectly balanced between classes.",
         ["50,000 reviews", "25K positive / 25K negative", f"{len(vocab):,} BoW features"])
    cols = st.columns(4)
    for c, (l, v, s, colr) in zip(cols, [
            ("Reviews", f"{D['n']:,}", "total", ACC), ("Mean length", f"{D['mean_len']:.0f}", "words", ACC2),
            ("Median length", f"{D['median_len']:.0f}", "words", POS), ("BoW sparsity", f"{D['sparsity']:.2f}%", "zeros in X_train", "#f59e0b")]):
        c.markdown(kpi(l, v, s, colr), unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns([1, 2])
    with c1:
        fig = go.Figure(go.Pie(labels=list(D["counts"].keys()), values=list(D["counts"].values()), hole=.62,
                               marker_colors=[POS if k == "positive" else NEG for k in D["counts"]],
                               textinfo="label+percent"))
        fig.update_layout(title="Class balance")
        st.plotly_chart(style(fig, 360, False), width="stretch")
    with c2:
        fig = go.Figure()
        for k, colr in [("positive", POS), ("negative", NEG)]:
            h, e = D["len_hist"][k]
            fig.add_bar(x=(e[:-1] + e[1:]) / 2, y=h, name=k, marker_color=colr, opacity=.7)
        fig.update_layout(barmode="overlay", title="Review length distribution (words, clipped at 1000)",
                          xaxis_title="words", yaxis_title="# reviews")
        st.plotly_chart(style(fig, 360), width="stretch")

    section("Browse random reviews", "400-review random sample. Click a row's text to read it, or run it through the predictor.")
    s = D["sample"]
    f = st.radio("Filter", ["all", "positive", "negative"], horizontal=True)
    if f != "all":
        s = s[s.sentiment == f]
    st.dataframe(s, width="stretch", height=360,
                 column_config={"review": st.column_config.TextColumn("Review", width="large"),
                                "sentiment": "Label", "n_words": "Words"})
    if st.button("🎲 Predict a random review from this sample"):
        row = s.sample(1).iloc[0]
        r = predict(row.review)
        ok = r["label"].lower() == row.sentiment
        st.markdown(f'<div class="rev">{highlight(row.review)}</div>', unsafe_allow_html=True)
        st.write(f"**True:** {row.sentiment} · **Model:** {r['label']} ({r['p']:.1%}) → " + ("✅ correct" if ok else "❌ wrong"))
