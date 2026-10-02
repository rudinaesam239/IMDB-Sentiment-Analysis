"""
train_model.py
--------------
Reproduces the notebook pipeline (BoW + Logistic Regression + BCE) and saves
everything the Streamlit app needs into artifacts.joblib.

Usage:  python train_model.py [path/to/IMDB_Dataset.csv]
"""
import sys, re, time, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (log_loss, accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, roc_auc_score)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
SEED = 42
np.random.seed(SEED)

path = sys.argv[1] if len(sys.argv) > 1 else "IMDB_Dataset.csv"
t0 = time.time()
df = pd.read_csv(path)
assert df.shape == (50000, 2), f"Unexpected shape {df.shape}"
df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})
df["n_words"] = df["review"].str.split().str.len()
print("Loaded", df.shape)

Xtr_t, Xte_t, ytr, yte = train_test_split(
    df["review"], df["label"], test_size=0.20, random_state=SEED, stratify=df["label"])

vec = CountVectorizer(max_features=10000, stop_words="english")
Xtr = vec.fit_transform(Xtr_t)
Xte = vec.transform(Xte_t)

grid = GridSearchCV(
    LogisticRegression(solver="lbfgs", max_iter=1000, random_state=SEED),
    {"C": [0.01, 0.1, 1, 10]}, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(Xtr, ytr)
best_C = grid.best_params_["C"]
print("Best C:", best_C)

model = LogisticRegression(solver="lbfgs", max_iter=1, warm_start=True,
                           C=best_C, random_state=SEED)
train_losses = []
for _ in range(300):
    model.fit(Xtr, ytr)
    train_losses.append(log_loss(ytr, model.predict_proba(Xtr)[:, 1]))

prob = model.predict_proba(Xte)[:, 1]
pred = (prob >= 0.5).astype(int)
metrics = dict(
    bce=log_loss(yte, prob), accuracy=accuracy_score(yte, pred),
    precision=precision_score(yte, pred), recall=recall_score(yte, pred),
    f1=f1_score(yte, pred), auc=roc_auc_score(yte, prob))
fpr, tpr, _ = roc_curve(yte, prob)
step = max(1, len(fpr) // 600)

thr_rows = []
for tau in np.round(np.arange(0.05, 0.96, 0.05), 2):
    yp = (prob >= tau).astype(int)
    thr_rows.append(dict(tau=float(tau), precision=precision_score(yte, yp, zero_division=0),
                         recall=recall_score(yte, yp), f1=f1_score(yte, yp),
                         accuracy=accuracy_score(yte, yp)))

# error analysis (a handful of examples from each type)
te = pd.DataFrame({"review": Xte_t.values, "y": yte.values, "p": prob, "pred": pred})
fp = te[(te.pred == 1) & (te.y == 0)].assign(conf=lambda d: d.p).sort_values("conf", ascending=False)
fn = te[(te.pred == 0) & (te.y == 1)].assign(conf=lambda d: 1 - d.p).sort_values("conf", ascending=False)
clean = lambda s: re.sub(r"<br\s*/?>", " ", s)
errors = {
    "fp": [dict(text=clean(r.review), p=float(r.p)) for r in fp.head(6).itertuples()],
    "fn": [dict(text=clean(r.review), p=float(r.p)) for r in fn.head(6).itertuples()],
}

# dataset facts + a small browsable sample
sample = df.sample(400, random_state=SEED)[["review", "sentiment", "n_words"]]
sample["review"] = sample["review"].map(clean)
data_info = dict(
    n=len(df), counts=df["sentiment"].value_counts().to_dict(),
    len_hist={k: np.histogram(df.loc[df.sentiment == k, "n_words"].clip(upper=1000),
                              bins=40, range=(0, 1000)) for k in ["positive", "negative"]},
    mean_len=float(df.n_words.mean()), median_len=float(df.n_words.median()),
    sparsity=float(100 * (1 - Xtr.nnz / (Xtr.shape[0] * Xtr.shape[1]))),
    sample=sample.reset_index(drop=True))

artifacts = dict(
    vectorizer=vec, model=model, best_C=best_C, metrics=metrics,
    grid=dict(C=[0.01, 0.1, 1, 10], acc=grid.cv_results_["mean_test_score"].tolist()),
    train_losses=train_losses,
    cm=confusion_matrix(yte, pred), roc=dict(fpr=fpr[::step], tpr=tpr[::step]),
    thresholds=thr_rows, errors=errors, data=data_info,
    split=dict(train=len(ytr), test=len(yte)), seed=SEED)
joblib.dump(artifacts, "artifacts.joblib", compress=3)
print(f"Done in {time.time()-t0:.0f}s | acc={metrics['accuracy']:.4f} auc={metrics['auc']:.4f}")
