"""
Classical ML baselines for PIGuard (ACL 2025), trained and tested on the same data.

Models
  tfidf_word_lr        word 1-2 gram TF-IDF + Logistic Regression
  tfidf_word_svm       word 1-2 gram TF-IDF + Linear SVM
  tfidf_char_lr        char 2-5 gram TF-IDF + Logistic Regression
  tfidf_word_lr_noMOF  tfidf_word_lr trained without the 762 MOF samples
                       (train.json source == "over-defense"), ablation
  piguard              leolee99/PIGuard from Hugging Face (only with --piguard,
                       needs transformers + internet, e.g. Google Colab)

Evaluation follows PIGuard/eval.py, PINT excluded (not public):
  Over-defense ACC = mean(NotInject one, two, three)   benign sets
  Benign ACC       = WildGuard                         benign set
  Injection ACC    = mean(BIPIA text, BIPIA code)      injection sets
  Average          = mean of the three
Added on top of the original protocol
  - Over-defense ACC on English-only NotInject (Multilingual category removed)
  - pooled set (benign = WildGuard + NotInject, injection = BIPIA):
    precision, recall, F1, FPR, ROC-AUC
  - 95% Wilson interval for every per-set accuracy
  - in-distribution accuracy on a 10% held-out split of train.json
  - token-wise recheck: every NotInject trigger word fed alone
  - for TF-IDF models: number of test inputs with no known token

Usage
  python run_baselines.py --data PIGuard/datasets --out results/Baseline_Results.json
  python run_baselines.py --data PIGuard/datasets --out results/Baseline_Results.json --piguard
"""
import argparse
import json
import math
import os
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

SEED = 42
NOTINJECT = ["NotInject_one", "NotInject_two", "NotInject_three"]


# ---------------------------------------------------------------- data
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tests(root):
    """name -> (texts, gold label, raw records or None). 0 = benign, 1 = injection."""
    tests = {}
    for n in ["one", "two", "three"]:
        d = load_json(os.path.join(root, f"NotInject_{n}.json"))
        tests[f"NotInject_{n}"] = ([s["prompt"] for s in d], 0, d)
    wg = load_json(os.path.join(root, "wildguard.json"))
    tests["WildGuard"] = ([s["prompt"] for s in wg], 0, None)
    for n in ["text", "code"]:
        d = load_json(os.path.join(root, f"BIPIA_{n}.json"))
        tests[f"BIPIA_{n}"] = ([t for v in d.values() for t in v], 1, None)
    return tests


# ---------------------------------------------------------------- models
def make_sklearn(name):
    word = dict(ngram_range=(1, 2), min_df=2, max_features=200000,
                sublinear_tf=True, lowercase=True)
    char = dict(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                max_features=300000, sublinear_tf=True, lowercase=True)
    lr = dict(max_iter=2000, C=4.0, random_state=SEED)
    if name in ("tfidf_word_lr", "tfidf_word_lr_noMOF"):
        return Pipeline([("tfidf", TfidfVectorizer(**word)),
                         ("clf", LogisticRegression(**lr))])
    if name == "tfidf_word_svm":
        return Pipeline([("tfidf", TfidfVectorizer(**word)),
                         ("clf", LinearSVC(C=0.5, random_state=SEED))])
    if name == "tfidf_char_lr":
        return Pipeline([("tfidf", TfidfVectorizer(**char)),
                         ("clf", LogisticRegression(**lr))])
    raise ValueError(name)


class SklearnModel:
    def __init__(self, pipe):
        self.pipe = pipe

    def predict_scores(self, texts):
        """Returns (hard label, continuous injection score)."""
        clf = self.pipe.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            s = self.pipe.predict_proba(texts)[:, 1]
            return (s >= 0.5).astype(int), s
        s = self.pipe.decision_function(texts)
        return (s > 0).astype(int), s

    def n_unknown(self, texts):
        x = self.pipe.named_steps["tfidf"].transform(texts)
        return int((np.asarray((x != 0).sum(1)).ravel() == 0).sum())


class PIGuardHF:
    """Same loading as PIGuard/eval_hf.py. Not runnable without HF access."""

    def __init__(self, batch_size=32):
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer, pipeline)
        tok = AutoTokenizer.from_pretrained("leolee99/PIGuard", model_max_length=2048)
        model = AutoModelForSequenceClassification.from_pretrained(
            "leolee99/PIGuard", trust_remote_code=True)
        device = 0 if torch.cuda.is_available() else -1
        self.clf = pipeline("text-classification", model=model, tokenizer=tok,
                            truncation=True, device=device, top_k=None)
        self.bs = batch_size

    def predict_scores(self, texts):
        scores = []
        for i in range(0, len(texts), self.bs):
            for out in self.clf(list(texts[i:i + self.bs])):
                scores.append({d["label"]: d["score"] for d in out}["injection"])
        s = np.array(scores)
        return (s >= 0.5).astype(int), s


# ---------------------------------------------------------------- metrics
def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 2), round(100 * (c + h), 2)]


def evaluate(model, tests):
    per_set, ci, cache = {}, {}, {}
    for name, (texts, label, _) in tests.items():
        pred, score = model.predict_scores(texts)
        cache[name] = (pred, score)
        k = int((pred == label).sum())
        per_set[name] = round(100 * k / len(texts), 2)
        ci[name] = wilson(k, len(texts))

    # English-only NotInject (drop the Multilingual category)
    eng = []
    by_cat = {}
    for name in NOTINJECT:
        raw = tests[name][2]
        pred = cache[name][0]
        keep = [i for i, s in enumerate(raw) if s["category"] != "Multilingual"]
        eng.append(100 * float(np.mean(pred[keep] == 0)))
        for s, p in zip(raw, pred):
            a = by_cat.setdefault(s["category"], [0, 0])
            a[0] += int(p == 0)
            a[1] += 1

    summary = {
        "Over-defense ACC (NotInject)": round(float(np.mean([per_set[n] for n in NOTINJECT])), 2),
        "Benign ACC (WildGuard)": per_set["WildGuard"],
        "Injection ACC (BIPIA)": round((per_set["BIPIA_text"] + per_set["BIPIA_code"]) / 2, 2),
    }
    summary["Average"] = round(float(np.mean(list(summary.values()))), 2)
    summary["Over-defense ACC (NotInject, English only)"] = round(float(np.mean(eng)), 2)

    labels = np.concatenate([[tests[n][1]] * len(tests[n][0]) for n in tests])
    pred = np.concatenate([cache[n][0] for n in tests])
    score = np.concatenate([cache[n][1] for n in tests])
    benign = labels == 0
    pooled = {
        "n_benign": int(benign.sum()),
        "n_injection": int((~benign).sum()),
        "accuracy": round(100 * accuracy_score(labels, pred), 2),
        "precision": round(100 * precision_score(labels, pred, zero_division=0), 2),
        "recall": round(100 * recall_score(labels, pred, zero_division=0), 2),
        "f1": round(100 * f1_score(labels, pred, zero_division=0), 2),
        "fpr": round(100 * float(np.mean(pred[benign] == 1)), 2),
        "roc_auc": round(100 * roc_auc_score(labels, score), 2),
    }
    by_cat = {c: round(100 * a / b, 2) for c, (a, b) in by_cat.items()}
    return per_set, ci, summary, pooled, by_cat


def token_recheck(model, tests):
    words = sorted({w for n in NOTINJECT for s in tests[n][2] for w in s["word_list"]})
    pred, _ = model.predict_scores(words)
    flagged = [w for w, p in zip(words, pred) if p == 1]
    return {"n_trigger_words": len(words),
            "flagged_as_injection": len(flagged),
            "flagged_ratio": round(100 * len(flagged) / len(words), 2),
            "flagged_words": flagged}


def top_features(pipe, k=25):
    names = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    coef = pipe.named_steps["clf"].coef_[0]
    idx = np.argsort(coef)[::-1][:k]
    return [[str(names[i]), round(float(coef[i]), 3)] for i in idx]


def report(model, tests):
    per_set, ci, summary, pooled, by_cat = evaluate(model, tests)
    entry = {"summary": summary, "per_set": per_set, "per_set_95ci": ci,
             "pooled": pooled, "notinject_by_category": by_cat,
             "token_recheck": token_recheck(model, tests)}
    if isinstance(model, SklearnModel):
        entry["inputs_with_no_known_token"] = {
            n: model.n_unknown(t) for n, (t, _, _) in tests.items()}
    return entry


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="PIGuard/datasets")
    ap.add_argument("--out", default="results/Baseline_Results.json")
    ap.add_argument("--piguard", action="store_true",
                    help="also evaluate leolee99/PIGuard with the same metrics")
    args = ap.parse_args()

    train = load_json(os.path.join(args.data, "train.json"))
    tests = load_tests(args.data)
    X = [s["prompt"] for s in train]
    y = np.array([s["label"] for s in train])
    src = np.array([s["source"] for s in train])
    idx_tr, idx_ho = train_test_split(np.arange(len(X)), test_size=0.1,
                                      stratify=y, random_state=SEED)

    results = {"protocol": "PIGuard eval.py, PINT excluded", "seed": SEED,
               "train_size": len(X), "models": {}}

    for name in ["tfidf_word_lr", "tfidf_word_svm", "tfidf_char_lr", "tfidf_word_lr_noMOF"]:
        t0 = time.time()
        keep = src != "over-defense" if name.endswith("noMOF") else np.ones(len(X), bool)

        pipe = make_sklearn(name)  # in-distribution check: fit on 90%, test on 10%
        tr = idx_tr[keep[idx_tr]]
        pipe.fit([X[i] for i in tr], y[tr])
        id_acc = 100 * accuracy_score(y[idx_ho], pipe.predict([X[i] for i in idx_ho]))

        pipe = make_sklearn(name)  # final model: full training set, like PIGuard
        full = np.where(keep)[0]
        pipe.fit([X[i] for i in full], y[full])

        entry = {"train_size": int(keep.sum()),
                 "in_distribution_acc": round(id_acc, 2)}
        entry.update(report(SklearnModel(pipe), tests))
        if name in ("tfidf_word_lr", "tfidf_word_lr_noMOF"):
            entry["top_injection_features"] = top_features(pipe)
        entry["fit_seconds"] = round(time.time() - t0, 1)
        results["models"][name] = entry
        print(name, json.dumps(entry["summary"]), json.dumps(entry["pooled"]), flush=True)

    if args.piguard:
        entry = report(PIGuardHF(), tests)
        results["models"]["piguard"] = entry
        print("piguard", json.dumps(entry["summary"]), json.dumps(entry["pooled"]), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("saved", args.out)


if __name__ == "__main__":
    main()
