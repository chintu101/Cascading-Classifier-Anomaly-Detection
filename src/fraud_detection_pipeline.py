"""
Sequential Machine Learning Pipeline for Financial Fraud Detection
==================================================================
Architecture: 3-stage sequential pipeline where transactions are evaluated
progressively, with only suspicious cases escalating to deeper analysis.

Stage 1 → Logistic Regression  (fast, lightweight filter)
Stage 2 → Isolation Forest     (unsupervised anomaly detection)
Stage 3 → Gradient Boosting    (powerful final classifier)

Python 3.13 compatibility notes:
  - 'from __future__ import annotations' enables modern type hint syntax on all 3.x
  - round() used instead of bare np.arrange float iteration to avoid drift
  - imbalanced-learn 0.12+ required for Python 3.13 support
"""

from __future__ import annotations  # enables 'list[dict]', 'tuple[...]' hints on all Python 3.x

import os
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count())

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# SMOTE is optional — pipeline falls back to class_weight="balanced" if unavailable
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Note: imbalanced-learn not installed. Using class_weight='balanced' instead of SMOTE.")


# ─────────────────────────────────────────────────────────────────────────────
# 1. SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_fraud_dataset(
    n_samples: int = 10_000,
    fraud_ratio: float = 0.02,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic credit-card-like dataset.

    Uses sklearn's make_classification to produce 20 named features that
    mimic real transaction signals (amount, velocity, device fingerprint, etc.).
    ~2% fraud rate mirrors real-world class imbalance.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=2,
        weights=[1 - fraud_ratio, fraud_ratio],
        flip_y=0.01,       # 1% label noise — prevents models from learning perfectly
        random_state=random_state,
    )

    feature_names = [
        "amount", "hour", "day_of_week", "merchant_category",
        "distance_from_home", "prev_txn_amount", "velocity_1h",
        "velocity_24h", "card_age_days", "country_mismatch",
        "device_fingerprint", "ip_risk_score", "billing_zip_match",
        "cvv_match", "pin_used", "contactless", "recurring",
        "high_risk_merchant", "large_amount_flag", "unusual_time",
    ]

    df = pd.DataFrame(X, columns=feature_names)
    df["fraud"] = y

    print(
        f"Dataset: {len(df):,} transactions  |  "
        f"Fraud: {y.sum():,} ({y.mean() * 100:.1f}%)  |  "
        f"Legitimate: {(1 - y).sum():,} ({(1 - y).mean() * 100:.1f}%)"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. PIPELINE STAGES
# ─────────────────────────────────────────────────────────────────────────────

class Stage1_LightweightClassifier:
    """
    Logistic Regression — fast probabilistic filter.

    Assigns a fraud probability to each transaction.
    - Confident legitimates  (prob < low_threshold)  → exit as Legitimate
    - Confident frauds       (prob > high_threshold)  → exit as Fraud
    - Uncertain grey zone                             → escalate to Stage 2
    """

    def __init__(self, low_threshold: float = 0.15, high_threshold: float = 0.70):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            class_weight="balanced",   # compensates for class imbalance
            max_iter=1000,
            random_state=42,
            C=1.0,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> Stage1_LightweightClassifier:
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict_proba_scaled(self, X: np.ndarray) -> np.ndarray:
        """Return the fraud probability (class 1) for each transaction."""
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]

    def evaluate(self, X: np.ndarray) -> list[dict]:
        """
        Return per-transaction decision dicts.
        Each dict has keys: 'decision' ∈ {'legitimate', 'fraud', 'escalate'},
        'score' (raw probability), and 'stage' (always 1 here).
        """
        probs = self.predict_proba_scaled(X)
        results = []
        for prob in probs:
            if prob < self.low_threshold:
                results.append({"decision": "legitimate", "score": prob, "stage": 1})
            elif prob > self.high_threshold:
                results.append({"decision": "fraud", "score": prob, "stage": 1})
            else:
                results.append({"decision": "escalate", "score": prob, "stage": 1})
        return results


class Stage2_AnomalyDetection:
    """
    Isolation Forest — unsupervised anomaly scorer.

    Does NOT require labels. Measures how easy it is to isolate a transaction
    from the rest of the data. Anomalous (potentially fraudulent) transactions
    are isolated quickly, yielding more negative decision function scores.

    Score interpretation:
        closer to  +0.5  →  normal
        closer to  -0.5  →  anomalous

    - Clear normals  (score > low_threshold)  → exit as Legitimate
    - Clear outliers (score < high_threshold) → exit as Fraud
    - Ambiguous                               → escalate to Stage 3
    """

    def __init__(self, low_threshold: float = 0.02, high_threshold: float = -0.10):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=200,
            contamination=0.02,   # prior: ~2% of transactions are anomalous
            random_state=42,
            n_jobs=os.cpu_count(),
        )

    def fit(self, X: np.ndarray) -> Stage2_AnomalyDetection:
        """Isolation Forest is unsupervised — no labels needed."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Return raw decision function score (higher = more normal)."""
        return self.model.decision_function(self.scaler.transform(X))

    def evaluate(self, X: np.ndarray) -> list[dict]:
        scores = self.anomaly_score(X)
        results = []
        for score in scores:
            if score > self.low_threshold:
                results.append({"decision": "legitimate", "score": score, "stage": 2})
            elif score < self.high_threshold:
                results.append({"decision": "fraud", "score": score, "stage": 2})
            else:
                results.append({"decision": "escalate", "score": score, "stage": 2})
        return results


class Stage3_PowerfulClassifier:
    """
    Gradient Boosting — high-capacity final arbiter.

    Only receives the ambiguous minority that survived Stages 1 and 2.
    Trained specifically on grey-zone samples so it specialises in hard cases.
    Uses SMOTE if available to further balance the grey-zone training set.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> Stage3_PowerfulClassifier:
        X_scaled = self.scaler.fit_transform(X)

        # Apply SMOTE if available and enough minority samples exist
        if SMOTE_AVAILABLE and y.sum() >= 6:
            smote = SMOTE(random_state=42, k_neighbors=min(5, int(y.sum()) - 1))
            try:
                X_scaled, y = smote.fit_resample(X_scaled, y)
            except ValueError:
                pass   # fallback to unbalanced fit — GBM handles it reasonably

        self.model.fit(X_scaled, y)
        return self

    def evaluate(self, X: np.ndarray) -> list[dict]:
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        probs = self.model.predict_proba(X_scaled)[:, 1]
        results = []
        for pred, prob in zip(preds, probs):
            decision = "fraud" if pred == 1 else "legitimate"
            results.append({"decision": decision, "score": prob, "stage": 3})
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. SEQUENTIAL PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class SequentialFraudPipeline:
    """
    Orchestrates the 3-stage sequential pipeline.

    Training flow:
        1. Stage 1 (LR) trained on full training set.
        2. Stage 2 (IsoForest) trained on full training set (unsupervised).
        3. Stage 3 (GBM) trained ONLY on grey-zone samples — those that
           would have been escalated through both Stage 1 and Stage 2.

    Inference flow:
        Transaction → Stage 1 → [exit] or → Stage 2 → [exit] or → Stage 3 → exit
    """

    def __init__(
        self,
        s1_low: float = 0.15,
        s1_high: float = 0.70,
        s2_low: float = 0.02,
        s2_high: float = -0.10,
    ):
        self.stage1 = Stage1_LightweightClassifier(s1_low, s1_high)
        self.stage2 = Stage2_AnomalyDetection(s2_low, s2_high)
        self.stage3 = Stage3_PowerfulClassifier()
        self.stage_counts: dict[int, int] = {1: 0, 2: 0, 3: 0}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> SequentialFraudPipeline:
        print("\n── Training Sequential Pipeline ──")

        print("  [Stage 1] Fitting Logistic Regression ...", end=" ")
        self.stage1.fit(X_train, y_train)
        print("done")

        print("  [Stage 2] Fitting Isolation Forest ...", end=" ")
        self.stage2.fit(X_train)   # unsupervised — no labels needed
        print("done")

        # Identify which training samples would reach Stage 3
        print("  [Stage 3] Identifying grey-zone training samples for GBM ...")
        s1_results = self.stage1.evaluate(X_train)
        escalated_after_s1 = np.array([r["decision"] == "escalate" for r in s1_results])

        X_after_s1 = X_train[escalated_after_s1]
        y_after_s1 = y_train[escalated_after_s1]

        if len(X_after_s1) > 0:
            s2_results = self.stage2.evaluate(X_after_s1)
            escalated_after_s2 = np.array([r["decision"] == "escalate" for r in s2_results])
            X_grey_zone = X_after_s1[escalated_after_s2]
            y_grey_zone = y_after_s1[escalated_after_s2]
        else:
            X_grey_zone = X_after_s1
            y_grey_zone = y_after_s1

        print(f"  [Stage 3] Fitting GradientBoosting on {len(X_grey_zone):,} grey-zone samples ...")

        if len(X_grey_zone) > 0 and len(np.unique(y_grey_zone)) > 1:
            self.stage3.fit(X_grey_zone, y_grey_zone)
        elif len(X_after_s1) > 0 and len(np.unique(y_after_s1)) > 1:
            # Fallback: not enough grey-zone data — train on all Stage 1 escalations
            self.stage3.fit(X_after_s1, y_after_s1)
        else:
            # Last resort: train on the full training set
            self.stage3.fit(X_train, y_train)

        print("  Pipeline training complete.\n")
        return self

    def predict(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Route each transaction through the pipeline.

        Returns:
            final_labels  — predicted class (0 = legitimate, 1 = fraud)
            final_stages  — which stage made the decision (1, 2, or 3)
        """
        n = len(X_test)
        final_labels = np.full(n, -1, dtype=int)   # -1 = not yet decided
        final_stages = np.zeros(n, dtype=int)

        # ── Stage 1 ──────────────────────────────────────────────────────────
        s1_results = self.stage1.evaluate(X_test)
        escalate_to_s2: list[int] = []

        for i, res in enumerate(s1_results):
            if res["decision"] == "legitimate":
                final_labels[i] = 0
                final_stages[i] = 1
            elif res["decision"] == "fraud":
                final_labels[i] = 1
                final_stages[i] = 1
            else:
                escalate_to_s2.append(i)

        self.stage_counts[1] = n - len(escalate_to_s2)

        if not escalate_to_s2:
            return final_labels, final_stages

        # ── Stage 2 ──────────────────────────────────────────────────────────
        X_s2 = X_test[escalate_to_s2]
        s2_results = self.stage2.evaluate(X_s2)
        escalate_to_s3: list[int] = []

        for orig_idx, res in zip(escalate_to_s2, s2_results):
            if res["decision"] == "legitimate":
                final_labels[orig_idx] = 0
                final_stages[orig_idx] = 2
            elif res["decision"] == "fraud":
                final_labels[orig_idx] = 1
                final_stages[orig_idx] = 2
            else:
                escalate_to_s3.append(orig_idx)

        self.stage_counts[2] = len(escalate_to_s2) - len(escalate_to_s3)

        if not escalate_to_s3:
            self.stage_counts[3] = 0
            return final_labels, final_stages

        # ── Stage 3 ──────────────────────────────────────────────────────────
        X_s3 = X_test[escalate_to_s3]
        s3_results = self.stage3.evaluate(X_s3)

        for orig_idx, res in zip(escalate_to_s3, s3_results):
            final_labels[orig_idx] = 1 if res["decision"] == "fraud" else 0
            final_stages[orig_idx] = 3

        self.stage_counts[3] = len(escalate_to_s3)
        return final_labels, final_stages


# ─────────────────────────────────────────────────────────────────────────────
# 4. BASELINE MODELS
# ─────────────────────────────────────────────────────────────────────────────

class SingleModelBaseline:
    """
    Simple Logistic Regression applied to all transactions.
    Represents the minimal production-grade approach: one model, all data.
    """

    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> SingleModelBaseline:
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class ParallelEnsembleBaseline:
    """
    All three models (LR + Random Forest + GBM) run in parallel.
    Final decision is a majority vote (≥ 2 of 3 predict fraud → Fraud).

    Every transaction hits every model — no routing, no compute savings.
    This simulates the traditional ensemble approach.
    """

    def __init__(self):
        self.lr = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ])
        self.rf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
            )),
        ])
        self.gb = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> ParallelEnsembleBaseline:
        print("  Fitting LR ...", end=" "); self.lr.fit(X, y); print("done")
        print("  Fitting RF ...", end=" "); self.rf.fit(X, y); print("done")
        print("  Fitting GB ...", end=" "); self.gb.fit(X, y); print("done")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        vote_sum = (
            self.lr.predict(X).astype(int)
            + self.rf.predict(X).astype(int)
            + self.gb.predict(X).astype(int)
        )
        return (vote_sum >= 2).astype(int)   # majority vote threshold


# ─────────────────────────────────────────────────────────────────────────────
# 5. THRESHOLD TUNER
# ─────────────────────────────────────────────────────────────────────────────

def tune_stage1_thresholds(
    stage1: Stage1_LightweightClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
    target_recall: float = 0.90,
) -> tuple[float, float]:
    """
    Grid-search Stage 1 thresholds on the validation set.

    Optimisation objective:
        Maximise fraud recall on decided transactions,
        subject to escalation_rate ≤ 45%.

    The escalation rate cap ensures the pipeline stays efficient —
    too many escalations defeats the purpose of early exit.

    Returns:
        (best_low_threshold, best_high_threshold)
    """
    print("\n── Threshold Tuning (Stage 1) ──")
    probs = stage1.predict_proba_scaled(X_val)

    best = {"recall": 0.0, "low": 0.15, "high": 0.70, "esc_rate": 1.0}

    # Use round() to avoid floating-point accumulation errors from np.arange
    low_values  = [round(v, 2) for v in np.arange(0.05, 0.40, 0.05).tolist()]
    high_values = [round(v, 2) for v in np.arange(0.50, 0.95, 0.05).tolist()]

    for low in low_values:
        for high in high_values:
            if high <= low:
                continue

            preds = np.where(probs < low, 0, np.where(probs > high, 1, -1))
            decided_mask = preds != -1
            escalation_rate = (~decided_mask).mean()

            if escalation_rate > 0.45:          # reject configurations that escalate too much
                continue
            if decided_mask.sum() == 0:
                continue

            rec = recall_score(y_val[decided_mask], preds[decided_mask], zero_division=0)

            if rec >= target_recall and escalation_rate < best["esc_rate"]:
                best = {"recall": rec, "low": low, "high": high, "esc_rate": escalation_rate}

    print(
        f"  Best thresholds → low={best['low']:.2f}, high={best['high']:.2f} "
        f"| recall={best['recall']:.3f}, escalation_rate={best['esc_rate'] * 100:.1f}%"
    )
    return best["low"], best["high"]


# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUATION & REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    """Return a dict of classification metrics for a given model."""
    return {
        "model":     name,
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }


def print_results_table(results: list[dict]) -> None:
    header = f"{'Model':<28} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}"
    print("\n" + "=" * 68)
    print(header)
    print("-" * 68)
    for r in results:
        print(
            f"{r['model']:<28} "
            f"{r['accuracy']:>9.4f} "
            f"{r['precision']:>10.4f} "
            f"{r['recall']:>8.4f} "
            f"{r['f1']:>8.4f}"
        )
    print("=" * 68)


def print_pipeline_efficiency(stage_counts: dict, total: int) -> None:
    """Print stage exit distribution and compute savings vs parallel ensemble."""
    print("\n── Sequential Pipeline — Stage Exit Distribution ──")
    stage_labels = {
        1: "Stage 1 (LR)     ",
        2: "Stage 2 (IsoFrst)",
        3: "Stage 3 (GBM)    ",
    }
    for stage, count in stage_counts.items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {stage_labels[stage]}: {count:>5,} txns ({pct:5.1f}%)  {bar}")

    sequential_invocations = (
        stage_counts[1] * 1    # only Stage 1
        + stage_counts[2] * 2  # Stages 1 + 2
        + stage_counts[3] * 3  # all 3 stages
    )
    parallel_invocations = total * 3   # every model sees every transaction
    savings = 1 - sequential_invocations / parallel_invocations

    print(f"\n  Model invocations (sequential) : {sequential_invocations:,}")
    print(f"  Model invocations (parallel)   : {parallel_invocations:,}")
    print(f"  Compute savings                : {savings * 100:.1f}%  ← efficiency gain")


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  {name} confusion matrix")
    print(f"  ┌──────────────┬───────────┬──────────┐")
    print(f"  │              │ Pred Legit│ Pred Fraud│")
    print(f"  ├──────────────┼───────────┼──────────┤")
    print(f"  │ True Legit   │   {tn:>7,} │   {fp:>6,} │")
    print(f"  │ True Fraud   │   {fn:>7,} │   {tp:>6,} │")
    print(f"  └──────────────┴───────────┴──────────┘")


def print_per_stage_breakdown(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    stage_assignments: np.ndarray,
) -> None:
    """Break down accuracy and recall for each stage's exit decisions."""
    print("\n── Per-Stage Decision Breakdown ──")
    for stage_id in [1, 2, 3]:
        mask = stage_assignments == stage_id
        if mask.sum() == 0:
            continue
        y_stage = y_test[mask]
        p_stage = y_pred[mask]
        print(
            f"  Stage {stage_id}: {mask.sum():>5,} decisions | "
            f"true fraud inside: {y_stage.sum():>4} | "
            f"accuracy: {accuracy_score(y_stage, p_stage):.4f} | "
            f"recall: {recall_score(y_stage, p_stage, zero_division=0):.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Sequential ML Pipeline for Financial Fraud Detection       ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    df = generate_fraud_dataset(n_samples=15_000, fraud_ratio=0.025)
    X = df.drop("fraud", axis=1).values
    y = df["fraud"].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    print(f"\nSplit → Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

    # ── Baseline 1: Single Model ──────────────────────────────────────────────
    print("\n── Training Single-Model Baseline (Logistic Regression) ──")
    single = SingleModelBaseline().fit(X_train, y_train)
    y_pred_single = single.predict(X_test)

    # ── Baseline 2: Parallel Ensemble ─────────────────────────────────────────
    print("\n── Training Parallel Ensemble Baseline ──")
    ensemble = ParallelEnsembleBaseline().fit(X_train, y_train)
    y_pred_ensemble = ensemble.predict(X_test)

    # ── Sequential Pipeline ────────────────────────────────────────────────────
    pipeline = SequentialFraudPipeline().fit(X_train, y_train)

    # Tune Stage 1 thresholds on validation set, then apply to pipeline
    low_t, high_t = tune_stage1_thresholds(pipeline.stage1, X_val, y_val)
    pipeline.stage1.low_threshold  = low_t
    pipeline.stage1.high_threshold = high_t

    print("\n── Running Sequential Pipeline on Test Set ──")
    y_pred_seq, stage_assignments = pipeline.predict(X_test)

    # ── Metrics ────────────────────────────────────────────────────────────────
    results = [
        compute_metrics(y_test, y_pred_single,   "Single Model (LR)"),
        compute_metrics(y_test, y_pred_ensemble, "Parallel Ensemble"),
        compute_metrics(y_test, y_pred_seq,      "Sequential Pipeline"),
    ]
    print_results_table(results)

    # ── Confusion Matrices ─────────────────────────────────────────────────────
    print_confusion_matrix(y_test, y_pred_single,   "Single Model (LR)")
    print_confusion_matrix(y_test, y_pred_ensemble, "Parallel Ensemble")
    print_confusion_matrix(y_test, y_pred_seq,      "Sequential Pipeline")

    # ── Efficiency Report ──────────────────────────────────────────────────────
    print_pipeline_efficiency(pipeline.stage_counts, len(X_test))

    # ── Per-Stage Breakdown ────────────────────────────────────────────────────
    print_per_stage_breakdown(y_test, y_pred_seq, stage_assignments)

    # ── Summary ───────────────────────────────────────────────────────────────
    seq_m    = results[2]
    ens_m    = results[1]
    single_m = results[0]

    savings_pct = (1 - (
        (pipeline.stage_counts[1] * 1
         + pipeline.stage_counts[2] * 2
         + pipeline.stage_counts[3] * 3)
        / (len(X_test) * 3)
    )) * 100

    print("\n── Summary ──────────────────────────────────────────────────────────")
    print(
        f"  Sequential pipeline  →  F1={seq_m['f1']:.4f} | "
        f"Recall={seq_m['recall']:.4f} | Precision={seq_m['precision']:.4f}"
    )
    print(
        f"  Parallel ensemble    →  F1={ens_m['f1']:.4f} | "
        f"Recall={ens_m['recall']:.4f} | Precision={ens_m['precision']:.4f}"
    )
    print(
        f"  Single model (LR)    →  F1={single_m['f1']:.4f} | "
        f"Recall={single_m['recall']:.4f} | Precision={single_m['precision']:.4f}"
    )
    print(
        f"\n  With ~{savings_pct:.0f}% fewer model invocations, the sequential pipeline "
        f"reserves expensive models for genuinely ambiguous transactions."
    )
    print()


if __name__ == "__main__":
    main()
