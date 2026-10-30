import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

st.set_page_config(
    page_title="LeakCheck | AI Privacy Auditor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------ Styling --------------------------------------
st.markdown(
    """
    <style>
    .stApp { background: #0b1020; color: #eef2ff; }
    [data-testid="stSidebar"] { background: #111a30; }
    /* Keep sidebar labels readable against the dark sidebar background. */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
      color: #eef2ff !important;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
      color: #b9c7e6 !important;
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {
      color: #eef2ff !important;
    }
    h1, h2, h3 { color: #f4f7ff; }
    .hero {
      padding: 1.5rem 1.7rem; border-radius: 18px;
      background: linear-gradient(120deg, #16284a 0%, #25204b 100%);
      border: 1px solid #34466d; margin-bottom: 1rem;
    }
    .hero p { color: #c8d4f2; font-size: 1rem; }
    .small-note { color: #aab8d7; font-size: 0.88rem; }
    div[data-testid="stMetric"] {
      background: #131e35; padding: 1rem; border: 1px solid #2a3b5e;
      border-radius: 14px;
    }
    .pill {
      display: inline-block; background: #202d4d; color: #cbd7ff;
      border: 1px solid #3a4c75; border-radius: 20px;
      padding: 4px 10px; font-size: .8rem; margin-right: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------- Experiment logic --------------------------------
def prediction_features(model, X, y):
    """Create confidence-based features commonly studied in membership attacks."""
    probs = model.predict_proba(X)
    class_order = list(model.classes_)
    class_to_index = {cls: i for i, cls in enumerate(class_order)}
    true_class_conf = np.array([probs[i, class_to_index[label]] for i, label in enumerate(y)])
    sorted_probs = np.sort(probs, axis=1)
    max_conf = sorted_probs[:, -1]
    second_conf = sorted_probs[:, -2] if probs.shape[1] > 1 else np.zeros(len(probs))
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
    return np.column_stack([true_class_conf, max_conf, entropy, max_conf - second_conf])


def make_synthetic_data(n_samples, n_features, random_state):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, min(n_features - 1, int(n_features * 0.65))),
        n_redundant=max(0, min(2, n_features - 2)),
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        weights=[0.5, 0.5],
        flip_y=0.02,
        class_sep=1.0,
        random_state=random_state,
    )
    return X, y


def train_target_model(X_train, y_train, mitigated=False, seed=42):
    if mitigated:
        # A deliberately less complex model for demonstrating one mitigation.
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=8,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_leaf=1,
            max_features=None,
            random_state=seed,
            n_jobs=-1,
        )
    model.fit(X_train, y_train)
    return model


def train_shadow_attack(n_features, n_shadow_models, samples_per_shadow, seed):
    shadow_features = []
    shadow_membership = []

    for i in range(n_shadow_models):
        Xs, ys = make_synthetic_data(
            samples_per_shadow, n_features, random_state=seed + 100 + i
        )
        X_in, X_out, y_in, y_out = train_test_split(
            Xs, ys, test_size=0.5, random_state=seed + 200 + i, stratify=ys
        )
        shadow = RandomForestClassifier(
            n_estimators=70,
            max_depth=None,
            min_samples_leaf=1,
            random_state=seed + 300 + i,
            n_jobs=-1,
        )
        shadow.fit(X_in, y_in)
        in_features = prediction_features(shadow, X_in, y_in)
        out_features = prediction_features(shadow, X_out, y_out)
        shadow_features.extend([in_features, out_features])
        shadow_membership.extend([
            np.ones(len(in_features), dtype=int),
            np.zeros(len(out_features), dtype=int),
        ])

    X_attack = np.vstack(shadow_features)
    y_attack = np.concatenate(shadow_membership)
    attack_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )
    attack_model.fit(X_attack, y_attack)
    return attack_model


def evaluate_attack(target_model, X_member, y_member, X_nonmember, y_nonmember, attack_model):
    f_member = prediction_features(target_model, X_member, y_member)
    f_nonmember = prediction_features(target_model, X_nonmember, y_nonmember)
    X_eval = np.vstack([f_member, f_nonmember])
    y_eval = np.concatenate([
        np.ones(len(f_member), dtype=int),
        np.zeros(len(f_nonmember), dtype=int),
    ])
    scores = attack_model.predict_proba(X_eval)[:, 1]
    preds = (scores >= 0.5).astype(int)
    precision = precision_score(y_eval, preds, zero_division=0)
    recall = recall_score(y_eval, preds, zero_division=0)
    auc = roc_auc_score(y_eval, scores)
    # Defined educational index: AUC 0.50 => 0; AUC 1.00 => 100.
    # This is a demonstration score, not a certified industry standard.
    risk = float(np.clip((auc - 0.5) * 200, 0, 100))
    return {
        "Precision": float(precision),
        "Recall": float(recall),
        "ROC-AUC": float(auc),
        "Privacy Risk Score": risk,
        "Target Model Accuracy": float(accuracy_score(
            np.concatenate([y_member, y_nonmember]),
            np.concatenate([
                target_model.predict(X_member),
                target_model.predict(X_nonmember),
            ]),
        )),
        "Member samples": int(len(X_member)),
        "Non-member samples": int(len(X_nonmember)),
        "Attack predictions": preds,
        "Attack scores": scores,
        "True membership": y_eval,
    }


@st.cache_data(show_spinner=False, ttl=1800)
def run_experiment(n_samples=1400, n_features=10, n_shadows=3, seed=42):
    X, y = make_synthetic_data(n_samples, n_features, seed)
    X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )

    attack_model = train_shadow_attack(
        n_features=n_features,
        n_shadow_models=n_shadows,
        samples_per_shadow=max(500, n_samples // 2),
        seed=seed,
    )

    baseline = train_target_model(X_target_train, y_target_train, mitigated=False, seed=seed)
    mitigated = train_target_model(X_target_train, y_target_train, mitigated=True, seed=seed + 1)

    base_eval = evaluate_attack(
        baseline, X_target_train, y_target_train,
        X_target_test, y_target_test, attack_model
    )
    mitigated_eval = evaluate_attack(
        mitigated, X_target_train, y_target_train,
        X_target_test, y_target_test, attack_model
    )

    rows = []
    for name, result in [("Baseline model", base_eval), ("Mitigated model", mitigated_eval)]:
        rows.append({
            "Model": name,
            "Attack Precision": result["Precision"],
            "Attack Recall": result["Recall"],
            "Attack ROC-AUC": result["ROC-AUC"],
            "Privacy Risk Score (0-100)": result["Privacy Risk Score"],
            "Target Model Accuracy": result["Target Model Accuracy"],
            "Member Samples": result["Member samples"],
            "Non-member Samples": result["Non-member samples"],
        })
    results_df = pd.DataFrame(rows)
    return results_df, base_eval, mitigated_eval


def explain_risk(score):
    if score < 10:
        return "Low observed attack advantage in this simulation. This does not prove that the model is private."
    if score < 30:
        return "Some membership signal was observed. Review model confidence and test additional settings."
    if score < 60:
        return "Moderate membership signal was observed. Consider stronger regularization and broader testing."
    return "High membership signal was observed in this experiment. Review training choices and privacy safeguards."


# ------------------------------ Navigation -----------------------------------
with st.sidebar:
    st.markdown("## 🛡️ LeakCheck")
    st.caption("AI Model Privacy Auditor")
    page = st.radio(
        "Navigate",
        ["Overview", "Run Privacy Audit", "Results & Analytics", "Mitigation Guide", "Methodology & Limits"],
    )
    st.markdown("---")
    st.markdown("**Project**")
    st.caption("TY B.Tech AI&DS · Trustworthy and Responsible AI")
    st.caption("Educational prototype · Synthetic data only")

st.markdown(
    """
    <div class="hero">
      <span class="pill">AI SECURITY</span><span class="pill">PRIVACY AUDITING</span>
      <h1>LeakCheck</h1>
      <p>Auditing membership inference vulnerabilities in machine-learning models.</p>
      <p>Explore whether a model's predictions reveal clues about which examples were used during training.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "experiment_result" not in st.session_state:
    st.session_state.experiment_result = None

if page == "Overview":
    st.subheader("Privacy risk, made easier to understand")
    st.write(
        "A membership inference attack tries to guess whether a particular record was part of a model's training data. "
        "LeakCheck demonstrates this idea using a synthetic classification dataset and an attack model."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Audit approach", "Membership inference")
    c2.metric("Data used", "Synthetic")
    c3.metric("Comparison", "Baseline vs. mitigated")
    st.markdown("### How it works")
    steps = [
        ("01 · Train", "Train a target model on one part of the synthetic dataset."),
        ("02 · Learn attack patterns", "Train shadow models and use their outputs to train an attack classifier."),
        ("03 · Test membership", "Evaluate whether the attack can distinguish target training samples from held-out samples."),
        ("04 · Compare protections", "Compare a baseline model with a less complex, regularized model."),
    ]
    for title, description in steps:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(description)
    st.info("This is an educational simulation. Its risk score is a project-defined index, not an official certification.")

elif page == "Run Privacy Audit":
    st.subheader("Configure your experiment")
    st.write("The app generates artificial data. Do not upload real or confidential personal information.")
    col1, col2 = st.columns(2)
    with col1:
        n_samples = st.slider("Synthetic dataset size", 800, 2400, 1400, step=200)
        n_features = st.slider("Number of input features", 6, 16, 10)
    with col2:
        n_shadows = st.slider("Number of shadow models", 2, 5, 3)
        seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
    st.caption("More samples and shadow models may take longer to run.")
    if st.button("▶ Run Privacy Audit", type="primary", use_container_width=True):
        with st.spinner("Training target and shadow models, then evaluating the membership attack..."):
            try:
                results_df, base_eval, mitigated_eval = run_experiment(
                    int(n_samples), int(n_features), int(n_shadows), int(seed)
                )
                st.session_state.experiment_result = {
                    "df": results_df,
                    "baseline": base_eval,
                    "mitigated": mitigated_eval,
                    "settings": {
                        "samples": int(n_samples),
                        "features": int(n_features),
                        "shadow_models": int(n_shadows),
                        "seed": int(seed),
                    },
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                st.success("Experiment completed. Open Results & Analytics to review the measured results.")
            except Exception as exc:
                st.error("The experiment could not finish. Try the default settings and run it again.")
                st.exception(exc)
    st.markdown("### What the experiment measures")
    st.write("- **Precision:** Of records the attack calls members, how many really are members?")
    st.write("- **Recall:** Of all actual member records, how many does the attack identify?")
    st.write("- **ROC-AUC:** How well the attack separates members from non-members; 0.50 is chance-level for a balanced evaluation.")
    st.write("- **Privacy risk score:** A project-defined transformation of AUC, from 0 at AUC 0.50 to 100 at AUC 1.00.")

elif page == "Results & Analytics":
    st.subheader("Experiment results")
    result = st.session_state.experiment_result
    if result is None:
        st.warning("No experiment has been run in this session yet. Go to Run Privacy Audit and run the experiment first.")
    else:
        df = result["df"]
        base = result["baseline"]
        mitigated = result["mitigated"]
        st.caption(f"Last run: {result['timestamp']} · Settings: {result['settings']}")
        a, b, c, d = st.columns(4)
        a.metric("Baseline attack AUC", f"{base['ROC-AUC']:.3f}")
        b.metric("Mitigated attack AUC", f"{mitigated['ROC-AUC']:.3f}", delta=f"{mitigated['ROC-AUC']-base['ROC-AUC']:+.3f}", delta_color="inverse")
        c.metric("Baseline risk score", f"{base['Privacy Risk Score']:.1f}/100")
        d.metric("Mitigated risk score", f"{mitigated['Privacy Risk Score']:.1f}/100")
        st.markdown("### Metrics comparison")
        display_df = df.copy()
        numeric_cols = display_df.select_dtypes(include="number").columns
        display_df[numeric_cols] = display_df[numeric_cols].round(3)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        chart_df = df.set_index("Model")[[
            "Attack Precision", "Attack Recall", "Attack ROC-AUC", "Target Model Accuracy"
        ]]
        st.markdown("### Model and attack metrics")
        st.bar_chart(chart_df, use_container_width=True)
        st.markdown("### Interpretation")
        st.write("**Baseline:** " + explain_risk(base["Privacy Risk Score"]))
        st.write("**Mitigated:** " + explain_risk(mitigated["Privacy Risk Score"]))
        if mitigated["ROC-AUC"] < base["ROC-AUC"]:
            st.success("In this run, the attack's AUC decreased after mitigation. Check target accuracy too, because privacy and utility can trade off.")
        elif mitigated["ROC-AUC"] > base["ROC-AUC"]:
            st.warning("In this run, the attack's AUC increased after mitigation. This shows that a mitigation choice must be evaluated rather than assumed to work.")
        else:
            st.info("The attack AUC was unchanged in this run. Different model settings or repeated trials may produce different results.")
        st.download_button(
            "⬇ Download results as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="leakcheck_audit_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption("Metrics are calculated from the current synthetic experiment. Results may vary with settings and random seed.")

elif page == "Mitigation Guide":
    st.subheader("How can membership inference risk be reduced?")
    recommendations = [
        ("Reduce overfitting", "Use regularization, sensible model complexity, and validation data. Models that memorize training samples may reveal stronger membership signals."),
        ("Dropout (for neural networks)", "Randomly disables some units during training to help reduce overfitting. This prototype uses a shallower random forest as its mitigation comparison, not dropout."),
        ("Limit unnecessary confidence exposure", "Avoid exposing detailed probability vectors to users when the application only needs a predicted class. This can reduce some attack signals, but does not guarantee privacy."),
        ("Use privacy-preserving training when needed", "Differential privacy methods can provide formal privacy guarantees when implemented and configured correctly."),
        ("Test more than once", "Repeat experiments across random seeds, datasets and model settings. One result is not a universal security guarantee."),
    ]
    for title, body in recommendations:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(body)
    st.warning("A lower attack score in one simulation does not prove that real-world privacy leakage has been eliminated.")

elif page == "Methodology & Limits":
    st.subheader("Methodology")
    st.markdown("**Target model:** A random forest trained on a synthetic binary classification dataset.")
    st.markdown("**Shadow models:** Models trained on separate synthetic samples. Their outputs create labelled member and non-member examples for the attack classifier.")
    st.markdown("**Attack model:** Logistic regression using four output features: confidence in the true class, maximum confidence, prediction entropy and the margin between the two highest class probabilities.")
    st.markdown("**Evaluation:** The attack is tested on member and non-member examples from the target model. Precision, recall and ROC-AUC are computed from these predictions.")
    st.markdown("**Mitigation comparison:** A baseline random forest is compared with a shallower forest that uses a larger minimum leaf size.")
    st.markdown("**Privacy risk score:** `max(0, min(100, (ROC-AUC − 0.50) × 200))`. This is a simple score created for this student project; it is not a recognised standard.")
    st.subheader("Limitations")
    st.write("- The dataset is synthetic and does not represent real people.")
    st.write("- Shadow data and target data are generated from the same general process; real settings may be substantially different.")
    st.write("- Membership inference performance depends on dataset, model, attack design and training conditions.")
    st.write("- The prototype does not audit arbitrary uploaded proprietary models.")
    st.write("- The demonstration is not a legal, compliance or security certification.")
    st.info("Use this prototype to learn and demonstrate the method. Do not use its result alone to make claims about the privacy of a real system.")

st.markdown("---")
st.markdown(
    '<p class="small-note">LeakCheck · Student micro-project · Trustworthy and Responsible AI · Synthetic data only</p>',
    unsafe_allow_html=True,
)
