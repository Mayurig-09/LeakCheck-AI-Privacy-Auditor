import time
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="LeakCheck | AI Privacy Auditor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------- DESIGN ---------------------------------------
st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: #080d1a; color: #edf2ff; }
[data-testid="stHeader"] { background: rgba(8,13,26,.92); }
[data-testid="stSidebar"] { background: #101a2d; border-right: 1px solid #233653; }
[data-testid="stSidebar"] * { color: #e7efff !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: #9eb0ce !important; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1440px; }
h1,h2,h3 { color: #f4f7ff; letter-spacing: -.025em; }
p,li { color: #bdc9df; }
.hero {
  background: linear-gradient(115deg, #14294a 0%, #20234b 58%, #30204e 100%);
  border: 1px solid #344c78; border-radius: 20px;
  padding: 28px 32px; margin: 4px 0 22px 0;
  box-shadow: 0 14px 40px rgba(0,0,0,.18);
}
.hero-kicker { color:#9fc4ff; text-transform:uppercase; letter-spacing:.14em;
  font-size:.74rem; font-weight:700; }
.hero h1 { font-size:2.35rem; margin:.35rem 0 .25rem 0; }
.hero p { color:#c8d6f1; font-size:1.02rem; margin:0; }
.section-label { color:#8fb8ff; text-transform:uppercase; letter-spacing:.12em;
  font-size:.72rem; font-weight:700; }
div[data-testid="stMetric"] {
  background: linear-gradient(145deg,#121f35,#101a2c);
  border:1px solid #293d5e; padding:16px; border-radius:15px;
}
div[data-testid="stMetric"] label { color:#9fb2d0 !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#f3f6ff; }
div[data-testid="stMetricDelta"] { font-size:.82rem; }
div[data-testid="stButton"] > button, div[data-testid="stDownloadButton"] > button {
  border-radius:10px; border:1px solid #477bd0; font-weight:650;
  min-height:2.7rem;
}
div[data-testid="stButton"] > button[kind="primary"] {
  background:linear-gradient(100deg,#2868d7,#6545c7); color:white; border:0;
}
div[data-testid="stButton"] > button[kind="secondary"],
div[data-testid="stDownloadButton"] > button {
  background:#14233b; color:#eaf1ff;
}
[data-testid="stVerticalBlockBorderWrapper"] {
  background:#101a2b; border-color:#263a58 !important; border-radius:15px;
}
[data-testid="stDataFrame"] { border:1px solid #293d5e; border-radius:12px; overflow:hidden; }
hr { border-color:#263650; }
div[data-testid="stRadio"] label { padding:5px 0; }
.small-muted { color:#91a4c3; font-size:.84rem; }
.status-pill { display:inline-block; padding:5px 10px; border-radius:30px;
  background:#17365a; border:1px solid #315b8d; color:#b9d7ff;
  font-size:.75rem; font-weight:650; margin-right:6px; }
.info-panel { background:#101d31; border:1px solid #293e60; border-radius:13px;
  padding:15px 17px; color:#c3d0e6; }
</style>
""", unsafe_allow_html=True)

# ----------------------------- AUDIT ENGINE -----------------------------------
def make_synthetic_data(n_samples, n_features, random_state):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, min(n_features - 1, int(n_features * .65))),
        n_redundant=max(0, min(2, n_features - 2)),
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        weights=[.5, .5],
        flip_y=.02,
        class_sep=1.0,
        random_state=random_state,
    )
    return X, y


def prediction_features(model, X, y):
    probs = model.predict_proba(X)
    class_to_index = {label: i for i, label in enumerate(model.classes_)}
    true_class_conf = np.array([
        probs[i, class_to_index[label]] for i, label in enumerate(y)
    ])
    sorted_probs = np.sort(probs, axis=1)
    max_conf = sorted_probs[:, -1]
    second_conf = sorted_probs[:, -2] if probs.shape[1] > 1 else np.zeros(len(probs))
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
    return np.column_stack([
        true_class_conf, max_conf, entropy, max_conf - second_conf
    ])


def train_target_model(X_train, y_train, mitigated=False, seed=42):
    if mitigated:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=8,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1
        )
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_leaf=1,
            max_features=None,
            random_state=seed,
            n_jobs=-1
        )

    model.fit(X_train, y_train)
    return model


def train_shadow_attack(n_features, n_shadow_models, samples_per_shadow, seed):
    shadow_features, shadow_membership = [], []

    for i in range(n_shadow_models):
        Xs, ys = make_synthetic_data(
            samples_per_shadow, n_features, seed + 100 + i
        )

        X_in, X_out, y_in, y_out = train_test_split(
            Xs,
            ys,
            test_size=.5,
            random_state=seed + 200 + i,
            stratify=ys
        )

        shadow = RandomForestClassifier(
            n_estimators=70,
            max_depth=None,
            min_samples_leaf=1,
            random_state=seed + 300 + i,
            n_jobs=-1
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
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed
        )
    )

    attack_model.fit(X_attack, y_attack)
    return attack_model


def evaluate_attack(
    target_model,
    X_member,
    y_member,
    X_nonmember,
    y_nonmember,
    attack_model
):
    f_member = prediction_features(target_model, X_member, y_member)
    f_nonmember = prediction_features(target_model, X_nonmember, y_nonmember)

    X_eval = np.vstack([f_member, f_nonmember])

    y_eval = np.concatenate([
        np.ones(len(f_member), dtype=int),
        np.zeros(len(f_nonmember), dtype=int),
    ])

    scores = attack_model.predict_proba(X_eval)[:, 1]
    preds = (scores >= .5).astype(int)

    auc = roc_auc_score(y_eval, scores)
    risk = float(np.clip((auc - .5) * 200, 0, 100))

    return {
        "Precision": float(precision_score(y_eval, preds, zero_division=0)),
        "Recall": float(recall_score(y_eval, preds, zero_division=0)),
        "ROC-AUC": float(auc),
        "Privacy Risk Score": risk,
        "Target Model Accuracy": float(accuracy_score(
            np.concatenate([y_member, y_nonmember]),
            np.concatenate([
                target_model.predict(X_member),
                target_model.predict(X_nonmember)
            ])
        )),
        "Member samples": int(len(X_member)),
        "Non-member samples": int(len(X_nonmember)),
    }


@st.cache_data(show_spinner=False, ttl=1800)
def run_experiment(n_samples=1400, n_features=10, n_shadows=3, seed=42):
    X, y = make_synthetic_data(n_samples, n_features, seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=.4,
        random_state=seed,
        stratify=y
    )

    attack_model = train_shadow_attack(
        n_features,
        n_shadows,
        max(500, n_samples // 2),
        seed
    )

    baseline = train_target_model(X_train, y_train, False, seed)
    mitigated = train_target_model(X_train, y_train, True, seed + 1)

    base_eval = evaluate_attack(
        baseline, X_train, y_train, X_test, y_test, attack_model
    )

    mitigated_eval = evaluate_attack(
        mitigated, X_train, y_train, X_test, y_test, attack_model
    )

    rows = []

    for name, result in [
        ("Baseline model", base_eval),
        ("Mitigated model", mitigated_eval)
    ]:
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

    return pd.DataFrame(rows), base_eval, mitigated_eval


def risk_label(score):
    if score < 10:
        return "Low observed signal"
    if score < 30:
        return "Limited observed signal"
    if score < 60:
        return "Moderate observed signal"
    return "High observed signal"


# ------------------------------- SIDEBAR --------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ LeakCheck")
    st.caption("AI MODEL PRIVACY AUDITOR")
    st.markdown("---")

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Run Privacy Audit",
            "Results & Analytics",
            "Mitigation Guide",
            "Methodology & Limits"
        ],
        label_visibility="visible",
    )

    st.markdown("---")
    st.markdown("**PROJECT STATUS**")
    st.markdown(
        '<span class="status-pill">● Prototype</span>',
        unsafe_allow_html=True
    )
    st.caption("Synthetic data environment")
    st.caption("TY B.Tech AI&DS · Trustworthy & Responsible AI")


if "experiment_result" not in st.session_state:
    st.session_state.experiment_result = None


# ------------------------------- HEADER ----------------------------------------
st.markdown("""
<div class="hero">
  <div class="hero-kicker">AI SECURITY &nbsp; / &nbsp; PRIVACY INTELLIGENCE</div>
  <h1>LeakCheck</h1>
  <p>Evaluate membership inference exposure in machine-learning models.</p>
</div>
""", unsafe_allow_html=True)


# -------------------------------- PAGES ----------------------------------------

if page == "Overview":

    st.markdown(
        '<div class="section-label">Workspace overview</div>',
        unsafe_allow_html=True
    )

    st.title("Privacy audit workspace")

    st.write(
        "Run a controlled membership-inference experiment and compare "
        "a baseline model with a less complex model."
    )

    r = st.session_state.experiment_result

    if r:
        base = r["baseline"]
        mitigated = r["mitigated"]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Baseline attack AUC", f"{base['ROC-AUC']:.3f}")

        c2.metric(
            "Mitigated attack AUC",
            f"{mitigated['ROC-AUC']:.3f}",
            delta=f"{mitigated['ROC-AUC']-base['ROC-AUC']:+.3f}",
            delta_color="inverse"
        )

        c3.metric(
            "Baseline risk index",
            f"{base['Privacy Risk Score']:.1f}/100"
        )

        c4.metric(
            "Mitigated risk index",
            f"{mitigated['Privacy Risk Score']:.1f}/100"
        )

        st.caption(f"Last experiment: {r['timestamp']}")

    else:
        c1, c2, c3 = st.columns(3)

        c1.metric("Audit technique", "Membership inference")
        c2.metric("Data environment", "Synthetic")
        c3.metric("Model comparison", "Baseline / regularized")

        st.info(
            "No audit has been run in this session. "
            "Open **Run Privacy Audit** to start."
        )

    st.markdown("### Audit workflow")

    w1, w2, w3, w4 = st.columns(4)

    for col, number, title, desc in [
        (w1, "01", "Configure", "Choose dataset and shadow-model settings."),
        (w2, "02", "Train", "Build target, shadow and attack models."),
        (w3, "03", "Evaluate", "Measure membership inference performance."),
        (w4, "04", "Compare", "Review baseline and mitigation results."),
    ]:
        with col:
            with st.container(border=True):
                st.markdown(
                    f'<div class="section-label">STEP {number}</div>',
                    unsafe_allow_html=True
                )
                st.markdown(f"**{title}**")
                st.caption(desc)

    st.markdown(
        '<div class="info-panel">DEMO ENVIRONMENT &nbsp; · &nbsp; '
        'This prototype generates synthetic data only. It does not inspect '
        'real HR, biometric, or uploaded datasets.</div>',
        unsafe_allow_html=True
    )


elif page == "Run Privacy Audit":

    st.markdown(
        '<div class="section-label">Experiment controls</div>',
        unsafe_allow_html=True
    )

    st.title("Run privacy audit")

    st.write(
        "Configure the synthetic experiment, then launch "
        "the membership-inference evaluation."
    )

    st.warning(
        "Synthetic data only. Do not enter or upload real personal "
        "or confidential information."
    )

    with st.container(border=True):

        st.markdown("### Dataset configuration")

        left, right = st.columns(2)

        with left:
            n_samples = st.slider(
                "Dataset size", 800, 2400, 1400, step=200
            )

            n_features = st.slider(
                "Number of features", 6, 16, 10
            )

        with right:
            n_shadows = st.slider(
                "Shadow models", 2, 5, 3
            )

            seed = st.number_input(
                "Random seed",
                min_value=0,
                max_value=9999,
                value=42,
                step=1
            )

        st.caption(
            "Larger datasets and more shadow models may increase execution time."
        )

        run = st.button(
            "▶  Run Privacy Audit",
            type="primary",
            use_container_width=True
        )

    if run:

        with st.spinner(
            "Training models and evaluating the membership-inference attack..."
        ):

            try:
                df, base, mitigated = run_experiment(
                    int(n_samples),
                    int(n_features),
                    int(n_shadows),
                    int(seed)
                )

                st.session_state.experiment_result = {
                    "df": df,
                    "baseline": base,
                    "mitigated": mitigated,
                    "settings": {
                        "samples": int(n_samples),
                        "features": int(n_features),
                        "shadow_models": int(n_shadows),
                        "seed": int(seed)
                    },
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                st.success(
                    "Audit completed. Open Results & Analytics to review the findings."
                )

            except Exception as exc:
                st.error(
                    "The experiment could not finish. Try the default settings."
                )
                st.exception(exc)

    st.markdown("### Metrics generated")

    m1, m2, m3 = st.columns(3)

    m1.markdown("**Precision**")
    m1.caption(
        "How many records predicted as members were actually members."
    )

    m2.markdown("**Recall**")
    m2.caption(
        "How many actual member records the attack identified."
    )

    m3.markdown("**ROC-AUC**")
    m3.caption(
        "How well the attack separates members from non-members."
    )


elif page == "Results & Analytics":

    st.markdown(
        '<div class="section-label">Audit intelligence</div>',
        unsafe_allow_html=True
    )

    st.title("Results & analytics")

    r = st.session_state.experiment_result

    if r is None:

        st.info("No results available yet. Run a privacy audit first.")

    else:

        base = r["baseline"]
        mitigated = r["mitigated"]
        df = r["df"]

        st.caption(
            f"Last run: {r['timestamp']}  ·  Configuration: {r['settings']}"
        )

        a, b, c, d = st.columns(4)

        a.metric(
            "Baseline attack AUC",
            f"{base['ROC-AUC']:.3f}"
        )

        b.metric(
            "Mitigated attack AUC",
            f"{mitigated['ROC-AUC']:.3f}",
            delta=f"{mitigated['ROC-AUC']-base['ROC-AUC']:+.3f}",
            delta_color="inverse"
        )

        c.metric(
            "Baseline risk index",
            f"{base['Privacy Risk Score']:.1f}/100"
        )

        d.metric(
            "Mitigated risk index",
            f"{mitigated['Privacy Risk Score']:.1f}/100"
        )

        st.markdown("### Model comparison")

        display = df.copy()

        numeric = display.select_dtypes(include="number").columns
        display[numeric] = display[numeric].round(3)

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

        chart = df.set_index("Model")[[
            "Attack Precision",
            "Attack Recall",
            "Attack ROC-AUC",
            "Target Model Accuracy"
        ]]

        st.markdown("### Performance metrics")

        st.bar_chart(
            chart,
            use_container_width=True
        )

        st.markdown("### Findings")

        x, y = st.columns(2)

        with x:
            st.markdown("**Baseline model**")

            st.metric(
                "Observed attack signal",
                risk_label(base["Privacy Risk Score"])
            )

            st.caption(
                f"Target accuracy: {base['Target Model Accuracy']:.3f}"
            )

        with y:
            st.markdown("**Less-complex model**")

            st.metric(
                "Observed attack signal",
                risk_label(mitigated["Privacy Risk Score"])
            )

            st.caption(
                f"Target accuracy: {mitigated['Target Model Accuracy']:.3f}"
            )

        if mitigated["ROC-AUC"] < base["ROC-AUC"]:

            st.success(
                "The attack AUC decreased in this run after reducing "
                "model complexity. Review accuracy alongside privacy metrics."
            )

        elif mitigated["ROC-AUC"] > base["ROC-AUC"]:

            st.warning(
                "The attack AUC increased in this run. "
                "A mitigation must be measured; it should not be assumed to work."
            )

        else:

            st.info("The attack AUC was unchanged in this run.")

        st.download_button(
            "⬇  Download audit results (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="leakcheck_audit_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.caption(
            "The privacy-risk index is project-defined and educational, "
            "not an industry-standard certification."
        )


elif page == "Mitigation Guide":

    st.markdown(
        '<div class="section-label">Recommended safeguards</div>',
        unsafe_allow_html=True
    )

    st.title("Mitigation guide")

    st.write(
        "Practical controls to consider when reducing membership-inference exposure."
    )

    items = [
        (
            "01",
            "Reduce overfitting",
            "Use regularization, validation data and sensible model complexity. "
            "This prototype compares a full-depth forest with a shallower forest."
        ),
        (
            "02",
            "Limit confidence exposure",
            "Avoid returning detailed probability vectors when an application "
            "only needs a predicted class. This can reduce some attack signals, "
            "but does not guarantee privacy."
        ),
        (
            "03",
            "Evaluate privacy repeatedly",
            "Repeat tests across random seeds, datasets and model settings. "
            "A single experiment is not a universal security guarantee."
        ),
        (
            "04",
            "Consider differential privacy",
            "For sensitive applications, investigate correctly configured "
            "differential privacy methods and their privacy-utility trade-offs."
        ),
    ]

    for number, title, desc in items:

        with st.container(border=True):

            st.markdown(
                f'<div class="section-label">CONTROL {number}</div>',
                unsafe_allow_html=True
            )

            st.markdown(f"**{title}**")
            st.write(desc)

    st.info(
        "Implemented in this prototype: reduced model complexity through "
        "a shallower Random Forest with larger minimum leaf size. "
        "Confidence masking and differential privacy are recommendations only; "
        "they are not implemented."
    )


elif page == "Methodology & Limits":

    st.markdown(
        '<div class="section-label">Technical reference</div>',
        unsafe_allow_html=True
    )

    st.title("Methodology & limitations")

    with st.expander("Experiment methodology", expanded=True):

        st.markdown("""
        1. Generate a synthetic binary-classification dataset.
        2. Train a target Random Forest model.
        3. Train shadow Random Forest models on separate synthetic datasets.
        4. Use shadow-model prediction features to train a Logistic Regression attack classifier.
        5. Evaluate whether the attack distinguishes target-model training records (members) from held-out records (non-members).
        6. Compare a baseline forest with a less-complex forest.
        """)

    with st.expander("Attack features"):

        st.markdown("""
        - Confidence in the true class
        - Maximum predicted class probability
        - Prediction entropy
        - Margin between the two highest class probabilities
        """)

    with st.expander("Metric definitions"):

        st.markdown("""
        - **Precision:** fraction of predicted members that are actual members.
        - **Recall:** fraction of actual members identified by the attack.
        - **ROC-AUC:** ability to rank members above non-members; 0.50 represents chance-level ranking on a balanced evaluation.
        - **Privacy-risk index:** `max(0, min(100, (ROC-AUC - 0.50) × 200))`. This is a project-defined educational index, not an official standard.
        """)

    st.markdown("### Scope and limitations")

    st.warning(
        "This is a controlled student prototype, not a production privacy-auditing tool."
    )

    st.markdown("""
    - Uses generated synthetic data, not real personal or confidential data.
    - The shadow and target datasets follow similar synthetic generation processes.
    - It does not audit arbitrary uploaded or proprietary models.
    - One lower attack score does not prove that a model is private.
    - The model-complexity comparison may not reduce attack success in every run.
    - Confidence masking and differential privacy are not implemented in this version.
    """)

    st.caption(
        "CIAP Micro-Project · Topic 5 · Trustworthy and Responsible AI"
    )
