# LeakCheck — AI Model Privacy Auditor

A student micro-project for **Trustworthy and Responsible AI**. LeakCheck demonstrates a membership inference attack using synthetic data, trains shadow models and an attack classifier, and compares a baseline model with a less complex model.

## Features

- Streamlit dashboard with navigation
- Synthetic binary classification data (no real personal data)
- Target random-forest model
- Shadow models used to train a membership attack classifier
- Attack precision, recall, ROC-AUC and a project-defined privacy-risk score
- Baseline vs. mitigation comparison
- Results table, chart and CSV download
- Methodology, protection recommendations and limitations

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install packages and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Upload `app.py`, `requirements.txt`, and `README.md` to a GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Create a new app and select your repository, branch, and `app.py`.
4. Deploy and test the public URL.

## How the demonstration works

1. A target model is trained on a synthetic training split.
2. Shadow models are trained on separately generated synthetic data. Their predictions on training (member) and held-out (non-member) samples are used to create labelled examples for the attack model.
3. A logistic-regression attack model learns from four prediction features: confidence in the true class, maximum class confidence, prediction entropy, and the margin between the two largest probabilities.
4. The attack is evaluated on member and non-member samples associated with the target model. Precision, recall and ROC-AUC are calculated from actual predictions.
5. A baseline random forest is compared with a shallower forest with a larger minimum leaf size.

## Privacy risk score

For demonstration only:

`risk_score = max(0, min(100, (ROC-AUC - 0.50) * 200))`

An AUC of 0.50 maps to 0 and an AUC of 1.00 maps to 100. This score is **defined for this project** and is not an official industry standard or security certification.

## Important limitations

- Uses generated synthetic data, not actual personal or confidential data.
- The shadow and target data follow similar synthetic data-generation processes.
- A lower attack score in one experiment does not prove that a real model is private.
- The prototype does not audit arbitrary uploaded or proprietary models.
- The comparison is an educational experiment, not a guarantee that the mitigation will always reduce attack success.

## CIAP topic mapping

Topic 5: **LeakCheck — Auditing Membership Inference Vulnerabilities in Proprietary Biometric/HR Models**.

This prototype demonstrates the target, shadow and attack model concept and mitigation evaluation in a controlled synthetic setting. It is a student demonstration and should not be described as a production-ready audit of proprietary HR or biometric systems.
