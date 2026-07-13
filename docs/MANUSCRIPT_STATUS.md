# Manuscript status

The current manuscript is being revised as a submission-oriented paper centered on X-MAG-COS-24B.

## Current scientific position

The paper should be framed as a communication-constrained evidence-messaging IDS, not as a deployed multi-agent system. The distributed monitors are emulated monitoring sources because the encoded datasets do not provide leakage-safe physical agent identities.

## Main claim

X-MAG-COS-24B provides a 24-byte evidence message consisting of:

```text
top-1 class evidence + top-1 attribution-proxy evidence + local anomaly scalar
```

Known-class classification and open-set rejection are separated into two heads. The open-set head uses uncertainty, anomaly evidence, and message-prototype residuals.

## Main 5G-NIDD result

Across eight leave-one-attack-family-out holdouts and three seeds:

```text
known macro-F1       = 0.998053 +/- 0.000496
unknown AUROC        = 0.943859 +/- 0.093135
unknown recall       = 0.839499 +/- 0.279391
false-alarm rate     = 0.050306 +/- 0.000836
worst unknown AUROC  = 0.709369
worst unknown recall = 0.218584
message size         = 24 bytes per flow
```

## Required framing

The paper should not claim that X-MAG-COS beats all full-feature centralized IDS baselines. A centralized full-feature Random Forest is a strong upper bound. The defensible claim is that X-MAG-COS provides a strong communication-constrained operating point and reveals the limits of compact evidence messaging.

## CICIoT2023 framing

CICIoT2023 should be presented as a stress test. The full-folder mapping shows partial transfer, but DoS/DDoS and Mirai open-set rejection remain weak. This result should be reported honestly as a limitation and future-work direction, not as a generalization win.

## Explainability framing

The method is explanation-aware because it transmits lightweight attribution-proxy evidence. It does not compute full SHAP or LIME values online. Stronger XAI claims require a separate fidelity or attribution-agreement experiment.

## Funding acknowledgement

The manuscript must include the required acknowledgement sentence:

> This research is supported by a grant (No. CRPG-00-0000) under the Cybersecurity Research and Innovation Pioneers Initiative, provided by the National Cybersecurity Authority (NCA) in the Kingdom of Saudi Arabia.
