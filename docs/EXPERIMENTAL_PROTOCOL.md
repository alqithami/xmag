# X-MAG-IDS experimental protocol v0.1

## Main empirical claim

Explanation-token communication can preserve a useful fraction of full-feature IDS performance while reducing message size and improving interpretability in distributed 5G-IoT intrusion detection.

## Pilot dataset

Primary pilot: 5G-NIDD, starting with `Encoded.csv`. The code does not redistribute 5G-NIDD. Reviewers should download the dataset separately and update `dataset.path` in the YAML config.

## First unknown-attack protocol

1. Use `SYN Flood` as the held-out unknown attack.
2. Train only on benign and known attack families.
3. Evaluate known-class macro-F1 on known test flows.
4. Evaluate unknown AUROC and unknown recall on the held-out attack.
5. Repeat with `UDP Scan` as a fallback or second holdout.

## Agent definition

Use transparent edge-monitor agents when physical source-device or base-station columns are absent from the leakage-controlled encoded file. Default: `N = 8` agents.

## Baselines

1. Centralized full-feature classifier.
2. Local owner-agent only.
3. Majority vote across agents.
4. Logit-average communication.
5. Raw top-k feature coordinator.
6. Explanation-token coordinator.

## Metrics

Known accuracy, known balanced accuracy, known macro-F1, unknown AUROC, unknown recall at threshold, known false-alarm rate, message bytes per flow, training time, and prediction time.
