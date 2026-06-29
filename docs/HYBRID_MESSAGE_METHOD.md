# Hybrid message method

The current method under investigation is a dual-head hybrid message for distributed IDS agents.

Each local agent sends a compact message:

```text
M_i = top-m class evidence + top-k explanation evidence + local anomaly scalar
```

where:

- `top-m class evidence` preserves known-attack discriminative information.
- `top-k explanation evidence` preserves sparse feature-level rationale.
- `local anomaly scalar` gives the coordinator a lightweight open-set signal.

The diagnostic implementation used locally is:

```bash
scripts/xmag_hybrid_message_diagnostic.py
```

The strongest compact setting found so far is:

```text
method: xmag_top1_class_token_anomaly_fusion
k: 1
top_m: 1
alpha: 1.0
message_bytes_per_flow: 24
```

This setting is referred to in notes as `X-MAG-DH-24B`.

## Current interpretation

`X-MAG-DH-24B` is compact and very strong on known-class classification. It works well for most 5G-NIDD unknown holdouts, but fails on `UDPFlood` and has low recall for `SlowrateDoS` at the current 5% false-alarm threshold. Therefore, it is not the final paper method yet; it is the current compact baseline that motivates a failure-focused sweep.

## Failure-focused success criteria

A candidate should improve both failure cases without losing lightweight behavior:

```text
min_known_macro_f1 >= 0.99
min_unknown_auroc >= 0.85
min_unknown_recall >= 0.40
max_false_alarm <= 0.06
max_message_bytes <= 72
```
