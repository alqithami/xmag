# X-MAG-COS method

The current paper method is **X-MAG-COS: Composite Open-Set Scoring for Explanation-Aware Multi-Agent IDS**.

## Agent message

Each agent sends:

```text
top-m class evidence + top-k explanation evidence + local anomaly scalar
```

The frozen compact variant is:

```text
X-MAG-COS-24B
m = 1
k = 1
message = 24 bytes per flow
```

## Two-head design

The method separates the two tasks:

```text
Known-attack classification head:
    coalition classifier over compact hybrid messages

Unknown/open-set head:
    composite direct score using message residual, owner-agent uncertainty,
    and local anomaly evidence
```

## Frozen open-set score

```text
linear_max_resid_owner_b025__uncert_anomaly_b05_gamma_0.25
```

This score was selected because it materially improved the two failure cases, especially UDPFlood and SlowrateDoS, while preserving the 24-byte message size.

## Main claim

X-MAG-COS does not claim perfect zero-day detection. The defensible claim is:

> Compact explanation-aware communication preserves known-attack classification and substantially improves worst-case open-set recall when paired with a separate composite open-set head.
