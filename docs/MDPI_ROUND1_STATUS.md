# MDPI Round-1 Revision Status

The protocol-matched five-seed/eight-holdout experiment suite is complete. Exact result archives are stored under `results/archives/`; extracted summary tables and figures are under `results/mdpi_r1/` and `results/mdpi_r1_ciciot/`.

One optional post-processing deliverable remains for Reviewer 2: ROC curves, score histograms, and accepted-unknown confusion analysis for UDPFlood and SlowrateDoS. Run `scripts/export_mdpi_hard_holdout_support.py`; it uses existing `score_components.npz` files and does not retrain models unless those arrays were removed.
