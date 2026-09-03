# MDPI Round-1 Revision Status

The protocol-matched five-seed/eight-holdout experiment suite is complete. Derived summary tables are committed under `results/mdpi_r1/` and `results/mdpi_r1_ciciot/`. Full per-trial arrays and ZIP archives remain local and can be regenerated from the checked-in scripts and public datasets.

One optional post-processing deliverable remains for Reviewer 2: ROC curves, score histograms, and accepted-unknown confusion analysis for UDPFlood and SlowrateDoS. Run `scripts/export_mdpi_hard_holdout_support.py`; it uses existing `score_components.npz` files and does not retrain models unless those arrays were removed.