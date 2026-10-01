# M conditional prediction and upstream isolation

This archive contains the completed grouped held-out predictive comparison, its independent coefficient replay, and the source-precision repair of the upstream isolation experiment. It preserves the failed v1 records rather than silently replacing them.

- H593: original five component folds and natural ColNomic C128. 570 target-present queries; 23 candidate misses reported separately.
- C+M versus C-only: group-equal held NLL difference -0.558735, exploratory group-bootstrap 95% interval [-0.782504, -0.359249]. All five fold differences are negative. Parameter-matched C, estimated M residual, and RAW+C controls are included.
- These are statistical probes on an already opened panel, not a new deployed-model accuracy claim, a direct conditional-mutual-information estimate, or a unique causal explanation.
- Upstream v1 accepted 63 complete pairs. Fourteen additional-group pairs encountered BF16 requantization that violated the intended rotation invariants. v2 reuses verified FP32 results and reruns all 13 arms for these 14 pairs with a common FP32 convention and explicit original-dtype native controls. The invariant gates remain unchanged.
- Repair chain: 5173346 (pilot), 5173347 (14 workers), 5173348 (join). It was submitted and spool-verified; this archive does not certify its final outcome. The obsolete failed-dependency join 5171647 was cancelled.

The Chinese report is `ICLR/new ROUTEA/RC/reports/REPORT_M_CONDITIONAL_PREDICTION_AND_ISOLATION_REPAIR_20261001.md`. Programs, launchers, protocols, scalar parameters, all held-out candidate scores, and source SHA records are included. Raw images, large weights, intermediate tensor/NPZ caches and scheduler logs remain in the workspace and are not uploaded. The manifest records this exact snapshot.
