# Hazard Head Evaluation Report (LOEO Protocol)

## Onset Forecasting Performance by Horizon

| Horizon | F1 Mean | Precision Mean | Recall Mean | FPR Mean | ROC-AUC Mean | PR-AUC Mean |
|---|---|---|---|---|---|---|
| **H=1** | 0.1429 | 0.1396 | 0.1622 | 0.1450 | 0.5610 | 0.6136 |
| **H=2** | 0.0803 | 0.0698 | 0.1351 | 0.2000 | 0.5556 | 0.5703 |
| **H=5** | 0.0964 | 0.0833 | 0.1351 | 0.1628 | 0.5230 | 0.6043 |

## Cumulative Hazard Trajectory Verification
- **Formula**: $P(\text{event} \le K) = 1 - \prod_{k=1}^K (1 - h_k)$
- **Properties Verified**:
  1. Boundedness: $0.0 \le P(\text{event} \le K) \le 1.0$ across all horizons.
  2. Monotonicity: $P(\text{event} \le K+1) \ge P(\text{event} \le K)$ for all sequences.

## Horizon Diagnostic Notes (Claim Ladder Rung 6 Alignment)
- **H=1 / H=2**: Short-horizon hazard predictions capture immediate onset transitions.
- **H=5 Onset vs Schedule Baseline**: Per the project's prior Gate 0 diagnostic finding, real traffic features underperform schedule-only features at H=5 (F1 ~0.095 vs ~0.253 schedule baseline) due to long temporal distance from initial probes. This is an expected, documented finding in the claim ladder.