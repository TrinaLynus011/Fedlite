# FedLite-Care Adaptive Privacy–Utility–Communication Controller

## Problem
The original simulator collected privacy telemetry, validation metrics, communication accounting, and training-time observations, but it did not feed that evidence back into the next training round. The workflow remained effectively static: fixed compression, fixed local epochs, and full client participation unless a user manually edited configuration files.

## Solution
The adaptive controller observes the completed round and selects the next-round configuration for:
- compression ratio
- local epochs
- client participation fraction

The controller is intentionally minimal and deterministic. It reuses the existing FedAvg simulation and DP-SGD accounting rather than replacing them.

## Inputs
The controller uses telemetry produced by the server after each round, including:
- round number
- epsilon
- privacy budget
- validation AUROC and AUPRC
- validation loss
- communication bytes for the round and cumulative totals
- communication budget
- training time
- mean/median gradient norms
- clipping fraction
- prior utility values
- previous controller configuration

## Outputs
The controller may choose:
- compression ratio from the configured candidate set
- local epochs from the configured candidate set
- participation fraction from the configured candidate set

The selected values are applied only to the next round, not the round currently in progress.

## Decision mechanism
The controller works as a closed loop:
1. Candidate generation from configuration files.
2. Hard-constraint filtering using privacy, communication, utility, and resource limits.
3. Deterministic heuristic scoring over remaining candidates.
4. Selection of the highest-scoring valid configuration.
5. Recording of selected action, score, and reasons for auditability.

The scoring is intentionally transparent and heuristic. It uses only observed telemetry and a documented trade-off between utility, communication cost, resource cost, and privacy pressure. It does not predict future utility.

## Limitations
- This is a sequential federated learning simulator, not a real hospital deployment.
- Communication and resource values are estimated from the existing simulator design.
- The controller does not magically know future model performance.
- Candidate scoring is heuristic and configuration-driven.
- Privacy guarantees remain those supported by the existing DP-SGD accountant.
- The controller does not change the underlying DP-SGD noise mechanism or reset epsilon.

## Auditable causal chain
OBSERVATION
  privacy telemetry + utility telemetry + communication telemetry + resource telemetry
      ↓
CONSTRAINT EVALUATION
      ↓
CANDIDATE CONFIGURATION SELECTION
      ↓
NEXT-ROUND ACTION
  compression + local epochs + client participation
      ↓
RESULTING ROUND PERFORMANCE
      ↓
NEXT OBSERVATION

This trail is persisted in the experiment metadata and round metrics so a later review can inspect the closed-loop behaviour.
