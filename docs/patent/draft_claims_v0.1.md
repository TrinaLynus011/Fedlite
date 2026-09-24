# Draft Claims v0.1

DRAFT — NOT LEGAL FINAL

## Claim 1
A computer-implemented method for adaptive configuration of a federated learning process, the method comprising:

1. receiving completed-round telemetry from a federated learning process;
2. constructing controller state from the completed-round telemetry;
3. generating a plurality of candidate configurations for a subsequent federated-learning round, each candidate configuration including at least:
   - a participation fraction,
   - a local training epoch count,
   - a compression ratio;
4. evaluating each candidate configuration against one or more predetermined feasibility constraints;
5. excluding infeasible candidate configurations;
6. deterministically evaluating remaining candidate configurations using a multi-objective scoring function;
7. selecting a configuration from the remaining candidate configurations based on the multi-objective scoring function;
8. applying the selected configuration to a subsequent federated-learning round; and
9. repeating steps 1 through 8 using subsequent completed-round telemetry.

## Claim 2
The method of claim 1, wherein the completed-round telemetry comprises validation metrics, communication metrics, privacy-accountant metrics, and resource-use metrics.

## Claim 3
The method of claim 1, wherein the one or more predetermined feasibility constraints comprise at least one of:

- a privacy-budget constraint,
- a communication-budget constraint,
- a minimum utility constraint,
- a maximum resource-time constraint.

## Claim 4
The method of claim 1, wherein the plurality of candidate configurations is generated as a Cartesian product of candidate values for participation fraction, local training epoch count, and compression ratio.

## Claim 5
The method of claim 4, wherein the candidate values comprise:

- compression ratios in a set including 1.0, 0.5, 0.25, 0.1, and 0.01;
- local training epochs in a set including 1, 2, 3, and 5;
- participation fractions in a set including 0.6, 0.8, and 1.0.

## Claim 6
The method of claim 1, wherein the multi-objective scoring function includes utility, communication cost, resource cost, and privacy-related cost.

## Claim 7
The method of claim 1, wherein the deterministic evaluation comprises a tie-breaking rule that selects a candidate according to a predetermined ordering.

## Claim 8
The method of claim 1, wherein the privacy-accountant metrics are received from a privacy accounting routine and used as controller input.

## Claim 9
The method of claim 1, wherein the completed-round telemetry is stored in a rolling history used by the controller state.

## Claim 10
The method of claim 1, wherein the selected configuration is applied before a subsequent federated-learning round begins.

## Claim 11
A system comprising one or more processors and memory storing instructions that, when executed, perform the method of claim 1.

## Claim 12
A non-transitory computer-readable medium storing instructions that, when executed by one or more processors, perform the method of claim 1.

## Claim 13
The method of claim 1, wherein the federated learning process is used for hospital readmission prediction based on patient-level healthcare data.

## Notes

- This claim set is intentionally aligned to the verified repository implementation.
- It does not broaden beyond the implemented mechanism.
- It is not a legal final draft and requires claim-level prior-art review.
