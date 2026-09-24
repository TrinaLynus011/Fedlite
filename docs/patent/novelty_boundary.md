# Novelty Boundary

## Candidate distinction summary

| Feature | Known prior art | Status | Role in patent | Separate |
| --- | --- | --- | --- | --- |
| Federated learning itself | Broadly known | Known / crowded | Not a patentable core | No |
| Resource-aware client selection | FedCS and related adaptive FL methods | Known / crowded | Not a patentable core | No |
| Adaptive local training effort / epoch control | FedDdrl, FedCER, FedJoint, and related methods | Known / crowded | Not a patentable core | No |
| Compression-aware FL control | FedCG, CN115796271A, and related methods | Known / crowded | Not a patentable core | No |
| Privacy-budget-aware adaptive control | CN117252253A and related methods | Known / crowded | Not a patentable core | No |
| Completed-round telemetry → controller state | Partially known in dynamic FL systems | Potentially distinguishing / requires claim-level prior-art analysis | Candidate inventive concept boundary | Yes, potentially |
| Hard feasibility filtering before scoring | Conventional constraint checking in optimization systems | Potentially distinguishing / requires claim-level prior-art analysis | Candidate inventive concept boundary | Yes, potentially |
| Multi-dimensional candidate generation across participation × epochs × compression | Present in some adaptive FL designs | Potentially distinguishing / requires claim-level prior-art analysis | Candidate inventive concept boundary | Potentially |
| Deterministic multi-objective selection of next configuration | Known in optimization and control literature | Potentially distinguishing / requires claim-level prior-art analysis | Candidate inventive concept boundary | Potentially |
| Application of selected configuration to the next round only | Implementation-specific and feedback-loop design | Potentially distinguishing / requires claim-level prior-art analysis | Candidate inventive concept boundary | Potentially |
| Edge-device telemetry simulation | Supporting embodiment only | Supporting embodiment only | Secondary support, not core independent claim | Not core |
| Hospital readmission prediction use case | Domain application | Supporting embodiment only | Optional field-of-use context | Not core |

## Known / crowded

These areas are well covered by existing work and should not be positioned as novel in themselves:

- federated learning
- adaptive client participation
- adaptive local epochs
- compression-aware FL
- privacy-budget-aware FL
- resource-aware FL orchestration
- feedback-driven adaptive FL

## Potentially distinguishing

The strongest potentially distinguishing combination is the implemented controller loop:

```
completed-round telemetry
→ controller state
→ hard feasibility filtering
→ multidimensional candidate generation
→ deterministic multi-objective selection
→ next-round configuration
→ feedback loop
```

This interaction is grounded in the repository implementation in `src/federated/adaptive_controller.py` and is applied in the round loop in `src/federated/server.py`.

This should be called:

**Potentially distinguishing / requires claim-level prior-art analysis**

It is not a definitive novelty statement.

## Supporting embodiment only

These are useful implementation details but should not define the patent core unless separately supported:

- Wokwi/virtual ESP32 telemetry pipeline in `src/edge/`
- edge availability states such as `AVAILABLE`, `BUSY`, `CONSTRAINED`, and `OFFLINE`
- the hospital readmission prediction use case
- hospital-specific metrics and training data handling

These elements may support a dependent claim or an embodiment but should not be treated as the independent inventive foundation unless supported by a stronger prior-art analysis.
