# Prior Art Risk Assessment

## Preliminary technical prior-art assessment

This document records a preliminary technical prior-art assessment only. It does not constitute legal advice and requires professional patent examination.

## FedCS

1. What it appears to disclose:
   Resource-aware client selection in federated learning, with participation decisions based on client state and resource constraints.

2. Overlap with proposed elements:
   Client selection, resource awareness, and adaptive participation decisions overlap with the controller’s participation-fraction and feasibility logic.

3. Potentially distinguishing elements:
   The repository’s closed-loop round-to-round controller using completed-round telemetry and a deterministic multi-objective selection across participation, epochs, and compression remains a narrower point of distinction.

4. Risk type:
   Novelty risk and obviousness risk.

## FedDdrl

1. What it appears to disclose:
   Adaptive client participation or training effort under dynamic conditions, often using a reinforcement-learning or optimization-based control policy.

2. Overlap with proposed elements:
   Adaptive local training effort and feedback-driven control overlap with the repository’s epoch and participation adaptation.

3. Potentially distinguishing elements:
   The repository implements a specific deterministic heuristic scoring process with hard feasibility constraints and next-round application, rather than a learned policy or a broader distributed optimization framework.

4. Risk type:
   Obviousness risk, with possible novelty risk depending on the exact disclosures.

## FedCG

1. What it appears to disclose:
   Client selection and gradient compression in a federated setting.

2. Overlap with proposed elements:
   Compression ratio and client participation are directly in the repository candidate space.

3. Potentially distinguishing elements:
   The repository combines these dimensions with completed-round telemetry, hard constraints, and deterministic selection for the next round.

4. Risk type:
   Obviousness risk.

## CN115796271A

1. What it appears to disclose:
   Per-round client selection and compression-related federated mechanisms.

2. Overlap with proposed elements:
   Compression choices, per-round adaptation, and client participation control overlap with the repository’s candidate generation and configuration changes.

3. Potentially distinguishing elements:
   The repository’s use of privacy, utility, communication, and resource constraints in a deterministic controller loop remains a possible distinction, but the exact claim scope must be narrowed carefully.

4. Risk type:
   Novelty risk and obviousness risk.

## CN117252253A

1. What it appears to disclose:
   Resource/privacy-budget-aware adaptive federated edge learning.

2. Overlap with proposed elements:
   Privacy-aware adaptation, resource-aware control, and edge settings overlap directly with the repository’s controller design.

3. Potentially distinguishing elements:
   The precise combination of telemetry-driven state construction, hard feasibility filtering, multidimensional candidate generation, and next-round sequencing may still be differentiating, but this is not yet established.

4. Risk type:
   Novelty risk and obviousness risk.

## FedCER

1. What it appears to disclose:
   Adaptive selection and epoch configuration under dynamic resource conditions.

2. Overlap with proposed elements:
   Local epochs, participation, and runtime/resource feedback overlap strongly with the repository implementation.

3. Potentially distinguishing elements:
   The repository’s combination of privacy and communication budgets with deterministic scoring and next-round assignment may still be a possible distinction.

4. Risk type:
   Obviousness risk.

## FedJoint

1. What it appears to disclose:
   Runtime-feedback-based adaptive federated orchestration.

2. Overlap with proposed elements:
   Runtime telemetry, feedback-driven configuration changes, and adaptive orchestration overlap with the closed-loop controller.

3. Potentially distinguishing elements:
   The candidate distinction remains in the specific structure of telemetry-to-state-to-candidate-to-next-round selection, not in the general idea of runtime adaptation itself.

4. Risk type:
   Obviousness risk.

## EcoFL

1. What it appears to disclose:
   Resource or energy-aware federated learning orchestration.

2. Overlap with proposed elements:
   Communication and resource-aware orchestration overlap with the repository’s communication-cost and resource-cost terms.

3. Potentially distinguishing elements:
   The repository’s explicit privacy constraint and the deterministic multi-objective scoring across compression, epochs, and participation may remain possible distinguishing features, but they are not yet proven novel.

4. Risk type:
   Obviousness risk.

## Overall preliminary position

The preliminary technical prior-art assessment suggests that broad claims covering adaptive federated optimization, client selection, local epochs, compression, or privacy-aware edge orchestration are likely to be crowded and high risk.

The more promising path is to narrow the claim set to the specific repository implementation:

- completed-round telemetry,
- controller state construction,
- hard feasibility filtering,
- multidimensional candidate generation,
- deterministic multi-objective selection,
- application to a subsequent round.

This remains a potentially distinguishing boundary and requires professional patent examination before any legal or commercial claim is made.
