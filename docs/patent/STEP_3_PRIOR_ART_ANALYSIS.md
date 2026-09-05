# FedLite-Care — Patent Step 3: Prior-Art Differentiation

This document is a technical differentiation analysis only. It does not make a legal patentability conclusion, does not redesign the system, and does not treat unverified results as evidence.

Scope:
- current implementation only
- use the strongest supported invention candidate from Step 2
- distinguish known techniques from the actual combination that FedLite-Care implements
- avoid overstating novelty or patentability

---

## 1. DEFINE THE CURRENT INVENTION CORE

From Step 2, the strongest current invention candidate is:

A custom sequential federated learning system for clinical readmission prediction that combines local record-level DP-SGD, per-example gradient clipping, Gaussian-noise perturbation, server-side FedAvg aggregation, and round-level privacy/communication/resource telemetry.

This is the technical core that is supported by the implemented code and by the verified pilot evidence. It is not a deployment system, not an adaptive control loop, and not a real hospital production runtime.

### Atomic components

A. distributed/local clinical data
- Hospital encounter data is partitioned into client datasets, with local training on each client.
- Implemented in src/data/partition.py and src/federated/server.py.
- This is not unique in itself; distributed clinical data processing is a known FL pattern.

B. local preprocessing
- Feature preparation and data partitioning are performed prior to model training.
- This is standard ML workflow logic, not the principal inventive core.

C. lightweight clinical prediction model
- A lightweight MLP is used for readmission prediction.
- Implemented in src/models/mlp.py.
- This is a conventional model family and is not a strong patent core by itself.

D. record-level DP-SGD
- The active privacy mechanism is per-record DP-SGD.
- Implemented in src/federated/client.py and src/privacy/dp_mechanism.py.
- This is an established privacy-preserving training technique, but the specific integration matters.

E. per-record gradient clipping
- Each example gradient is clipped before noise is applied.
- This is a standard DP-SGD component.

F. Gaussian perturbation
- Gaussian noise is added to the accumulated clipped gradients.
- This is standard differentially private training practice.

G. federated aggregation
- The server aggregates client updates via a custom FedAvg-style averaging procedure.
- Implemented in src/federated/server.py.
- FedAvg itself is a known technique.

H. privacy accounting
- A tight second-order subsampled Gaussian RDP accountant is present.
- Implemented in src/privacy/dp_mechanism.py.
- This is more specific than generic DP and is a technical strength, but the accounting method is not unique in the broader literature.

I. gradient/clipping telemetry
- Mean/median gradient norms, clipping fraction, and related metrics are logged per round.
- This is measurement, not adaptive control.

J. communication telemetry
- Upstream/downstream bytes are estimated and tracked.
- This is a reporting mechanism, not a new communication protocol.

K. model/resource measurements
- Model size, latency, memory, throughput, and benchmark measurements are captured.
- This is resource characterization, not a runtime optimizer.

L. other implemented technical components
- Top-K sparsification/compression is implemented as a static configuration mechanism.
- Validation-only threshold tuning and metadata export are included.
- These are support features, not the strongest novel core.

Important boundary:
- telemetry exists,
- but no closed-loop system changes privacy, compression, or resource settings based on measured behavior.
- Therefore passive measurement should not be described as adaptive resource optimization.

---

## 2. PRIOR-ART SEARCH

### Category A — Federated healthcare prediction

Search themes:
- federated learning hospital readmission prediction
- federated clinical risk prediction
- federated diabetes readmission prediction
- privacy-preserving hospital readmission prediction

Relevant prior art and related literature:

1. McMahan et al., 2016 — Communication-Efficient Learning of Deep Networks from Decentralized Data
- Authors: H. Brendan McMahan et al.
- Date: 2016
- Source: arXiv / AISTATS 2017
- URL: https://arxiv.org/abs/1602.05629
- Relevant overlap: foundational FedAvg approach; decentralized training with model averaging; communication-efficiency motivation.
- FedLite-Care difference: FedLite-Care uses a custom sequential simulator and a privacy-aware client training loop, not just standard FedAvg itself.

2. Sheller et al., 2020 — Federated learning in medicine: facilitating multi-institutional collaborations without sharing patient data
- Authors: Micah J. Sheller et al.
- Date: 2020
- Source: Scientific Reports / Nature Research
- URL: https://www.nature.com/articles/s41598-020-69250-1
- Relevant overlap: healthcare-oriented FL in multi-institution settings; emphasizes collaboration without sharing patient-level data.
- FedLite-Care difference: the project is a single-research-framework simulation of readmission prediction with explicit DP-SGD and telemetry, not a broad multi-site medical imaging federation.

3. Rieke et al., 2020 — The future of digital health with federated learning
- Authors: Nicola Rieke et al.
- Date: 2020
- Source: NPJ Digital Medicine
- URL: https://www.nature.com/articles/s41746-020-00323-1
- Relevant overlap: healthcare domain, privacy-preserving medical ML, institution-level collaboration.
- FedLite-Care difference: this is general healthcare FL context rather than a concrete readmission-specific combined DP+telemetry architecture.

4. Kaissis et al., 2020 — Secure, privacy-preserving and federated machine learning in medical imaging
- Authors: Georgios A. Kaissis et al.
- Date: 2020
- Source: Nature Machine Intelligence
- URL: https://www.nature.com/articles/s42256-020-0186-1
- Relevant overlap: privacy-preserving medical machine learning; federated learning in healthcare.
- FedLite-Care difference: no direct match to readmission prediction + record-level DP-SGD + round telemetry.

### Category B — Federated learning + differential privacy

Search themes:
- federated learning DP-SGD healthcare
- record-level differential privacy federated learning
- per-example gradient clipping federated healthcare
- DP-SGD federated clinical prediction

Relevant prior art:

5. Abadi et al., 2016 — Deep Learning with Differential Privacy
- Authors: Martín Abadi et al.
- Date: 2016
- Source: ACM CCS
- URL: https://arxiv.org/abs/1607.00133
- Relevant overlap: DP-SGD algorithm, per-example clipping, Gaussian noise, accountant ideas.
- FedLite-Care difference: FedLite-Care uses the DP-SGD pattern inside a federated client loop and records privacy telemetry; this is an integration of standard DP-SGD into a federated setting.

6. Geyer et al., 2017 — Differentially Private Federated Learning: A Client Level Perspective
- Authors: Robin C. Geyer, Tassilo Klein, Moin Nabi
- Date: 2017
- Source: arXiv
- URL: https://arxiv.org/abs/1712.07557
- Relevant overlap: DP-aware FL; client-level privacy; central server aggregation with privacy constraints.
- FedLite-Care difference: the project is record-level (example-level) rather than client-level and also tracks round-level communication/privacy telemetry, not just client-level DP.

7. Kairouz et al., 2019 — Advances and Open Problems in Federated Learning
- Authors: Peter Kairouz et al.
- Date: 2019
- Source: Foundations and Trends in Machine Learning
- URL: https://arxiv.org/abs/1912.04977
- Relevant overlap: FL overview, formal privacy concerns, communication constraints, system architecture landscape.
- FedLite-Care difference: broad foundation, but no direct match to the exact DP-SGD + readmission prediction + telemetry workflow.

8. Bonawitz et al., 2019 — Towards Federated Learning at Scale: System Design
- Authors: Keith Bonawitz et al.
- Date: 2019
- Source: MLSys
- URL: https://arxiv.org/abs/1902.01046
- Relevant overlap: federated system design, secure aggregation, realistic FL deployment architecture.
- FedLite-Care difference: FedLite-Care is a sequential custom simulator, not a large-scale production FL system, and it does not implement secure aggregation or a real orchestration runtime.

### Category C — Resource-aware federated learning

Search themes:
- resource-aware federated learning healthcare
- communication-efficient federated learning edge healthcare
- lightweight federated learning clinical systems
- federated learning edge device resource monitoring

Relevant prior art:

9. Sattler et al., 2019 — Robust and Communication-Efficient Federated Learning from Non-IID Data
- Authors: Felix Sattler et al.
- Date: 2019
- Source: arXiv / ICML workshop / related venues
- URL: https://arxiv.org/abs/1903.02899
- Relevant overlap: communication efficiency in federated learning; non-IID data; model update compression and robustness.
- FedLite-Care difference: FedLite-Care includes a static sparsification option and communication measurement, but not a dynamic resource-aware optimization loop.

10. Reisizadeh et al., 2020 — FedPAQ: A Communication-Efficient Federated Learning Method with Periodic Averaging and Quantization
- Authors: Amirhossein Reisizadeh et al.
- Date: 2020
- Source: arXiv / ICML-related work
- URL: https://arxiv.org/abs/2002.08305
- Relevant overlap: communication-efficient FL, quantization, periodic averaging.
- FedLite-Care difference: FedLite-Care is not a new communication-efficient FL algorithm; it is a configured communication accounting environment with optional sparsification.

11. Li et al., 2020 — Federated Optimization in Heterogeneous Networks
- Authors: Tian Li et al.
- Date: 2020
- Source: MLSys / arXiv
- URL: https://arxiv.org/abs/1812.06127
- Relevant overlap: heterogeneous client environments, distributed optimization, FL practicality.
- FedLite-Care difference: project is not a new heterogeneous optimization method; it is a simpler sequential simulator with privacy and measurement reporting.

### Category D — Privacy + resource optimization

Search themes:
- differential privacy communication cost model size gradient telemetry edge constraints federated healthcare

Relevant prior art:

12. Alistarh et al., 2018 — QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding
- Authors: Dan Alistarh et al.
- Date: 2018
- Source: NeurIPS
- URL: https://proceedings.neurips.cc/paper/2017/hash/6c340f25d9d80d30c1c6c2d8c40d0b18-Abstract.html
- Relevant overlap: communication cost reduction through gradient quantization / compression.
- FedLite-Care difference: compression exists as a static mechanism, but there is no evidence of a new privacy-aware optimization loop that jointly controls privacy, model size, and communication adaptively.

13. Dwork and Roth, 2014 — The Algorithmic Foundations of Differential Privacy
- Authors: Cynthia Dwork, Aaron Roth
- Date: 2014
- Source: Foundations and Trends in Theoretical Computer Science
- URL: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
- Relevant overlap: foundational DP, sensitivity, privacy accounting principles.
- FedLite-Care difference: foundational support rather than direct technical overlap; FedLite-Care applies those ideas in a federated healthcare pipeline, but does not invent the DP definition or basic accountant concepts.

14. Mironov, 2017 — Rényi Differential Privacy
- Authors: Ilya Mironov
- Date: 2017
- Source: IEEE Symposium on Security and Privacy / arXiv
- URL: https://arxiv.org/abs/1702.07476
- Relevant overlap: RDP formulation used in the privacy accountant.
- FedLite-Care difference: this is a standard accountant component, not the novelty source by itself.

### Category E — Patent literature

Targeted patent search themes:
- federated learning healthcare clinical prediction
- differential privacy healthcare model training
- edge resource-constrained federated learning
- communication-efficient FL in clinical applications

The current constrained review did not identify an exact close patent for the combination of:
- healthcare readmission prediction,
- federated learning,
- record-level DP-SGD,
- per-example clipping,
- communication/privacy telemetry,
- and a non-production simulation measurement framework.

This is not evidence of novelty; it indicates the patent landscape is broad, dominated by general FL and DP patents, while the specific combination remains relatively narrow in the literature. It also means that the safest patent strategy is to describe the combination precisely rather than relying on broad healthcare-FL language alone.

---

## 3. SOURCE QUALITY

The strongest sources are used where available:
- arXiv, Nature, ACM CCS, MLSys, and foundational DP/FL papers
- patent databases were checked only at the search-theme level, with emphasis on exact combination terms rather than generic FL keywords
- no direct exact-match patent was identified in the current focused search set for the current combination

For the important references above, the evidence is based on available published documents and technical abstracts, not on search-result snippets only. The references are sufficiently concrete to support a technical differentiation analysis.

---

## 4. PRIOR-ART MATRIX

| Prior art | Federated healthcare | Readmission prediction | DP-SGD / record-level DP | Resource awareness | Communication telemetry | Privacy accounting | Main overlap | Potential FedLite-Care distinction |
|---|---|---:|---:|---:|---:|---:|---|---|
| McMahan et al., 2016 — FedAvg | No (general FL) | No | No | Yes | Yes | No | Foundational model averaging, communication-efficient FL | FedLite-Care integrates DP-SGD and telemetry into a clinical simulation workflow |
| Abadi et al., 2016 — Deep Learning with Differential Privacy | No | No | Yes | No | No | Yes | DP-SGD, clipping, noise, privacy accounting | FedLite-Care applies DP-SGD inside a federated client loop, not in a centralized setting |
| Geyer et al., 2017 — DP Federated Learning | Yes (general FL) | No | Yes (client-level DP) | Some | No | Yes | DP in FL and privacy-preserving aggregation | FedLite-Care is record-level and includes resource and gradient telemetry |
| Kairouz et al., 2019 — FL survey | Yes | No | Yes | Yes | Yes | Some | Broad FL system landscape and privacy concerns | No direct algorithmic overlap; broad survey background only |
| Bonawitz et al., 2019 — System design at scale | Yes (system design) | No | Some | Yes | Yes | Some | FL orchestration and secure aggregation context | FedLite-Care is not a production-scale system design or secure aggregation implementation |
| Sheller et al., 2020 — medical FL | Yes | No | Some | No | No | Some | Healthcare FL without data centralization | FedLite-Care is readmission-focused and explicitly DP-SGD-based |
| Rieke et al., 2020 — future of digital health with FL | Yes | No | Some | Some | No | Some | Medical FL and privacy tradeoff context | FedLite-Care is more specific to record-level DP and per-round telemetry |
| Kaissis et al., 2020 — medical imaging FL | Yes | No | Some | No | No | Some | Privacy-preserving healthcare ML | FedLite-Care is a predictive healthcare workflow with a different task and mechanism |
| Sattler et al., 2019 — robust / communication-efficient FL | Some | No | No | Yes | Yes | No | Communication efficiency and non-IID robustness | FedLite-Care uses static sparsification, not a new robust communication optimizer |
| Reisizadeh et al., 2020 — FedPAQ | Some | No | No | Yes | Yes | No | Compression/quantization in FL | FedLite-Care adds privacy and telemetry, but not a novel communication optimization algorithm |
| Alistarh et al., 2018 — QSGD | No | No | No | Yes | Yes | No | Gradient compression / communication reduction | FedLite-Care is not primarily a compression algorithm; the compression is support-level |
| Dwork & Roth, 2014 — foundations of DP | No | No | Yes | No | No | Yes | DP foundations, sensitivity, privacy accounting | FedLite-Care applies known DP primitives in a particular clinical FL system |
| Mironov, 2017 — Rényi DP | No | No | Yes | No | No | Yes | RDP privacy accounting | This is a standard accountant basis, not the main inventive core |

This matrix shows the broad overlap clearly: the individual building blocks are well known. The more interesting question is whether the specific combination and implementation boundary provide differentiation.

---

## 5. FEATURE-BY-FEATURE OVERLAP

Classification scale:
- HIGH = strong literal or near-literal overlap in the literature
- MEDIUM = same concept, but not same combination or architecture
- LOW = analogous but materially different
- NOT FOUND IN SEARCH = no direct or close evidence identified in the current review

| FedLite-Care component | Prior-art overlap | Evidence basis | Comment |
|---|---|---|---|
| A. distributed/local clinical data | MEDIUM | FL healthcare papers, multi-site medical ML | Routine in healthcare FL; not uniquely distinguishing |
| B. local preprocessing | HIGH | Standard ML pipelines and healthcare FL preprocessing | Well-established |
| C. lightweight clinical prediction model | MEDIUM | Many medical ML baselines using MLPs/logistic models | Standard architecture |
| D. record-level DP-SGD | MEDIUM | Abadi et al. 2016; Geyer et al. 2017; DP-SGD literature | established technique; integration matters |
| E. per-record gradient clipping | HIGH | DP-SGD literature | standard |
| F. Gaussian perturbation | HIGH | DP-SGD and DP literature | standard |
| G. federated aggregation | HIGH | McMahan et al. 2016; generic FL literature | well known |
| H. privacy accounting | MEDIUM | Abadi, Mironov, DP literature | standard but implementation detail can be meaningful |
| I. gradient/clipping telemetry | LOW | not unique as simple reporting; some monitoring exists in FL systems | measurement-only feature |
| J. communication telemetry | MEDIUM | communication-aware FL literature | common monitoring practice |
| K. model/resource measurements | MEDIUM | edge benchmark research, model footprint measurement | common in edge ML, not usually the core invention |
| L. static Top-K compression | MEDIUM | sparse/quantized FL literature | present but not a new algorithm |
| L. validation-only threshold tuning / metadata packaging | LOW | standard engineering practice | not a technical novelty driver |
| L. closed-loop adaptive optimization | NOT FOUND IN SEARCH | no implementation in project code | should not be described as implemented |
| L. real hospital deployment | NOT FOUND IN SEARCH | no runtime or deployment layer in code | should not be claimed |

This classification is intentionally conservative. Some features were found in the literature but not in the exact combination or system boundary used in FedLite-Care.

---

## 6. IDENTIFY THE REAL DIFFERENTIATION

The strongest candidate distinction does not appear to be any single component like FedAvg or DP-SGD. The more plausible differentiation is the specific combination of:

- distributed clinical data partitioned into local client datasets,
- client-side local training with record-level DP-SGD,
- per-example clipping plus Gaussian noise inside a federated update path,
- server-side model averaging,
- and explicit round-level telemetry measuring privacy, gradient behavior, communication, and model/resource characteristics.

What is NOT the strongest distinction:
- FedAvg alone
- DP-SGD alone
- Gaussian noise alone
- MLP architecture alone
- telemetry alone

Important nuance:
- Telemetry alone is measurement.
- Measurement does not become adaptive optimization unless the system actually changes behavior based on the measurements.
- In the current code, the telemetry is recorded and reported, but it does not control training configuration in real time.
- Therefore, the project is better characterized as a privacy-preserving federated training simulator with resource and privacy monitoring, not as a closed-loop adaptive resource optimization system.

The likely strongest meaningful combination is therefore not "adaptive optimization" but rather:

- federated training architecture + local record-level DP-SGD + measurement of privacy and communication cost within a clinically relevant prediction task.

That combination is more defensible as a technical system description than a pure claim about telemetry-driven adaptation.

---

## 7. CLAIM RISK ANALYSIS

### LOW-RISK / WELL-SUPPORTED

These are clearly implemented and suitable for technical description:

- sequential federated learning simulation
- local client-side training on partitioned data
- record-level DP-SGD in the implemented pipeline
- per-example gradient clipping and Gaussian noise addition
- server-side FedAvg aggregation
- privacy accounting and metric export
- resource/communication measurement and benchmark reporting
- clinical readmission prediction task using structured data and a lightweight MLP

These features are not completely novel in the abstract, but they are well described by the actual code and the verified project artifacts. They are strong support for a technical description and a specification narrative.

### HIGH-RISK

These elements are heavily covered by prior art or are very broad:

- FedAvg
- DP-SGD
- Gaussian noise addition as a DP primitive
- MLP model architecture
- healthcare prediction using FL in general
- readmission prediction as a clinical task (not unique per se)
- record-level privacy as a general DP concept
- privacy accounting conceptually
- communication monitoring and resource tracking in general FL systems

These are important components of the project but they are not likely to carry a strong, standalone patent narrative unless tightly tied to a more specific integration or system behavior.

### DO NOT CLAIM

These should not be asserted as part of the project’s strong patent case without new evidence or a real implementation:

- adaptive resource optimization
- real-time control loop driven by gradient or communication telemetry
- hardware deployment or edge-device orchestration
- production hospital or healthcare system deployment
- a completed sigma=2.3 result as evidence
- any claim that the implementation is a proven production method in a clinical network
- legal novelty conclusions or legal patentability conclusions

### Specific evaluation

- FedAvg — high prior-art risk; known technique
- DP-SGD — high prior-art risk; known technique
- Gaussian noise — high prior-art risk; standard DP primitive
- MLP — high prior-art risk; common model family
- healthcare prediction — medium risk; broad domain concept
- readmission prediction — medium risk; strong task-specific but not uniquely patentable by itself
- record-level privacy — medium risk; standard DP framing
- privacy accounting — medium risk; standard accounting methods exist
- resource monitoring — medium risk; common in edge/FL systems
- communication measurement — medium risk; common in FL systems
- adaptive resource optimization — do not claim; not implemented
- gradient compression — medium risk; known compression literature
- hardware deployment — do not claim; not implemented

---

## 8. INVENTIVE-STEP PRESSURE TEST

Technical differentiation analysis — not a legal patentability opinion.

Ask: If an engineer already knew federated learning, DP-SGD, healthcare prediction, and resource monitoring individually, what exactly would they have to invent or implement to reproduce the FedLite-Care system?

The answer is not a single conceptual leap. It is a specific system integration task:

1. define a healthcare readmission prediction task over partitioned local data,
2. implement a client-side training loop where each example is clipped and noise is applied before aggregation,
3. aggregate client updates using FedAvg in a sequential simulation,
4. instrument the run with gradient, privacy, and communication telemetry,
5. export metadata and benchmarks without turning telemetry into an optimizer.

This is a meaningful integration problem, but it is still a technical engineering combination rather than a novel conceptual breakthrough by itself.

The strongest pressure point is not the individual components but the combination in a lightweight, reproducible, measurement-heavy research pipeline. The combination is more defensible as a system-level architecture than as a broad patentable innovation over the general literature.

---

## 9. PATENT STRATEGY

### Broad concept to describe

A privacy-aware, communication-aware federated learning system for clinical prediction that keeps local data on client sites while performing local record-level DP-SGD and aggregating model updates centrally for a shared predictive model.

This is a broad-but-accurate framing that matches the implemented system and avoids over-claiming.

### Narrower technical core

The narrower technical core should be:
- structured clinical prediction task,
- client-side local training with record-level DP-SGD,
- per-example clipping and Gaussian perturbation,
- weighted FedAvg aggregation,
- and round-level telemetry for privacy, communication, and resource behavior.

This is the strongest technical center supported by the code and the verified artifacts.

### Fallback embodiments

- same architecture on alternate healthcare prediction tasks
- same privacy/telemetry pattern with alternative clinical tabular models
- same client-side record-level DP training with different aggregation methods
- same metrics pipeline with variable communication compression settings

These fallback embodiments preserve the technical idea while avoiding overdependence on a single exact task or model.

### Features to emphasize

- client-side DP-SGD under record-level sensitivity control
- explicit privacy accounting and telemetry logging
- communication and model-size characterization in a federated healthcare workflow
- use of local clipping to bound per-record contributions before aggregation
- documentation of the implementation boundary: simulation with measurement, not adaptive runtime control

### Features to describe only as supporting implementation details

- Top-K sparsification/compression
- throughput and benchmark measurements
- edge benchmark or memory profiling
- metadata packaging and experiment registry generation
- validation-only threshold tuning

These are useful but not core to the strongest patent narrative unless linked to a more precise technical result.

### Features not to claim

- real-time adaptive resource optimization
- closed-loop privacy tuning
- hospital deployment or edge-device orchestration
- hardware-specific optimization claims
- broad readmission prediction novelty as a standalone concept
- any claim that telemetry itself is a control mechanism

---

## 10. REQUIRED FINAL OUTPUT

### A. Strongest technical differentiation

The strongest defensible technical differentiation is the specific integration of a sequential federated learning pipeline for hospital readmission prediction with client-side record-level DP-SGD, per-example gradient clipping, Gaussian perturbation, server-side averaging, and round-level privacy/communication/resource telemetry. This combination is materially more specific than generic FL or generic DP, but it is still not a single novel primitive; it is a carefully integrated research system whose real value lies in the operational and measurement integration rather than in any one algorithmic component alone.

### B. Three potential claim directions

1. System architecture claim direction
- A federated clinical prediction system in which local client devices or sites train on local records, apply per-example clipping and Gaussian noise, and periodically send updates to a central aggregator that combines them into a shared model.

2. Method/process claim direction
- A computer-implemented method for privacy-preserving distributed model training in which each local training step computes per-record gradients, clips them to a bound, adds Gaussian noise, and transmits aggregated updates to a coordinating server for model averaging while recording privacy and communication metrics.

3. Resource/privacy management claim direction
- A training system that records privacy, gradient, and communication telemetry during federated learning to characterize model and data flow behavior under privacy constraints, without requiring the telemetry to perform runtime control.

### C. Major prior-art threats

1. FedAvg and general federated learning literature
2. DP-SGD and differentially private deep learning literature
3. client-level DP federated learning literature
4. healthcare FL and medical privacy literature
5. communication-efficient / compressed FL literature

These are the largest overlap areas. They do not prove lack of novelty, but they show why the patent strategy should stay narrow and system-specific.

### D. Evidence still needed

1. A clear, final patent specification narrative tying the actual code to a precise system boundary
2. A close prior-art review in the exact combination space, especially FL + healthcare + record-level DP + communication telemetry
3. A more specific known-art comparison for any exact combination of per-example clipping, Gaussian noise, and metrics logging in healthcare FL
4. A stronger differentiation narrative if claim drafting intends to emphasize measurement/telemetry as more than a support feature

### E. Exact next step

STEP 4 should perform a narrower, claim-oriented prior-art review focused on the exact combination of:
- healthcare prediction,
- local record-level DP-SGD,
- federated aggregation,
- communication and privacy telemetry,
- and system-level implementation details.

The purpose of STEP 4 is to test whether the combination can be framed as a defensible, precise technical architecture rather than as a broad statement about FL or DP in medicine.

---

PATENT WORKFLOW: 3/9

This document is intentionally a technical differentiation analysis. It should be read as a patent-preparation support document, not a legal opinion or a conclusion that the system is novel or patentable.
