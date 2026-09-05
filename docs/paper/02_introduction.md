# 02. Introduction

Hospital readmission prediction is an important clinical modeling task, but patient data are distributed across institutions and are not easily centralized. This motivates federated learning: models can learn from local hospital datasets without sharing raw records. However, federated training introduces communication constraints, heterogeneity, and privacy risks. This paper presents FedLite-Care, a lightweight framework designed to evaluate the tradeoff between privacy and utility in a federated clinical setting.

The current project includes a verified DP-SGD pilot and a calibrated sigma=2.3 target. The project remains explicit that the sigma=2.3 utility characterization is pending final verified completion, instead of fabricating results.
