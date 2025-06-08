# Cost-Aware and Resilient Placement of Parallel Service Function Chains Using Multi-Stage Graph Modeling

This repository contains the full research code and thesis for the project:

(Cost-Aware and Resilient Placement of Parallel Service Function Chains Using Multi-Stage Graph Modeling
Himanshu Yadav, Indian Institute of Technology (BHU), Varanasi, 2025)

# Contents
* MSG_SFC_Placement.py
Python implementation of the Multi-Stage Graph (MSG) heuristic for cost-aware, delay-constrained, and reliable placement of parallel SFCs (Service Function Chains) with active and backup paths, and efficient VMF (Virtual Monitoring Function) placement.

* BruteForce_SFC_Placement.py
Brute-force baseline algorithm for optimal SFC placement (for small networks), used to validate the approximation and performance of the MSG heuristic.

* CORONET_and_PANEUR_topologies.xlsx
Realistic network topology dataset (CORONET) used for simulation and evaluation.

* Himanshu_final_thesis.pdf
Full thesis document (LaTeX source and PDF) describing the problem, methodology, experiments, and results.

# Features
* Parallel SFC Placement:
Supports parallel entities (PEs) within SFCs, with both active and disjoint backup path computation.

* Cost and Delay Optimization:
Minimizes total deployment cost (node activation, resource usage, link cost, backup delay) under resource and delay constraints.

* VMF Placement:
Greedy algorithm for efficient monitoring function deployment.

* Comparison Baseline:
Includes brute-force method for validation and approximation analysis.

* Extensive Documentation:
See thesis for full methodology, complexity analysis, and experimental results.

# Usage
$ Run MSG_SFC_Placement.py for heuristic placement and VMF deployment.

$ Run BruteForce_SFC_Placement.py for optimal placement on small networks.

$ Use the provided CORONET dataset or your own topology for evaluation.

$ Refer to Himanshu_final_thesis.pdf for detailed background, methodology, and analysis.
