# Project scope and reproduction status

This repository contains the implementation supplied in the project archive, with repository setup and documentation additions.

## Included

- Gemini REST client and optimization planning/code-generation agents.
- DAG-based search, stagnation tracking, parent selection, and optional semantic novelty.
- CPLEX/DOcplex-oriented solver generation and static checks.
- Independent verifier generation and artifact verification flow.
- Docker and local execution backends.
- MAX-CUT, Gset file loading, Circle Packing, Motorcycle Production, and user-defined problem support.
- Automated tests and a GitHub Actions test workflow.

## Not included / not claimed

- The paper's original benchmark datasets, including Gset70 and formulation benchmark files.
- The paper's original experiment environment, model configuration, solver configuration, or raw experiment logs.
- Reproduction of the paper's reported benchmark numbers or statistical results.
- A formal security boundary for arbitrary generated code.

The supplied implementation and its small-instance tests demonstrate the project workflow; they do not establish reproduction of the paper's published results. See the README for setup and run instructions.
