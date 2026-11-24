# OptMaster

A Gemini-backed, DAG-based optimization research prototype inspired by *OptMaster: A DAG-Based Framework for Formulation and Heuristic Discovery in Optimization* (ICML 2026).

This repository contains a runnable implementation of selected ideas from the paper. It is not a reproduction package for the paper's full benchmark results.

## Overview

OptMaster explores optimization approaches using a directed acyclic graph (DAG) of candidate solution attempts. Each node represents a candidate with its plan, generated solver code, execution artifact, independent verification report, score, and summary.

The system combines:
- LLM-assisted planning and solver-code generation.
- Static checks on generated Python solver code.
- Execution through Docker or a local subprocess backend.
- Independently generated verification code.
- Repair attempts based on execution and verification diagnostics.
- Reward-based parent selection and stagnation-aware exploration.
- Optional semantic novelty using SentenceTransformer embeddings.

Gemini is the model provider used by this implementation. Generated optimization solver code is checked and prompted to use IBM CPLEX through DOcplex/CPLEX.

## Architecture

The project is organized into five main Python modules:

| File | Responsibility |
|---|---|
| `main.py` | Command-line entry point and node lifecycle orchestration. |
| `agents.py` | Gemini client, planner, solver and verifier generation, repair, scoring, and summaries. |
| `engine.py` | Search DAG, nodes, parent selection, stagnation tracking, expansion control, and semantic novelty. |
| `execution.py` | Docker/local execution backends, MAX-CUT verification helpers, scoring, and diagnostics. |
| `problems.py` | Optimization problem definitions, including MAX-CUT, Circle Packing, custom problems, and Gset loading. |

Additional project files:

- `tests/test_all.py` — automated test suite.
- `Dockerfile.executor` — environment for executing generated solver scripts.
- `.env.example` — example environment configuration.
- `requirements-ci.txt` — lightweight test dependencies.
- `requirements-embeddings.txt` — optional semantic-embedding dependencies.
- `docs/PROJECT_SCOPE.md` — implementation scope and limitations.

## Requirements

- Python 3.12
- A Gemini API key for live LLM-backed runs
- Docker Desktop or Docker Engine for isolated solver execution

The Docker image installs DOcplex and IBM CPLEX Community Edition. The Community Edition has model-size limits, including limits on variables and constraints. A separate host-side CPLEX installation is not required for the supplied Docker execution setup.

The automated tests do not require a Gemini API key, Docker, or a host CPLEX installation.

## Configuration

Create a local `.env` file from the supplied example:

```powershell
Copy-Item .env.example .env
```

Add your Gemini API key to `.env`. The default model setting is defined by `GEMINI_MODEL` in `.env.example`; change it if you want to use another model available to your API project.

Keep `.env` private. It is excluded from version control by `.gitignore`. Never commit API keys or other credentials.

Relevant runtime settings include:

| Environment variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Gemini API authentication. |
| `GEMINI_MODEL` | Gemini model identifier. |
| `OPTMASTER_EXECUTOR` | Execution backend: `docker` or `local`. |
| `OPTMASTER_TIMEOUT_SECONDS` | Solver execution timeout. |
| `OPTMASTER_MAX_REPAIRS` | Maximum repair attempts configured for the lifecycle. |
| `OPTMASTER_PARALLELISM` | Candidate evaluation parallelism. |

## Installation

Create and activate a Python 3.12 virtual environment, then install the project:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Build the solver execution image:

```powershell
docker build -f Dockerfile.executor -t optmaster-python:3.12 .
```

## Running OptMaster

Run the default small MAX-CUT example:

```powershell
.\.venv\Scripts\python.exe main.py
```

The default problem is a small graph intended for a quick functional run. Generated artifacts are written under `artifacts/`.

Docker is the default execution backend. It runs generated solver scripts in a restricted container with network access disabled and resource limits applied.

To use the local subprocess backend instead, set the following in `.env`:

```dotenv
OPTMASTER_EXECUTOR=local
```

Local execution is useful for development but does not provide the same isolation as Docker and should not be used for untrusted generated code.

## Optional semantic novelty

The SentenceTransformer embedding stack is optional. Install it to enable semantic cosine-distance calculations during multi-parent selection:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-embeddings.txt
```

The embedding model may be downloaded the first time it is needed, so an internet connection is required for that initial download.

## Supported problem examples

### MAX-CUT and Gset instances

The default MAX-CUT problem is a small example. A Gset instance file is not bundled with this repository. If you have a compatible unit-weight Gset70 edge file, run:

```powershell
.\.venv\Scripts\python.exe main.py --problem gset70 --gset-file .\data\Gset70.txt --max-steps 30 --parallelism 3 --seed 42
```

The reader accepts either a DIMACS-style `p edge N M` header or an `N M` header, followed by unit-weight edge rows.

### Circle Packing

Circle Packing maximizes the sum of circle radii subject to square-containment and pairwise non-overlap constraints. Example runs:

```powershell
.\.venv\Scripts\python.exe main.py --problem circle_packing --circles 26 --max-steps 30 --parallelism 3 --seed 42
.\.venv\Scripts\python.exe main.py --problem circle_packing --circles 32 --max-steps 30 --parallelism 3 --seed 42
```

### Custom formulation problems

Custom problems can be supplied as JSON containing a `description`, an `artifact_spec`, and optionally a `reward_objective`:

```powershell
.\.venv\Scripts\python.exe main.py --problem custom --problem-file .\my_problem.json
```

The repository includes `custom_problem.template.json` as a starting point.

## Search and verification approach

The search engine maintains candidate nodes in a DAG. It selects parents using reward-based sampling and, when multiple parents are selected, can incorporate semantic distance between candidate plans. A stagnation tracker adjusts the probability of refinement, crossover, and fusion-style expansion.

For each candidate, the lifecycle broadly performs the following steps:

1. Generate a solver plan and Python implementation.
2. Apply static checks to the generated code.
3. Execute the solver and collect its artifact.
4. Run an independent verifier against the artifact.
5. Attempt repairs when execution or verification fails.
6. Calculate a score from the verified objective, using a normalized LLM-based fallback only when needed.
7. Summarize the candidate for transfer into later search steps.

Static syntax checks do not establish mathematical correctness. The independent verifier is responsible for checking feasibility and recomputing objective values from the produced artifact.

## Tests

Install the lightweight test dependencies and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ci.txt
.\.venv\Scripts\python.exe -m pytest -q
```

The tests use mocked Gemini HTTP responses where applicable, so they do not require a live API key.

## Scope and reproduction limitations

The benchmark datasets for IndustryOR, Mamo-Complex, OptMATH, ComplexOR, LogiOR, and Gset70 were not included in the supplied project resources. The Gset reader and formulation entry point are present, but the corresponding benchmark files must be provided separately.

The implementation includes Circle Packing formulation and feasibility-verification support. However, the paper's discovered heuristics, published objective values, multi-seed tables, and statistical results have not been reproduced by this repository. Results from this prototype must not be represented as the paper's reported results.

The supplied execution setup uses IBM CPLEX Community Edition, which has model-size restrictions. Some general nonconvex nonlinear optimization formulations may require reformulation or may not be supported by the selected solver.

## Security considerations

Generated solver code is untrusted input. Docker execution applies practical isolation measures, including disabled network access, dropped Linux capabilities, a read-only container root, and CPU, memory, and process limits. These measures are not a formal security guarantee.

API credentials remain in the host environment and should not be passed into solver containers. The local execution backend does not provide a security sandbox.

## License

No open-source license is currently included. Unless the project owners add a license, the repository should be treated as all rights reserved. Public visibility does not by itself grant permission to reuse, modify, or redistribute the code.
