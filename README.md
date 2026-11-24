# OptMaster

A Gemini-backed, DAG-based optimization research prototype inspired by *OptMaster: A DAG-Based Framework for Formulation and Heuristic Discovery in Optimization* (ICML 2026). This repository is a runnable project implementation, not a reproduction package for the paper's full benchmark results.

## Repository status

- Python 3.12 project metadata and editable installation are configured.
- Git ignores, development instructions, security notes, and GitHub Actions CI are included.
- The automated test suite is key-free and does not require Docker or a CPLEX installation on the host.
- Real runs require a Gemini API key and Docker for the intended isolated solver execution.


## Compact project layout

This ZIP keeps the application in five Python files instead of separate package folders:

- `agents.py`: Gemini client, planning, solver/verifier generation, repair, scoring, summaries, and context helpers.
- `engine.py`: nodes, DAG, parent choice, stagnation, expansion, and semantic novelty.
- `execution.py`: Docker/local backends, verification, scoring, and diagnostics.
- `problems.py`: supported problem definitions and Gset reader.
- `main.py`: node lifecycle and command-line entry point.
- `tests/test_all.py`: combined automated tests.

The nested source modules are merged into these files. Only `tests` and `artifacts` remain as project folders.

OptMaster is a Gemini-backed optimization research prototype. Every generated optimization solver is required by prompt and code checks to use IBM CPLEX through DOcplex/CPLEX (or CP Optimizer for a suitable constraint-programming model). Gemini is the only model provider.

## Run on Windows

1. Install Python 3.12 and Docker Desktop. CPLEX's Python runtime compatibility depends on the Studio version; Python 3.12 is used for the supplied setup. Start Docker Desktop before running OptMaster.
2. In this folder, make the private key file:

   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```

   Replace `PASTE_YOUR_GEMINI_API_KEY_HERE` with your Gemini API key and save. `GEMINI_MODEL` defaults to `gemini-3.8-flash`; you can change it to another model available to your Gemini API project. Do not upload the `.env` file; it contains your API key.

3. Install the small host-side Python requirements and build the Docker solver image:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
   docker build -f Dockerfile.executor -t optmaster-python:3.12 .
   ```

   Host requirements are intentionally small. The Docker image installs DOcplex and IBM CPLEX Community Edition, which is free but limited to 1,000 variables and 1,000 constraints. The solver runs in Docker; no separate CPLEX install on Windows is needed.

4. Run a small MAX-CUT search:

   ```powershell
   .\.venv\Scripts\python.exe main.py
   ```

The default is one expansion round for a quick, low-dependency run. For the paper's multi-parent novelty selection, install the optional embedding package in the host environment with `python -m pip install -r requirements-embeddings.txt`; this adds the SentenceTransformer/PyTorch stack, and the model downloads the first time a multi-parent selection is needed. Gemini requests and model downloads require an internet connection. Artifacts are written under `artifacts/`.

Docker is the default execution backend for the paper's sandboxed generated-code execution. It disables network access and applies process, memory, and CPU limits. Without Docker, set `OPTMASTER_EXECUTOR=local` in `.env`; solver scripts then run directly on Windows with no sandbox. Local mode may be useful for a setup check, but it does not meet a sandboxed-execution requirement. A cluster-specific adapter is not included because the uploaded resources contain no cluster scheduler, endpoint, or credentials.

## Gemini API configuration

The default model is `gemini-3.8-flash`. For this model, the client omits the legacy `temperature` generation parameter because the current Gemini 3.8 Flash API guidance lists temperature controls among unsupported parameters. The agent-specific temperature preferences remain in the code as hints for other explicitly configured compatible models; they are not sent to Gemini 3.8 Flash.

## Paper mechanisms implemented

- Planning proposes distinct methods and can explore alternate interpretations when the statement is ambiguous.
- Generated optimizer scripts are statically checked to import `docplex` or `cplex`, call a solve API, and avoid other solver libraries. Prompts prohibit replacing CPLEX with a different solver.
- DOcplex builds the optimization model and invokes IBM CPLEX; CP Optimizer from the IBM Studio family is permitted for a suitable constraint-programming model.
- Nodes hold plans, code, artifacts, independent verifier reports, scores, and transferable summaries in a DAG.
- First-parent sampling uses stable Boltzmann/softmax reward selection. Additional parents are selected by reward plus stagnation-weighted SentenceTransformer cosine novelty.
- Stagnation controls the 1/2/3-parent refine, cross, and fuse expansion modes; the selected mode is included in the solver prompt.
- Candidate generation and verification run as separate code stages. The verifier receives the problem and artifact, not the solver source.
- Failed nodes enter a repair queue before the next expansion. Repair retains the same trajectory and expansion mode.
- Verified numeric objectives are used directly. When a feasible report has no reliable numeric objective, Gemini supplies a validated normalized score in [0, 1]; the search can stop at 0.94.
- Search rounds, deterministic search seeds, and parallel candidate count are configurable. The paper's experiment settings can be requested with `--max-steps 30 --parallelism 3`.
- Execution uses an interchangeable backend contract with Docker and local subprocess implementations.
- Minimal host setup avoids the heavyweight embedding stack for a one-round, single-parent run; install the optional embedding requirements for the full SentenceTransformer cosine-novelty mechanism.
- Formulation tasks can be supplied as JSON. The verifier checks feasibility and recomputes objectives from the artifact.

## Paper task examples

### MAX-CUT / Gset70

The default `max_cut` is a small graph for a quick run. The uploaded project resources do not contain the Gset70 instance file. If you obtain the unit-weight Gset70 edge file, pass its path:

```powershell
.\.venv\Scripts\python.exe main.py --problem gset70 --gset-file .\data\Gset70.txt --max-steps 30 --parallelism 3 --seed 42
```

The reader accepts either a `p edge N M` DIMACS header or an `N M` header, followed by unit-weight edge rows.

### Circle Packing

Circle Packing uses independent radii and maximizes their sum, with square-containment and pairwise non-overlap constraints. Its independent verifier checks the constraints and recomputes the sum. Run the paper's sizes as:

```powershell
.\.venv\Scripts\python.exe main.py --problem circle_packing --circles 26 --max-steps 30 --parallelism 3 --seed 42
.\.venv\Scripts\python.exe main.py --problem circle_packing --circles 32 --max-steps 30 --parallelism 3 --seed 42
```

### Other formulation tasks

Create a JSON file with `description`, `artifact_spec`, and optionally `reward_objective`, then run:

```powershell
.\.venv\Scripts\python.exe main.py --problem custom --problem-file .\my_problem.json
```

For a task without a directly measurable objective, omit `reward_objective` and the normalized evaluator is used after feasibility verification.

## Publish this project to GitHub

After extracting this ZIP, open a terminal in the extracted project root (the folder containing `main.py`) and run:

```powershell
git init
git add .
git commit -m "Initial OptMaster project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Create an empty GitHub repository first and replace the remote URL with your repository URL. Do not initialize the GitHub repository with an additional README, license, or `.gitignore`, because those files are already present here. Before pushing, confirm that `.env` is not staged (`git status --short`) and that no private datasets or generated artifacts have been added.

No open-source license is included. Unless the project owners choose and add a license, the repository should be treated as “all rights reserved”; public visibility alone does not grant reuse rights.

## Tests

The ZIP includes a key-free `.env.example`. Keep your real API key in `.env` and do not share that file.

Run tests locally:

```powershell
python -m pip install -r requirements-ci.txt
python -m pytest -q
```

Tests use a stubbed HTTP response for the Gemini client and do not need an API key or CPLEX license.

## Reproduction limits

The paper's benchmark files for IndustryOR, Mamo-Complex, OptMATH, ComplexOR, LogiOR, and Gset70 were not included in the uploaded project archives. The Gset reader and formulation entry point are implemented, but those benchmark files must be supplied to run those evaluations. Circle Packing formulation and feasibility verification support the paper's 26/27/32 sizes, but the paper's discovered heuristics, published objective values, multi-seed tables, and significance results have not been reproduced. Each run accepts an explicit seed; multi-seed aggregation must be run and reported separately. CPLEX supports LP/MIP and supported convex quadratic/QCP classes; general nonconvex nonlinear models may need a justified linearization or reformulation, and results can differ from the paper. Do not present this prototype's outputs as the paper's reported results.

## Execution safety

Docker execution disables network access, drops Linux capabilities, sets process/memory/CPU limits, uses a read-only container root, and mounts only a temporary run directory. This is a practical isolation layer, not a formal security guarantee. Keep API credentials in the host `.env`; they are not passed into solver containers. Local execution has no sandbox and should not be used for untrusted generated code.



