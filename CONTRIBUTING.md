# Contributing

1. Create a feature branch from `main`.
2. Use Python 3.12 and create a virtual environment.
3. Install development dependencies with `python -m pip install -e ".[dev]"`.
4. Run `python -m compileall -q agents.py engine.py execution.py main.py problems.py`.
5. Run `python -m pytest -q` and ensure all tests pass.
6. Keep API keys, `.env`, generated artifacts, private datasets, and local model files out of commits.
7. Open a pull request describing the change, tests run, and any limitations.

Changes to solver generation, artifact formats, verification, or scoring should include focused tests. Never weaken independent verification merely to make a generated candidate pass.
