

# ---- merged from agents\parsing.py ----

import json
import re

FENCE = "`" * 3

def extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(text[start:end + 1], strict=False)

def parse_solver_response(text: str) -> tuple[str, str]:
    plan_match = re.search(r"<plan>(.*?)</plan>", text, re.DOTALL)
    code_match = re.search(FENCE + r"python\s*(.*?)" + FENCE, text, re.DOTALL)
    if not code_match:
        raise ValueError("No python code block found")
    return (plan_match.group(1).strip() if plan_match else "", code_match.group(1).strip())


# ---- merged from agents\gemini_client.py ----

"""Minimal Gemini REST client; reads its credential only from the environment."""
import os
import requests


class GeminiClient:
    def __init__(self, model_name=None, timeout=120):
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("Set GEMINI_API_KEY in your environment or .env file.")
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        self.timeout = timeout

    def generate(self, prompt, system=None, temperature=None, json_mode=False):
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        generation = {}
        # Gemini 3.8 Flash no longer accepts temperature/top-p/top-k controls.
        # Keep temperature support for explicitly configured older compatible models.
        if temperature is not None and not self.model_name.startswith("gemini-3.8-flash"):
            generation["temperature"] = temperature
        if json_mode:
            generation["responseMimeType"] = "application/json"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        try:
            payload = {"contents": contents, "generationConfig": generation}
            if system:
                payload["systemInstruction"] = {"parts": [{"text": system}]}
            response = requests.post(url, headers={"x-goog-api-key": self.api_key},
                json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (requests.RequestException, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Gemini generation request failed: {exc}") from exc


# ---- merged from agents\solver_generator.py ----

import ast
from agents import parse_solver_response, FENCE

SOLVER_SYSTEM = "You are an expert optimization engineer who writes correct, runnable Python code."

SOLVER_PROMPT = """PROBLEM:
{problem}

ARTIFACT SPECIFICATION (you must follow it exactly):
{artifact_spec}

SELECTED TRAJECTORY:
- Paradigm: {paradigm}
- Interpretation: {interpretation}
- Approach: {approach}

EXPANSION MODE: {expansion_strategy}
Interpret the parent set according to this mode: refine = improve one approach;
cross = combine complementary ideas from two parents; fuse = integrate useful
components from three parents. Preserve all original problem constraints.
Random seed for this attempt: {seed}.

KNOWLEDGE FROM PARENT NODES:
{parents}

TASK:
Write a plan and a complete Python script that implements the selected trajectory.

Rules:
- Use IBM CPLEX only for optimization. For mathematical programming, build and solve the model
  with DOcplex (docplex.mp.Model) backed by IBM CPLEX. CP Optimizer is allowed only for a
  genuine constraint-programming formulation.
- Do not use PuLP/CBC, SciPy optimize, OR-Tools, Gurobi, or a non-CPLEX solver. NumPy may only
  format the final artifact, not replace CPLEX optimization.
- For unsupported nonconvex/general nonlinear models, state the limitation and use a justified
  CPLEX-compatible linearization or supported convex quadratic model. Never claim unsupported solving.
- Follow the trajectory's interpretation faithfully, and enforce ALL constraints stated in the problem.
- The script must be self-contained, fix random seeds, and finish in under 60 seconds.
- If the interpretation has a secondary objective, you MUST implement it in code (for example lexicographic: solve the primary objective, fix its optimal value as a constraint, then optimize the secondary objective).
- Integer quantities must be saved as Python ints (use int(round(x))).
- Save the artifact exactly as specified. Do not print anything except one final line: DONE

Respond in exactly this format and nothing else:
<plan>
Key modeling decisions: variables, objective, constraints, algorithm (short).
</plan>
""" + FENCE + """python
# full script here
""" + FENCE


def format_parents(parent_contexts):
    if not parent_contexts:
        return "None (this is an initial node)."
    return "\n\n".join(parent_contexts)


class SolverGenerator:

    def __init__(self, llm, max_retries=2):
        self.llm = llm
        self.max_retries = max_retries

    def generate(self, problem, trajectory, parent_contexts=None, expansion_strategy="refine", seed=42):
        prompt = SOLVER_PROMPT.format(
            problem=problem["description"],
            artifact_spec=problem["artifact_spec"],
            paradigm=trajectory["paradigm"],
            interpretation=trajectory["interpretation"],
            approach=trajectory["approach"],
            parents=format_parents(parent_contexts),
            expansion_strategy=expansion_strategy,
            seed=seed,
        )
        last_error = None

        for attempt in range(1, self.max_retries + 2):
            raw = self.llm.generate(prompt, system=SOLVER_SYSTEM, temperature=0.4)
            try:
                plan, code = parse_solver_response(raw)
                ast.parse(code)
                validate_cplex_solver_code(code)
                return plan, code
            except (ValueError, SyntaxError) as e:
                last_error = e
                print(f"Attempt {attempt} failed: {e}")

        raise RuntimeError(f"SolverGenerator failed after retries: {last_error}")


def validate_cplex_solver_code(code):
    """Reject generated scripts that don't call into IBM's optimization stack."""
    tree = ast.parse(code)
    imported = set()
    calls = []
    forbidden = {"pulp", "scipy", "ortools", "gurobipy", "pyomo"}
    for item in ast.walk(tree):
        if isinstance(item, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in item.names)
        elif isinstance(item, ast.ImportFrom) and item.module:
            imported.add(item.module.split(".")[0])
        elif isinstance(item, ast.Call):
            if isinstance(item.func, ast.Attribute):
                calls.append(item.func.attr)
            elif isinstance(item.func, ast.Name):
                calls.append(item.func.id)
    if not ({"docplex", "cplex"} & imported):
        raise ValueError("Solver must import DOcplex or the IBM CPLEX Python API")
    if imported & forbidden:
        raise ValueError("Solver must not use non-CPLEX optimization libraries")
    if "solve" not in calls:
        raise ValueError("Solver must call the CPLEX/DOcplex solve API")


# ---- merged from agents\planner.py ----

import json
from agents import extract_json

PLANNER_SYSTEM = "You are an expert in operations research and optimization."

PLANNER_PROMPT = """Problem description:
{problem}

Propose {n} DISTINCT solution trajectories for this problem.

Trajectories may differ in:
- HOW to solve using IBM's optimization stack: alternative CPLEX/DOcplex formulations,
  relaxations used to guide CPLEX, different CPLEX settings, or CPLEX-based methods.
- WHAT to solve: different readings of parts that are genuinely ambiguous, such as which objective has priority or how multiple objectives are combined.

Rules:
- NEVER relax, soften or drop a constraint that the problem states explicitly (e.g. "minimum", "at most", "must"). Every trajectory must satisfy all stated constraints.
- Only vary the interpretation where the text is truly ambiguous. If it is clear, keep the same interpretation and vary the solving approach instead.
- Every trajectory must use IBM CPLEX Optimization Studio engines through DOcplex/CPLEX
  (or CP Optimizer when a constraint-programming formulation is suitable). Do not use PuLP,
  SciPy optimize, OR-Tools, Gurobi, or a standalone non-CPLEX heuristic.
- If a model class is not directly supported, propose a justified CPLEX-compatible linear,
  mixed-integer, convex quadratic, or piecewise-linear formulation; never silently switch solvers.
- If the problem has several conflicting objectives, trajectories must differ in which objective is prioritized (or how they are combined), so that every reasonable priority is covered at least once.
- Each "approach" must be consistent with its "interpretation".
- NEVER add requirements, constraints or secondary objectives that the problem does not state (for example "balanced sets" or "minimize the number of changes"). If the problem is clear, every trajectory keeps the same interpretation.
- Trajectories must differ in the solving METHOD, not just in wording. Do not propose two trajectories that use the same paradigm and the same technique. Different heuristics (for example greedy, local search, simulated annealing) count as different methods.

Return ONLY a JSON object in this format:
{{
  "trajectories": [
    {{
      "id": 1,
      "paradigm": "solver" or "heuristic",
      "interpretation": "how this trajectory reads the problem",
      "approach": "the planned solving approach in 2-3 sentences",
      "confidence": a number between 0 and 1
    }}
  ]
}}"""


class Planner:

    def __init__(self, llm, max_retries=2):
        self.llm = llm
        self.max_retries = max_retries

    def plan(self, problem, n=3):
        prompt = PLANNER_PROMPT.format(problem=problem, n=n)
        last_error = None

        for attempt in range(1, self.max_retries + 2):
            raw = self.llm.generate(
                prompt,
                system=PLANNER_SYSTEM,
                temperature=0.8,
                json_mode=True,
            )
            try:
                data = extract_json(raw)
                trajectories = data["trajectories"]
                for t in trajectories:
                    for key in ("paradigm", "interpretation", "approach"):
                        if key not in t:
                            raise KeyError(f"missing '{key}'")
                return trajectories
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                last_error = e
                print(f"Attempt {attempt} failed: {e}")

        raise RuntimeError(f"Planner failed after retries: {last_error}")


# ---- merged from agents\repair_agent.py ----

import ast
from agents import parse_solver_response, FENCE
from agents import validate_cplex_solver_code

REPAIR_SYSTEM = "You are an expert optimization engineer who fixes broken Python code without changing its strategy."

REPAIR_PROMPT = """PROBLEM:
{problem}

ARTIFACT SPECIFICATION (you must follow it exactly):
{artifact_spec}

STRATEGY TO KEEP (do NOT change it):
- Paradigm: {paradigm}
- Interpretation: {interpretation}
- Approach: {approach}
Random seed for this attempt: {seed}

KNOWLEDGE FROM THE FAILED NODE:
{parent_context}

FAILED CODE:
{failed_code}

DIAGNOSTICS (what went wrong when this code was executed and checked):
{diagnostics}

TASK:
Fix the failed code so that it runs and produces a valid artifact. This is a REPAIR: keep the same strategy, interpretation and approach, and change only what is needed to fix the problem shown in the diagnostics.

Rules:
- Do NOT switch to a different paradigm or reinterpret the problem.
- Do NOT relax, soften or drop any constraint stated in the problem.
- Keep optimization in IBM CPLEX/DOcplex (or CP Optimizer for a genuine constraint-programming model). Never switch to PuLP/CBC, SciPy optimize, OR-Tools, Gurobi, or a standalone heuristic.
- For unsupported nonlinear models, explain the limitation and use a justified CPLEX-compatible linearization or supported convex quadratic formulation.
- The script must be self-contained, fix random seeds, and finish in under 60 seconds.
- Integer quantities must be saved as Python ints (use int(round(x))).
- Save the artifact exactly as specified. Do not print anything except one final line: DONE

Respond in exactly this format and nothing else:
<plan>
What was wrong and what you changed (short).
</plan>
""" + FENCE + """python
# full corrected script here
""" + FENCE


class RepairAgent:

    def __init__(self, llm, max_retries=2):
        self.llm = llm
        self.max_retries = max_retries

    def repair(self, problem, trajectory, parent_context, failed_code, diagnostics, seed=42):
        base_prompt = REPAIR_PROMPT.format(
            problem=problem["description"],
            artifact_spec=problem["artifact_spec"],
            paradigm=trajectory["paradigm"],
            interpretation=trajectory["interpretation"],
            approach=trajectory["approach"],
            parent_context=parent_context or "(none)",
            failed_code=failed_code,
            diagnostics=str(diagnostics)[-1500:],
            seed=seed,
        )
        prompt = base_prompt
        last_error = None

        for attempt in range(1, self.max_retries + 2):
            raw = self.llm.generate(prompt, system=REPAIR_SYSTEM, temperature=0.3)
            try:
                plan, code = parse_solver_response(raw)
                ast.parse(code)
                validate_cplex_solver_code(code)
                if code.strip() == failed_code.strip():
                    raise ValueError("the code is identical to the failed code, nothing was fixed")
                if "savez" not in code:
                    raise ValueError("the code never saves the artifact with numpy.savez")
                return plan, code
            except (ValueError, SyntaxError) as e:
                last_error = e
                prompt = base_prompt + f"\n\nYour previous answer was REJECTED because: {e}\nFix this and follow the format exactly."

        raise RuntimeError(f"RepairAgent failed after retries: {last_error}")


# ---- merged from agents\scorer.py ----

"""Normalized evaluator for tasks without a directly comparable numeric metric."""
import json
from agents import extract_json


SCORER_SYSTEM = "You are a careful optimization evaluator. Score only from the problem and independent verifier report."
SCORER_PROMPT = """Score this candidate from 0.0 to 1.0 for progress toward the stated objective.
Do not reward unverified claims or infeasible solutions. Compare the original problem with
the independent verifier report. Do not inspect or score source code.

PROBLEM:
{problem}

VERIFIER REPORT:
{report}

Return JSON only: {{"score": number between 0 and 1, "rationale": "one short sentence"}}"""


class LLMScorer:
    def __init__(self, llm):
        self.llm = llm

    def score(self, problem, report):
        if not report.get("feasible", False):
            return 0.0, "Verifier marked the candidate infeasible."
        raw = self.llm.generate(
            SCORER_PROMPT.format(problem=problem["description"], report=json.dumps(report)),
            system=SCORER_SYSTEM, temperature=0.0, json_mode=True)
        result = extract_json(raw)
        score = float(result["score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError("LLM evaluator score must be in [0, 1]")
        return score, str(result.get("rationale", ""))


# ---- merged from agents\summarizer.py ----

import json
import re
from agents import FENCE

SUMMARIZER_SYSTEM = "You are a precise technical writer who summarizes optimization experiments factually."

SUMMARIZER_PROMPT = """You are summarizing one node of an optimization search. Write a short summary for FUTURE nodes that will build on it.

PROBLEM:
{problem}

PLAN (P):
Paradigm: {paradigm}
Interpretation: {interpretation}
Approach: {approach}
Modeling notes: {plan}

SOLVER CODE (C), for your reading only:
{code}

VERIFIED REPORT (V), produced independently by a verifier:
{report}

Write the summary in exactly 3 short parts, plain text, 4 to 5 sentences total:
1. DECISIONS: the key modeling decisions that matter for follow-up nodes (1-2 sentences).
2. OUTCOME: what worked or what failed, based only on the report (1-2 sentences).
3. LESSON: one concrete suggestion for a follow-up node (1 sentence).

Strict rules:
- Do NOT include any code, code fragments, function names, or code blocks.
- Do NOT repeat numeric objective values; they are attached separately.
- The verifier checks feasibility and recomputes objectives. It does NOT prove optimality, so never say a solution is optimal or best.
- Every constraint in the problem is mandatory. The LESSON must never suggest relaxing, softening, dropping or ignoring any stated constraint (including integrality). Suggest other ways to improve instead (different approach, different search, better enforcement of constraints)."""


def contains_raw_code(text):
    if FENCE in text:
        return "contains a code block"
    for line in text.splitlines():
        s = line.strip()
        if re.match(r"^(def |import |from \S+ import |for .+:|while .+:|class |if .+:|return\b)", s):
            return f"contains a code line: {s[:40]}"
        if re.search(r"\b(np|pulp|scipy|docplex|cplex)\.\w+\(", s):
            return f"contains a library call: {s[:40]}"
    return None


def contains_bad_claims(text):
    m = re.search(r"\b(relax\w*|(?:is|are|was|were|proven|globally|provably)\s+optimal|optimal\s+solution)\b", text, re.IGNORECASE)
    if m:
        return f"contains a forbidden claim or suggestion: '{m.group(0)}'"
    return None


class Summarizer:

    def __init__(self, llm, max_retries=2, max_chars=800, max_code_chars=6000):
        self.llm = llm
        self.max_retries = max_retries
        self.max_chars = max_chars
        self.max_code_chars = max_code_chars

    def summarize(self, problem, trajectory, plan, code, report):
        base_prompt = SUMMARIZER_PROMPT.format(
            problem=problem["description"],
            paradigm=trajectory["paradigm"],
            interpretation=trajectory["interpretation"],
            approach=trajectory["approach"],
            plan=plan or "(none)",
            code=code[: self.max_code_chars],
            report=json.dumps(report, indent=2),
        )
        prompt = base_prompt
        last_error = None

        for attempt in range(1, self.max_retries + 2):
            text = self.llm.generate(prompt, system=SUMMARIZER_SYSTEM, temperature=0.2).strip()
            reason = contains_raw_code(text) or contains_bad_claims(text)
            if reason is None and len(text) > self.max_chars:
                reason = f"is too long ({len(text)} chars, max {self.max_chars})"
            if reason is None and len(text) < 40:
                reason = "is too short"
            if reason is None:
                return text
            last_error = reason
            print(f"Attempt {attempt} rejected: {reason}")
            prompt = base_prompt + f"\n\nYour previous summary was REJECTED because it {reason}. Rewrite it following all rules."

        raise RuntimeError(f"Summarizer failed after retries: {last_error}")


# ---- merged from agents\verifier_generator.py ----

import ast
from agents import parse_solver_response

VERIFIER_SYSTEM = "You are a meticulous verification engineer who writes correct, runnable Python code."

VERIFIER_PROMPT = """PROBLEM:
{problem}

ARTIFACT SPECIFICATION:
{artifact_spec}

TASK:
Write ONE Python function named verify(artifact) that independently verifies a saved solution.
`artifact` is a dict mapping each saved key to a plain Python value (numbers or lists), already loaded from the artifact file. Do not load files, do not print, do not use json.

Rules:
- Allowed imports: numpy and the standard library only. Do NOT solve or optimize the problem.
- Use the numbers given in the problem statement and a tolerance of 1e-6.
- Resource limits (hours, budget, capacity) are constraints too: if total usage exceeds the limit, that is a violation.
- A mismatch between a claimed_* value and the recomputed value is NOT a constraint violation. It goes only into claim_errors.
- Check integrality with a tolerance: abs(v - round(v)) <= 1e-6. NEVER use isinstance(v, int) for this, because integer values may arrive as floats such as 25.0.
- The only allowed early return is when a decision key is missing, or when a decision value has the wrong shape or length (for example fewer entries than the artifact specification says): add a clear violation message and return with empty objectives.

Follow EXACTLY this structure:

def verify(artifact):
    violations = []
    claim_errors = []
    # STEP 0 (comments only): number EVERY constraint stated in the problem:
    #   C1: ...
    #   C2: ...
    #   (include bounds, minimums, maximums, resource limits, integrality, non-negativity)
    # STEP 1: read the raw decision values from the artifact.
    #         If a key is missing or its shape/length is wrong, add a violation and return early with empty objectives.
    # STEP 2: recompute ALL objectives into the dict `objectives`. Always do this, even if the solution is infeasible.
    objectives = {{}}
    # STEP 3: one separate `if` per constraint C1, C2, ... (never elif, never return early).
    #         Each failing check appends a short message to `violations`.
    # STEP 4: for each claimed_* key in the artifact, compare with the recomputed value.
    #         If different (or missing), append a short message to `claim_errors`.
    return {{
        "feasible": len(violations) == 0,
        "objectives": objectives,
        "claims_match": len(claim_errors) == 0,
        "violations": violations,
        "diagnostics": {{"claim_errors": claim_errors}},
    }}

Add useful numbers to diagnostics (slacks, resource utilization, recomputed vs claimed values).
Respond with ONLY one python code block containing the imports and the function."""

VERIFIER_HARNESS = '''

import json as _json
import sys as _sys
import numpy as _np

def _to_py(o):
    if isinstance(o, _np.generic):
        return o.item()
    if isinstance(o, _np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): _to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_to_py(v) for v in o]
    return o

def _fail(msg, diag=None):
    return {"feasible": False, "objectives": {}, "claims_match": False,
            "violations": [msg], "diagnostics": diag or {}}

if __name__ == "__main__":
    try:
        _data = _np.load("solution.npz")
        _artifact = {k: _to_py(_data[k]) for k in _data.files}
    except Exception as _e:
        print(_json.dumps(_fail("cannot load solution.npz: " + str(_e))))
        _sys.exit(0)
    try:
        _report = _to_py(verify(_artifact))
        if not _report.get("objectives"):
            _report["claims_match"] = False
    except Exception as _e:
        _report = _fail("verifier error: " + repr(_e), {"verifier_error": True})
    print(_json.dumps(_report))
'''


def check_verifier_style(code):
    problems = []
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            problems.append("elif chain found (line %d): use a separate `if` for every constraint" % node.lineno)
    return problems


class VerifierGenerator:

    def __init__(self, llm, max_retries=3):
        self.llm = llm
        self.max_retries = max_retries

    def generate(self, problem):
        base_prompt = VERIFIER_PROMPT.format(
            problem=problem["description"],
            artifact_spec=problem["artifact_spec"],
        )
        prompt = base_prompt
        last_error = None

        for attempt in range(1, self.max_retries + 2):
            raw = self.llm.generate(prompt, system=VERIFIER_SYSTEM, temperature=0.1)
            try:
                _, code = parse_solver_response(raw)
                ast.parse(code)
                if "def verify(" not in code:
                    raise ValueError("function verify() not found")
                problems = check_verifier_style(code)
                if problems:
                    raise ValueError("; ".join(problems))
                return code + VERIFIER_HARNESS
            except (ValueError, SyntaxError) as e:
                last_error = e
                print(f"Attempt {attempt} failed: {e}")
                prompt = base_prompt + f"\n\nYour previous answer was REJECTED because: {e}\nFix this and follow the structure exactly."

        raise RuntimeError(f"VerifierGenerator failed after retries: {last_error}")


# ---- merged from agents\context_builder.py ----

def compact_report(report, max_items=3):
    objs = report.get("objectives") or {}
    viol = (report.get("violations") or [])[:max_items]
    claim_errs = ((report.get("diagnostics") or {}).get("claim_errors") or [])[:max_items]
    return (f"feasible={report.get('feasible')}, claims_match={report.get('claims_match')}, "
            f"objectives={objs if objs else 'none'}, "
            f"violations={viol if viol else 'none'}, claim_errors={claim_errs if claim_errs else 'none'}")


def build_node_context(node_id, trajectory, report, summary):
    return (f"[Parent node {node_id}]\n"
            f"Strategy: {trajectory['paradigm']} | {trajectory['interpretation']}\n"
            f"Approach: {trajectory['approach']}\n"
            f"Verified result: {compact_report(report)}\n"
            f"Summary: {summary}")


def build_parent_contexts(parents):
    return [build_node_context(p["id"], p["trajectory"], p["report"], p["summary"]) for p in parents]


# ---- merged from common\config.py ----

"""Load the local Gemini credential and create the required Gemini client."""
import os
from pathlib import Path


def load_dotenv(path=Path(".env")):
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def make_llm():
    load_dotenv()
    from agents import GeminiClient
    return GeminiClient()

