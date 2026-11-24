

# ---- merged from execution\base.py ----

"""Common contract for interchangeable node execution backends."""
from pathlib import Path
from typing import Protocol


class ExecutionBackend(Protocol):
    def run_script(self, code: str, working_directory: Path) -> dict:
        """Run one script and return success, stdout, stderr, and returncode."""


# ---- merged from execution\contracts.py ----


def create_maxcut_problem(nodes, edges):
    """Build a detached, normalized graph payload for a MAX-CUT check."""
    vertex_ids = list(nodes)
    edge_pairs = []
    for edge in edges:
        if len(edge) != 2:
            raise ValueError("Every MAX-CUT edge must contain two endpoints")
        left, right = edge
        edge_pairs.append((left, right))
    return {"problem_type": "MAXCUT", "nodes": vertex_ids, "edges": edge_pairs}


def create_solver_result(solution):
    """Wrap an assignment while keeping later caller changes from mutating it."""
    try:
        assignment = dict(solution)
    except (TypeError, ValueError) as exc:
        raise ValueError("A solver result must be a node-to-group mapping") from exc
    return {"solution": assignment}


# ---- merged from execution\diagnostics.py ----


def create_diagnostics(execution_report, verification_report=None):
    """Summarize execution and verifier state for logs and repair prompts."""
    was_successful = execution_report.get("success") is True
    state = execution_report.get("status") or ("success" if was_successful else "failed")
    messages = []
    details = {
        "execution_status": state,
        "runtime": execution_report.get("runtime"),
        "verification_status": "not_checked",
        "score": None,
        "messages": messages,
    }
    if state != "success" and not was_successful:
        reason = execution_report.get("error") or execution_report.get("stderr") or "unknown error"
        messages.append(f"Execution failed: {reason}")
        return details

    messages.append("Execution completed successfully.")
    if verification_report is None:
        return details

    accepted = bool(verification_report.get("valid", verification_report.get("feasible", False)))
    if not accepted:
        details["verification_status"] = "invalid"
        messages.extend(verification_report.get("errors") or verification_report.get("violations") or [])
        return details

    details["verification_status"] = "valid"
    values = verification_report.get("objectives") or {}
    details["score"] = verification_report.get("cut_value")
    if details["score"] is None and values:
        details["score"] = next(iter(values.values()))
    messages.append("Solution passed verification.")
    return details


# ---- merged from execution\docker_executor.py ----

"""Optional Docker backend with network disabled and resource limits."""
from pathlib import Path
import subprocess


class DockerExecutor:
    def __init__(self, timeout_seconds=30, image="optmaster-python:3.12"):
        self.timeout_seconds, self.image = timeout_seconds, image

    def run_script(self, code, working_directory):
        work = Path(working_directory).resolve()
        (work / "solver.py").write_text(code, encoding="utf-8")
        args = ["docker", "run", "--rm", "--network", "none", "--read-only",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--pids-limit", "128", "--memory", "1g", "--cpus", "1",
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m", "-v",
                f"{work}:/work:rw", "-w", "/work", self.image, "python", "solver.py"]
        try:
            result = subprocess.run(args, capture_output=True, text=True,
                                    timeout=self.timeout_seconds, check=False)
            return {"returncode": result.returncode, "stdout": result.stdout,
                    "stderr": result.stderr, "success": result.returncode == 0 and
                    (work / "solution.npz").exists()}
        except subprocess.TimeoutExpired as exc:
            return {"returncode": None, "stdout": exc.stdout or "", "stderr": "timeout", "success": False}
        except OSError as exc:
            return {"returncode": None, "stdout": "", "stderr": str(exc), "success": False}


# ---- merged from execution\executor.py ----


import time

def execute_solver(solver_function, problem_data):
    """Run a trusted in-process callable and return a stable execution record."""
    started = time.perf_counter()
    try:
        value = solver_function(problem_data)
    except Exception as exc:
        return {
            "status": "failed", "result": None,
            "runtime": time.perf_counter() - started,
            "error": str(exc), "error_type": type(exc).__name__,
        }
    return {
        "status": "success", "result": value,
        "runtime": time.perf_counter() - started,
        "error": None, "error_type": None,
    }


# ---- merged from execution\local_executor.py ----

"""Local subprocess backend for trusted experiments; not an isolation boundary."""
from pathlib import Path
import subprocess
import sys


class LocalExecutor:
    def __init__(self, timeout_seconds=30):
        self.timeout_seconds = timeout_seconds

    def run_script(self, code, working_directory):
        work = Path(working_directory).resolve()
        script = work / "solver.py"
        script.write_text(code, encoding="utf-8")
        try:
            result = subprocess.run([sys.executable, str(script)], cwd=work,
                capture_output=True, text=True, timeout=self.timeout_seconds, check=False)
            return {"returncode": result.returncode, "stdout": result.stdout,
                    "stderr": result.stderr, "success": result.returncode == 0 and
                    (work / "solution.npz").is_file()}
        except subprocess.TimeoutExpired as exc:
            return {"returncode": None, "stdout": exc.stdout or "",
                    "stderr": f"Execution exceeded {self.timeout_seconds} seconds", "success": False}
        except OSError as exc:
            return {"returncode": None, "stdout": "", "stderr": str(exc), "success": False}


# ---- merged from execution\scorer.py ----

def calculate_score(verification_report, objective=None, invalid_score=-1.0):
    """Score only independently verified feasible reports."""
    if not verification_report.get("feasible", verification_report.get("valid", False)):
        return invalid_score
    values = verification_report.get("objectives") or {}
    if objective is not None:
        if objective in values:
            return float(values[objective])
        if objective == "cut_value" and "cut_value" in verification_report:
            return float(verification_report["cut_value"])
        raise KeyError(f"Verified objective {objective!r} is missing")
    if values:
        return float(next(iter(values.values())))
    if "cut_value" in verification_report:
        return float(verification_report["cut_value"])
    raise ValueError("No independently verified objective is available")


# ---- merged from execution\verifier.py ----


def verify_maxcut_solution(problem_data, solver_result):
    """Independently validate a binary partition and recompute its cut value."""
    vertices = list(problem_data["nodes"])
    edges = list(problem_data["edges"])
    raw_assignment = (solver_result or {}).get("solution", {})
    issues = []
    if not isinstance(raw_assignment, dict):
        assignment = {}
        issues.append("Solution must be a node-to-group mapping")
    else:
        assignment = raw_assignment

    expected, supplied = set(vertices), set(assignment)
    issues.extend(f"Node {node} is missing an assignment" for node in vertices if node not in supplied)
    issues.extend(f"Unknown node {node} in solution" for node in assignment if node not in expected)
    issues.extend(
        f"Node {node} has an invalid group: {group}"
        for node, group in assignment.items()
        if group not in (0, 1)
    )
    for edge in edges:
        if len(edge) != 2 or edge[0] not in expected or edge[1] not in expected:
            issues.append(f"Invalid graph edge: {edge!r}")

    if issues:
        return {
            "valid": False, "feasible": False, "cut_value": None,
            "objectives": {}, "claims_match": False, "cut_edges": [],
            "errors": issues, "violations": issues,
            "diagnostics": {"claim_errors": []},
        }

    crossing_edges = [(u, v) for u, v in edges if assignment[u] != assignment[v]]
    value = len(crossing_edges)
    return {
        "valid": True, "feasible": True, "cut_value": value,
        "objectives": {"cut_value": value}, "claims_match": True,
        "cut_edges": crossing_edges, "errors": [], "violations": [],
        "diagnostics": {"claim_errors": []},
    }

