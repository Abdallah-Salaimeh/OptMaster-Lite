# ---- merged from integration/lifecycle.py ----

"""Connects the agents, execution, verification, scoring and Core Engine node."""
from pathlib import Path
import json
import shutil
import tempfile
import numpy as np

from agents import SolverGenerator
from agents import VerifierGenerator
from agents import Summarizer
from execution import calculate_score
from execution import create_diagnostics
from agents import RepairAgent
from agents import LLMScorer
import os


class NodeLifecycle:
    def __init__(self, llm, problem, artifact_directory: Path, seed=42):
        self.problem = problem.to_dict()
        self.seed = seed
        self.solver_generator = SolverGenerator(llm)
        self.verifier_generator = VerifierGenerator(llm)
        self.summarizer = Summarizer(llm)
        self.llm_scorer = LLMScorer(llm)
        timeout = int(os.getenv("OPTMASTER_TIMEOUT_SECONDS", "30"))
        # Host execution can use the user's full academic CPLEX installation.
        # Docker remains selectable, but pip runtimes may be size-limited.
        # Generated solver/verifier code runs in the isolated Docker backend by default.
        executor_kind = os.getenv("OPTMASTER_EXECUTOR", "docker").lower()
        if executor_kind == "docker":
            from execution import DockerExecutor
            self.executor = DockerExecutor(timeout_seconds=timeout)
        elif executor_kind == "local":
            from execution import LocalExecutor
            self.executor = LocalExecutor(timeout_seconds=timeout)
        else:
            raise ValueError("OPTMASTER_EXECUTOR must be 'docker' or 'local'")
        self.executor_kind = executor_kind
        self.repair_agent = RepairAgent(llm, max_retries=int(os.getenv("OPTMASTER_MAX_REPAIRS", "2")))
        self.artifact_directory = Path(artifact_directory)
        self.artifact_directory.mkdir(parents=True, exist_ok=True)

    def evaluate(self, node, trajectory, parent_contexts=None,
                 expansion_strategy="refine", repair_mode=False, seed=None):
        node.status = "running"
        run_seed = node.seed if repair_mode else (self.seed if seed is None else seed)
        if repair_mode:
            trajectory = node.trajectory or trajectory
            plan = node.plan
            solver_code = node.code
            verifier_code = node.verifier_code
            if not solver_code or not verifier_code:
                raise ValueError("Cannot repair this node: solver or verifier code is missing")
            _, solver_code = self.repair_agent.repair(
                self.problem, trajectory, "\n".join(parent_contexts or []),
                solver_code, node.verification or {}, seed=run_seed)
            expansion_strategy = node.expansion_strategy or expansion_strategy
        else:
            plan, solver_code = self.solver_generator.generate(
                self.problem, trajectory, parent_contexts or [], expansion_strategy, run_seed)
            verifier_code = self.verifier_generator.generate(self.problem)
        node.trajectory = trajectory
        node.expansion_strategy = expansion_strategy
        node.verifier_code = verifier_code
        node.plan = plan
        node.seed = run_seed

        with tempfile.TemporaryDirectory(prefix="optmaster_node_") as temp:
            workdir = Path(temp)
            run_result = self.executor.run_script(solver_code, workdir)
            for _ in range(self.repair_agent.max_retries):
                if run_result["success"]:
                    break
                try:
                    _, repaired_code = self.repair_agent.repair(
                        self.problem, trajectory, "\n".join(parent_contexts or []),
                        solver_code, run_result.get("stderr", "execution failed"), seed=run_seed)
                except (RuntimeError, ValueError):
                    break
                solver_code = repaired_code
                run_result = self.executor.run_script(solver_code, workdir)
            if not run_result["success"]:
                node.status = "failed"
                node.verification = {"feasible": False, "violations": [run_result["stderr"] or "solver failed"], "objectives": {}, "claims_match": False}
                node.reward = -1.0
                node.plan = plan
                node.code = solver_code
                node.reward_source = "invalid"
                node.verification["execution_diagnostics"] = create_diagnostics(
                    {"success": False, "stderr": run_result.get("stderr", "")}, node.verification)
                summary = "Solver execution failed after repair attempts; inspect diagnostics before trying another approach."
                node.summary = summary
                return self._metadata(node, trajectory, plan, node.verification, summary,
                                      parent_contexts, expansion_strategy)

            checked = self.executor.run_script(verifier_code, workdir)
            try:
                report = json.loads(checked["stdout"].strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError):
                report = {"feasible": False, "claims_match": False, "objectives": {}, "violations": [checked.get("stderr") or "Verifier returned invalid JSON"], "diagnostics": {}}

            # Verification failures enter repair mode and are independently checked again.
            for _ in range(self.repair_agent.max_retries):
                if report.get("feasible"):
                    break
                try:
                    _, solver_code = self.repair_agent.repair(
                        self.problem, trajectory, "\n".join(parent_contexts or []),
                        solver_code, report, seed=run_seed)
                    run_result = self.executor.run_script(solver_code, workdir)
                    if not run_result["success"]:
                        continue
                    checked = self.executor.run_script(verifier_code, workdir)
                    report = json.loads(checked["stdout"].strip().splitlines()[-1])
                except (RuntimeError, ValueError, IndexError, json.JSONDecodeError):
                    break

            artifact_target = self.artifact_directory / f"{node.node_id}.npz"
            if (workdir / "solution.npz").is_file():
                shutil.copy2(workdir / "solution.npz", artifact_target)

        node.plan = plan
        node.code = solver_code
        node.artifact = str(artifact_target) if artifact_target.exists() else None
        node.verification = report
        try:
            node.reward = calculate_score(report, objective=self.problem_objective)
            node.reward_source = "verified_objective"
        except (ValueError, KeyError):
            if report.get("feasible"):
                try:
                    node.reward, report["score_rationale"] = self.llm_scorer.score(self.problem, report)
                    node.reward_source = "llm_normalized"
                except (ValueError, KeyError, TypeError, RuntimeError):
                    node.reward = -1.0
                    node.reward_source = "invalid"
            else:
                node.reward = -1.0
                node.reward_source = "invalid"
        node.status = "success" if report.get("feasible") else "failed"
        report["execution_diagnostics"] = create_diagnostics(
            {"success": run_result["success"], "stderr": run_result.get("stderr", "")}, report)
        summary = self.summarizer.summarize(self.problem, trajectory, plan, solver_code, report)
        node.summary = summary
        return self._metadata(node, trajectory, plan, report, summary,
                              parent_contexts, expansion_strategy)

    @staticmethod
    def _metadata(node, trajectory, plan, report, summary, parent_contexts, expansion_strategy):
        return {"trajectory": trajectory, "plan": plan, "report": report,
                "summary": summary, "parent_contexts": list(parent_contexts or []),
                "expansion_strategy": expansion_strategy}

    @property
    def problem_objective(self):
        if self.problem.get("name") == "user_defined":
            return self.problem.get("reward_objective")
        from problems import PROBLEMS
        for cls in PROBLEMS.values():
            if cls.name == self.problem.get("name"):
                return cls().reward_objective
        return None

    def evaluate_many(self, items, max_workers=None):
        """Evaluate independent node candidates concurrently; each item has node/plan/context/mode."""
        from concurrent.futures import ThreadPoolExecutor
        workers = max_workers or int(os.getenv("OPTMASTER_PARALLELISM", "1"))
        def run(item):
            node, trajectory, contexts, strategy, seed = (*item, "refine", None)[:5]
            return self.evaluate(node, trajectory, contexts, strategy, seed=seed)
        if workers <= 1:
            return [run(item) for item in items]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run, item) for item in items]
            return [future.result() for future in futures]


# ---- merged from main.py ----

"""Run the Gemini-backed OptMaster exploration."""
import argparse
import json
import random
from pathlib import Path

from agents import make_llm
from agents import Planner
from engine import Node
from engine import SearchEngine

from problems import get_problem
from problems import MaxCutProblem, CirclePackingProblem, UserDefinedProblem
from problems import load_gset_instance
from agents import build_parent_contexts


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Gemini-powered OptMaster search")
    parser.add_argument("--problem", choices=("max_cut", "gset70", "circle_packing", "motorcycle_production", "custom"), default="max_cut")
    parser.add_argument("--gset-file", type=Path, help="Local unit-weight Gset edge file; required for --problem gset70")
    parser.add_argument("--problem-file", type=Path, help="JSON task spec with description, artifact_spec, and optional reward_objective; required for --problem custom")
    parser.add_argument("--circles", type=int, choices=(26, 27, 32), default=26)
    parser.add_argument("--max-steps", type=int, default=1,
                        help="Expansion rounds (paper experiments use up to 30)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for search and solver prompts")
    parser.add_argument("--parallelism", type=int, choices=(1, 2, 3), default=1,
                        help="Independent candidates per round (paper setup uses 3)")
    args = parser.parse_args(argv)
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    if args.max_steps > 30:
        parser.error("--max-steps must not exceed the paper's 30-step budget")
    if args.problem == "gset70" and args.gset_file is None:
        parser.error("--gset-file is required for --problem gset70")
    if args.problem == "custom" and args.problem_file is None:
        parser.error("--problem-file is required for --problem custom")
    return args


def make_problem(args):
    if args.problem == "gset70":
        return MaxCutProblem(load_gset_instance(args.gset_file))
    if args.problem == "circle_packing":
        return CirclePackingProblem({"n": args.circles, "width": 1.0, "height": 1.0})
    if args.problem == "custom":
        spec = json.loads(args.problem_file.read_text(encoding="utf-8"))
        if not isinstance(spec.get("description"), str) or not spec["description"].strip():
            raise ValueError("Custom task JSON needs a non-empty 'description'")
        if not isinstance(spec.get("artifact_spec"), str) or not spec["artifact_spec"].strip():
            raise ValueError("Custom task JSON needs a non-empty 'artifact_spec'")
        return UserDefinedProblem(spec)
    return get_problem(args.problem)


def _parent_contexts(engine, child, metadata):
    records = []
    for parent in engine.dag.get_parents(child.node_id):
        meta = metadata[parent.node_id]
        records.append({"id": parent.node_id, "trajectory": meta["trajectory"],
                        "report": meta["report"], "summary": meta["summary"]})
    return build_parent_contexts(records)


def main(argv=None):
    args = parse_args(argv)
    print("=" * 68)
    print("OptMaster -- Gemini optimization run")
    print("=" * 68)
    try:
        llm = make_llm()
    except ValueError as exc:
        raise SystemExit(
            f"{exc}\nCopy .env.example to .env, paste your key after GEMINI_API_KEY=, and run again."
        ) from None
    print("Using the Gemini API.\n")
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"Search seed: {args.seed}")

    problem = make_problem(args)
    problem_data = problem.to_dict()
    trajectories = Planner(llm).plan(problem_data["description"], n=3)
    print(f"Planner produced {len(trajectories)} exploration trajectories.")
    print(f"Search budget: {args.max_steps} rounds, {args.parallelism} candidate(s) per round.")

    output_dir = Path(__file__).parent / "artifacts"
    lifecycle = NodeLifecycle(llm, problem, output_dir, seed=args.seed)
    if lifecycle.executor_kind == "local":
        print("WARNING: local mode runs Gemini-generated code directly on this PC; use Docker mode for isolation.")
    else:
        print("Generated code will run in the Docker container.\n")
    engine = SearchEngine(patience_window=10)
    node_metadata = {}
    repair_attempted = set()

    root = Node(step=0)
    engine.add_initial_node(root)
    root_meta = lifecycle.evaluate(root, trajectories[0])
    node_metadata[root.node_id] = root_meta
    engine.record_evaluation(root)
    if root.status != "failed":
        improved = engine.update_search(root)
    else:
        improved = False
    print_node_result(1, root, improved)

    for step in range(1, args.max_steps + 1):
        # Failed attempts get a same-strategy repair pass before new exploration.
        for buggy in engine.pop_buggy_nodes():
            if buggy.node_id in repair_attempted:
                continue
            repair_attempted.add(buggy.node_id)
            old = node_metadata[buggy.node_id]
            print(f"\nRepair mode: retrying failed node {buggy.node_id[:8]} before expansion.")
            repaired = lifecycle.evaluate(
                buggy, old["trajectory"], old["parent_contexts"],
                old["expansion_strategy"], repair_mode=True)
            node_metadata[buggy.node_id] = repaired
            engine.record_evaluation(buggy)
            improved = engine.update_search(buggy) if buggy.status != "failed" else False
            print_node_result(buggy.step, buggy, improved)

        if not any(node.reward is not None and node.status != "failed"
                   for node in engine.dag.get_all_nodes()):
            print("No feasible parent remains after repair; stopping this search run.")
            break

        batch = []
        for worker in range(args.parallelism):
            child, expansion = engine.create_next_node()
            child.step = step
            contexts = _parent_contexts(engine, child, node_metadata)
            trajectory = trajectories[(step + worker) % len(trajectories)]
            candidate_seed = args.seed + (step - 1) * args.parallelism + worker + 1
            batch.append((child, trajectory, contexts, expansion["strategy"], candidate_seed))

        results = lifecycle.evaluate_many(batch, max_workers=args.parallelism)
        stop_early = False
        for (child, trajectory, contexts, strategy, candidate_seed), meta in zip(batch, results):
            node_metadata[child.node_id] = meta
            engine.record_evaluation(child)
            improved = engine.update_search(child) if child.status != "failed" else False
            print(f"\nExpansion decision: {strategy} ({len(child.parent_ids)} parent(s))")
            print_node_result(child.step, child, improved)
            if child.reward_source == "llm_normalized" and child.reward >= 0.94:
                print("Normalized evaluator reached the 0.94 early-stop threshold.")
                stop_early = True
        if stop_early:
            break

    best = engine.get_best_node()
    print("\n" + "=" * 68)
    print(f"DAG nodes: {len(engine.dag.get_all_nodes())}")
    print(f"Best verified reward: {best.reward if best else 'N/A'}")
    print(f"Best node status: {best.status if best else 'N/A'}")
    print(f"Artifacts saved in: {output_dir}")
    print("\nOPTMASTER RUN COMPLETE -- Gemini-backed exploration finished.")


def print_node_result(number, node, improved):
    report = node.verification or {}
    print(f"\nNode {number}: {node.node_id[:8]}")
    print(f"  Status: {node.status}")
    print(f"  Reward: {node.reward}")
    print(f"  Reward source: {node.reward_source}")
    print(f"  Feasible: {report.get('feasible')}")
    print(f"  Claims match: {report.get('claims_match')}")
    print(f"  Objectives: {report.get('objectives', {})}")
    print(f"  Global improvement: {improved}")
    if report.get("violations"):
        print(f"  Violations: {report['violations']}")


if __name__ == "__main__":
    main()


