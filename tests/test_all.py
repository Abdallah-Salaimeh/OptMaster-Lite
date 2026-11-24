

# ---- merged from tests/test_dag.py ----


import pytest #type: ignore

from engine import Node
from engine import ExplorationDAG


def test_add_node():
    dag = ExplorationDAG()
    node = Node()

    dag.add_node(node)

    assert dag.get_node(node.node_id) is node


def test_add_edge():
    dag = ExplorationDAG()
    parent = Node()
    child = Node()

    dag.add_node(parent)
    dag.add_node(child)
    dag.add_edge(parent.node_id, child.node_id)

    assert parent in dag.get_parents(child.node_id)
    assert child in dag.get_children(parent.node_id)
    assert parent.node_id in child.parent_ids


def test_multiple_parents():
    dag = ExplorationDAG()

    parent1 = Node()
    parent2 = Node()
    child = Node()

    dag.add_node(parent1)
    dag.add_node(parent2)
    dag.add_node(child)

    dag.add_edge(parent1.node_id, child.node_id)
    dag.add_edge(parent2.node_id, child.node_id)

    assert len(dag.get_parents(child.node_id)) == 2


def test_duplicate_node_rejected():
    dag = ExplorationDAG()
    node = Node()

    dag.add_node(node)

    with pytest.raises(ValueError):
        dag.add_node(node)


def test_cycle_rejected():
    dag = ExplorationDAG()

    a = Node()
    b = Node()
    c = Node()

    for node in [a, b, c]:
        dag.add_node(node)

    dag.add_edge(a.node_id, b.node_id)
    dag.add_edge(b.node_id, c.node_id)

    with pytest.raises(ValueError):
        dag.add_edge(c.node_id, a.node_id)


def test_duplicate_edge_does_not_duplicate_parent():
    dag = ExplorationDAG()

    parent = Node()
    child = Node()

    dag.add_node(parent)
    dag.add_node(child)

    dag.add_edge(parent.node_id, child.node_id)
    dag.add_edge(parent.node_id, child.node_id)

    assert child.parent_ids.count(parent.node_id) == 1


def test_best_node():
    dag = ExplorationDAG()

    node1 = Node(reward=0.5)
    node2 = Node(reward=0.9)
    node3 = Node(reward=0.7)

    for node in [node1, node2, node3]:
        dag.add_node(node)

    assert dag.get_best_node() is node2


def test_best_node_when_no_rewards():
    dag = ExplorationDAG()
    dag.add_node(Node())

    assert dag.get_best_node() is None


def test_root_and_leaf_nodes():
    dag = ExplorationDAG()

    a = Node()
    b = Node()
    c = Node()

    for node in [a, b, c]:
        dag.add_node(node)

    dag.add_edge(a.node_id, b.node_id)
    dag.add_edge(b.node_id, c.node_id)

    assert dag.get_root_nodes() == [a]
    assert dag.get_leaf_nodes() == [c]



# ---- merged from tests/test_expansion.py ----


import pytest #type: ignore

from engine import ExpansionController


def test_zero_stagnation_always_selects_one_parent():
    controller = ExpansionController()

    for _ in range(20):
        assert controller.select_parent_count(0.0) == 1


def test_selected_parent_count_is_valid():
    controller = ExpansionController()

    for _ in range(100):
        result = controller.select_parent_count(0.5)
        assert result in [1, 2, 3]


def test_strategy_mapping():
    controller = ExpansionController()

    assert controller.get_strategy(1) == "refine"
    assert controller.get_strategy(2) == "cross"
    assert controller.get_strategy(3) == "fuse"


def test_invalid_parent_count():
    controller = ExpansionController()

    with pytest.raises(ValueError):
        controller.get_strategy(4)


def test_invalid_stagnation():
    controller = ExpansionController()

    with pytest.raises(ValueError):
        controller.select_parent_count(1.0)


def test_select_expansion_returns_expected_structure():
    controller = ExpansionController()

    result = controller.select_expansion(0.0)

    assert result == {
        "parent_count": 1,
        "strategy": "refine"
    }


# ---- merged from tests/test_gemini_client.py ----

import pytest

from agents import GeminiClient


def test_gemini_client_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiClient()


def test_gemini_request_uses_auth_header_not_url(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-only")
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "generated"}]}}]}

    calls = {}

    def fake_post(*args, **kwargs):
        calls["args"], calls["kwargs"] = args, kwargs
        return Response()

    post = fake_post
    monkeypatch.setattr("agents.requests.post", post)

    result = GeminiClient().generate("prompt", system="rules", json_mode=True)

    assert result == "generated"
    args, kwargs = calls["args"], calls["kwargs"]
    assert "test-key-only" not in args[0]
    assert kwargs["headers"]["x-goog-api-key"] == "test-key-only"
    assert kwargs["json"]["systemInstruction"]["parts"][0]["text"] == "rules"
    assert kwargs["json"]["generationConfig"]["responseMimeType"] == "application/json"


def test_gemini_38_flash_omits_unsupported_temperature(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-only")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "generated"}]}}]}

    calls = {}

    def fake_post(*args, **kwargs):
        calls["kwargs"] = kwargs
        return Response()

    monkeypatch.setattr("agents.requests.post", fake_post)
    GeminiClient().generate("prompt", temperature=0.4)

    assert "temperature" not in calls["kwargs"]["json"]["generationConfig"]


# ---- merged from tests/test_node.py ----

from engine import Node


def test_nodes_have_different_ids():
    node1 = Node()
    node2 = Node()

    assert node1.node_id != node2.node_id


def test_nodes_have_independent_parent_lists():
    node1 = Node()
    node2 = Node()

    assert node1.parent_ids is not node2.parent_ids


def test_transferable_context():
    node = Node(
        plan="Test optimization strategy",
        summary="Test summary"
    )

    context = node.get_transferable_context()

    assert context["plan"] == "Test optimization strategy"
    assert context["summary"] == "Test summary"

# ---- merged from tests/test_parent_selector.py ----

import pytest #type: ignore

from engine import Node
from engine import ParentSelector


def test_selector_returns_a_valid_parent():
    selector = ParentSelector()

    nodes = [
        Node(reward=0.2),
        Node(reward=0.8),
        Node(reward=0.5),
    ]

    selected = selector.select_first_parent(nodes)

    assert selected in nodes


def test_selector_ignores_nodes_without_rewards():
    selector = ParentSelector()

    valid_node = Node(reward=0.8)
    nodes = [Node(), valid_node]

    selected = selector.select_first_parent(nodes)

    assert selected is valid_node


def test_selector_rejects_empty_candidates():
    selector = ParentSelector()

    with pytest.raises(ValueError):
        selector.select_first_parent([])


def test_beta_must_be_positive():
    with pytest.raises(ValueError):
        ParentSelector(beta=0)

# ---- merged from tests/test_problem_benchmarks.py ----

import pytest

from problems import CirclePackingProblem, UserDefinedProblem
from problems import load_gset_instance
from agents import LLMScorer
from engine import SemanticNovelty
from engine import Node
from engine import SearchEngine


def test_gset_reader_accepts_dimacs_unit_weight_graph(tmp_path):
    graph = tmp_path / "Gset70.txt"
    graph.write_text("p edge 4 3\ne 1 2 1\ne 2 3 1\ne 3 4 1\n", encoding="utf-8")
    loaded = load_gset_instance(graph)
    assert loaded["n"] == 4
    assert loaded["edges"] == [(0, 1), (1, 2), (2, 3)]


def test_gset_reader_rejects_wrong_edge_count(tmp_path):
    graph = tmp_path / "bad.txt"
    graph.write_text("4 2\n1 2 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contains 1"):
        load_gset_instance(graph)


def test_circle_packing_verifies_independent_radii_and_sum():
    problem = CirclePackingProblem({"n": 2, "width": 1.0, "height": 1.0})
    report = problem.verify({
        "centers": [[0.25, 0.5], [0.75, 0.5]],
        "radii": [0.25, 0.25],
        "claimed_sum_radii": 0.5,
    })
    assert report["feasible"] is True
    assert report["objectives"] == {"sum_radii": 0.5}
    assert report["claims_match"] is True


def test_circle_packing_rejects_overlap_and_bad_claim():
    problem = CirclePackingProblem({"n": 2, "width": 1.0, "height": 1.0})
    report = problem.verify({
        "centers": [[0.3, 0.5], [0.6, 0.5]],
        "radii": [0.2, 0.2],
        "claimed_sum_radii": 1.0,
    })
    assert report["feasible"] is False
    assert report["claims_match"] is False


def test_user_defined_problem_uses_natural_language_spec():
    problem = UserDefinedProblem({
        "description": "Maximize a nonnegative scalar x.",
        "artifact_spec": "Save x and claimed_x in solution.npz.",
        "reward_objective": "x",
    })
    assert problem.to_dict()["description"].startswith("Maximize")
    assert problem.reward_objective == "x"


def test_llm_scorer_returns_normalized_value_for_feasible_report():
    class GeminiStub:
        def generate(self, *args, **kwargs):
            return '{"score": 0.7, "rationale": "Verified progress."}'

    score, rationale = LLMScorer(GeminiStub()).score(
        {"description": "Maximize x."}, {"feasible": True, "objectives": {}})
    assert score == 0.7
    assert rationale == "Verified progress."


def test_llm_scorer_rejects_score_outside_unit_interval():
    class GeminiStub:
        def generate(self, *args, **kwargs):
            return '{"score": 1.2, "rationale": "bad"}'

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        LLMScorer(GeminiStub()).score(
            {"description": "Maximize x."}, {"feasible": True, "objectives": {}})


def test_semantic_novelty_uses_cosine_distance():
    class EmbeddingModel:
        def encode(self, texts, normalize_embeddings):
            assert normalize_embeddings is True
            return [[1.0, 0.0], [0.0, 1.0]]

    novelty = SemanticNovelty()
    novelty._model = EmbeddingModel()
    assert novelty.distance("plan a", "plan b") == 1.0


def test_failed_nodes_are_queued_and_can_be_reactivated():
    search = SearchEngine()
    node = Node(status="failed", reward=-1.0)
    search.add_initial_node(node)
    search.record_evaluation(node)
    assert search.pop_buggy_nodes() == [node]
    node.status = "success"
    node.reward = 1.0
    search.record_evaluation(node)
    assert search.buggy_nodes == set()


# ---- merged from tests/test_search_engine.py ----


import pytest #type: ignore

from engine import Node
from engine import SearchEngine


def test_initial_node_is_added():
    engine = SearchEngine()

    node = Node(reward=0.5)

    engine.add_initial_node(node)

    assert engine.dag.get_node(node.node_id) is node


def test_update_search_detects_improvement():
    engine = SearchEngine()

    node = Node(reward=0.8)

    engine.add_initial_node(node)

    result = engine.update_search(node)

    assert result is True


def test_update_search_detects_no_improvement():
    engine = SearchEngine()

    node1 = Node(reward=0.8)
    node2 = Node(reward=0.6)

    engine.add_initial_node(node1)
    engine.add_initial_node(node2)

    engine.update_search(node1)

    result = engine.update_search(node2)

    assert result is False


def test_update_search_rejects_unknown_node():
    engine = SearchEngine()

    node = Node(reward=0.5)

    with pytest.raises(ValueError):
        engine.update_search(node)


def test_update_search_rejects_missing_reward():
    engine = SearchEngine()

    node = Node()

    engine.add_initial_node(node)

    with pytest.raises(ValueError):
        engine.update_search(node)


def test_create_next_node_with_one_parent():
    engine = SearchEngine()

    parent = Node(reward=0.8)

    engine.add_initial_node(parent)

    child, expansion = engine.create_next_node()

    assert child in engine.dag.get_all_nodes()
    assert parent.node_id in child.parent_ids
    assert expansion["parent_count"] == 1
    assert expansion["strategy"] == "refine"


def test_child_is_connected_to_parent():
    engine = SearchEngine()

    parent = Node(reward=0.8)

    engine.add_initial_node(parent)

    child, _ = engine.create_next_node()

    assert child in engine.dag.get_children(parent.node_id)


def test_multiple_parent_selection_uses_distance_function():
    engine = SearchEngine(patience_window=1, distance_fn=lambda a, b: 0.5)

    nodes = [
        Node(reward=0.8),
        Node(reward=0.7),
        Node(reward=0.6)
    ]

    for node in nodes:
        engine.add_initial_node(node)

    engine.stagnation_tracker.update(0.8)
    engine.stagnation_tracker.update(0.5)

    child, expansion = engine.create_next_node()
    assert len(child.parent_ids) == expansion["parent_count"]


# ---- merged from tests/test_solver_backend.py ----

import pytest

from agents import validate_cplex_solver_code


def test_accepts_docplex_solver_script():
    validate_cplex_solver_code("""
from docplex.mp.model import Model
m = Model()
x = m.binary_var()
m.maximize(x)
s = m.solve(log_output=False)
""")


def test_rejects_non_cplex_solver_script():
    with pytest.raises(ValueError, match="must import DOcplex"):
        validate_cplex_solver_code("""
import scipy.optimize
result = scipy.optimize.minimize(lambda x: x[0], [0])
""")


def test_rejects_mixing_cplex_and_alternative_solver():
    with pytest.raises(ValueError, match="must not use non-CPLEX"):
        validate_cplex_solver_code("""
import docplex
import pulp
model.solve()
""")


def test_requires_solver_call():
    with pytest.raises(ValueError, match="must call"):
        validate_cplex_solver_code("""
import cplex
model = cplex.Cplex()
""")


# ---- merged from tests/test_stagnation.py ----


import pytest

from engine import StagnationTracker


def test_initial_state():
    tracker = StagnationTracker()

    assert tracker.best_reward is None
    assert tracker.steps_without_improvement == 0
    assert tracker.get_stagnation() == 0.0


def test_first_reward_is_an_improvement():
    tracker = StagnationTracker()

    result = tracker.update(0.5)

    assert result is True
    assert tracker.best_reward == 0.5
    assert tracker.steps_without_improvement == 0


def test_global_improvement_resets_counter():
    tracker = StagnationTracker()

    tracker.update(0.5)
    tracker.update(0.4)
    tracker.update(0.3)

    result = tracker.update(0.8)

    assert result is True
    assert tracker.best_reward == 0.8
    assert tracker.steps_without_improvement == 0


def test_no_improvement_increments_counter():
    tracker = StagnationTracker()

    tracker.update(0.8)
    result = tracker.update(0.6)

    assert result is False
    assert tracker.steps_without_improvement == 1


def test_stagnation_calculation():
    tracker = StagnationTracker(patience_window=10)

    tracker.update(0.8)

    for _ in range(5):
        tracker.update(0.5)

    assert tracker.get_stagnation() == 0.5


def test_stagnation_does_not_exceed_cap():
    tracker = StagnationTracker(patience_window=10, epsilon=0.01)

    tracker.update(0.8)

    for _ in range(20):
        tracker.update(0.5)

    assert tracker.get_stagnation() == pytest.approx(0.99)


def test_invalid_patience_window():
    with pytest.raises(ValueError):
        StagnationTracker(patience_window=0)


def test_invalid_epsilon():
    with pytest.raises(ValueError):
        StagnationTracker(epsilon=1.0)


# Independent checks for the compact execution utilities.
from execution import (
    create_maxcut_problem,
    create_solver_result,
    create_diagnostics,
    execute_solver,
    verify_maxcut_solution,
)


def test_problem_and_result_contracts_are_detached():
    nodes = [0, 1]
    edges = [(0, 1)]
    problem = create_maxcut_problem(nodes, edges)
    nodes.append(2)
    edges.append((1, 2))
    assert problem == {"problem_type": "MAXCUT", "nodes": [0, 1], "edges": [(0, 1)]}

    assignment = {0: 0, 1: 1}
    result = create_solver_result(assignment)
    assignment[1] = 0
    assert result["solution"] == {0: 0, 1: 1}


def test_independent_maxcut_checker_recomputes_cut():
    report = verify_maxcut_solution(
        {"nodes": [0, 1, 2], "edges": [(0, 1), (1, 2), (0, 2)]},
        {"solution": {0: 0, 1: 1, 2: 0}},
    )
    assert report["feasible"] is True
    assert report["cut_value"] == 2
    assert report["cut_edges"] == [(0, 1), (1, 2)]


def test_independent_maxcut_checker_rejects_incomplete_assignment():
    report = verify_maxcut_solution(
        {"nodes": [0, 1], "edges": [(0, 1)]}, {"solution": {0: 0}}
    )
    assert report["feasible"] is False
    assert report["cut_value"] is None
    assert "Node 1 is missing an assignment" in report["violations"]


def test_in_process_executor_records_success_and_errors():
    good = execute_solver(lambda payload: payload["answer"], {"answer": 42})
    bad = execute_solver(lambda _payload: 1 / 0, {})
    assert good["status"] == "success" and good["result"] == 42
    assert good["error"] is None and good["runtime"] >= 0
    assert bad["status"] == "failed" and bad["error_type"] == "ZeroDivisionError"


def test_diagnostics_include_verifier_outcome():
    report = create_diagnostics(
        {"success": True, "runtime": 0.1},
        {"feasible": True, "objectives": {"cut_value": 4}},
    )
    assert report["execution_status"] == "success"
    assert report["verification_status"] == "valid"
    assert report["score"] == 4

