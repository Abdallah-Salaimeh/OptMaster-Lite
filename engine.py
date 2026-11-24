

# ---- merged from core\node.py ----


from dataclasses import dataclass, field
from uuid import uuid4
from typing import Literal


@dataclass
class Node:
    node_id: str = field(default_factory=lambda: str(uuid4()))
    plan: str = ""
    code: str | None = None
    artifact: str | None = None
    verification: dict | None = None
    summary: str | None = None
    trajectory: dict | None = None
    expansion_strategy: str | None = None
    verifier_code: str | None = None
    reward: float | None = None
    reward_source: str | None = None
    seed: int = 42
    parent_ids: list[str] = field(default_factory=list)

    status: Literal["pending", "running", "success", "failed"] = "pending"
    step: int = 0

    def get_transferable_context(self) -> dict:
        return {
            "plan": self.plan,
            "verification": self.verification,
            "summary": self.summary,
        }


# ---- merged from core\dag.py ----

import networkx as nx

from engine import Node


class ExplorationDAG:
    def __init__(self):
        self.graph = nx.DiGraph()


    def add_node(self, node: Node):
        if node.node_id in self.graph:
            raise ValueError(f"Node {node.node_id} already exists.")
        self.graph.add_node(node.node_id, data=node)


    def add_edge(self, parent_id: str, child_id: str):
        if parent_id not in self.graph:
            raise ValueError(f"Parent {parent_id} does not exist.")
        if child_id not in self.graph:
            raise ValueError(f"Child {child_id} does not exist.")
        if parent_id == child_id:
            raise ValueError("A node cannot be its own parent.")
        if nx.has_path(self.graph, child_id, parent_id):
            raise ValueError("Adding this edge might create a cycle.")
        if self.graph.has_edge(parent_id, child_id):
            return 
        
        self.graph.add_edge(parent_id, child_id)
        child = self.graph.nodes[child_id]["data"]
        child.parent_ids.append(parent_id)


    def get_node(self, node_id: str) -> Node:
        if node_id not in self.graph:
            raise ValueError(f"Node {node_id} does not exist.")
        return self.graph.nodes[node_id]["data"]


    def get_parents(self, node_id: str) -> list[Node]:
        if node_id not in self.graph:
            raise ValueError(f"Node {node_id} does not exist.")
        parent_ids = self.graph.predecessors(node_id)
        return [self.get_node(parent_id) for parent_id in parent_ids]


    def get_children(self, node_id: str) -> list[Node]:
        if node_id not in self.graph:
            raise ValueError(f"Node {node_id} does not exist.")
        child_ids = self.graph.successors(node_id)
        return [self.get_node(child_id) for child_id in child_ids]

    
    def get_all_nodes(self) -> list[Node]:
        return [self.graph.nodes[node_id]["data"] for node_id in self.graph.nodes]


    def get_best_node(self) -> Node | None:
        nodes = self.get_all_nodes()
        valid_nodes = [node for node in nodes if node.reward is not None and node.status != "failed"]

        if not valid_nodes:
            return None
        return max(valid_nodes, key=lambda node: node.reward)

    def get_root_nodes(self) -> list[Node]:
        return [self.get_node(node_id) for node_id in self.graph.nodes if self.graph.in_degree(node_id) == 0
    ]


    def get_leaf_nodes(self) -> list[Node]:
        return [self.get_node(node_id) for node_id in self.graph.nodes if self.graph.out_degree(node_id) == 0
    ]


# ---- merged from core\expansion.py ----


import numpy as np


class ExpansionController:

    def select_parent_count(self, stagnation: float) -> int:
        if not 0 <= stagnation < 1:
            raise ValueError("Stagnation must be between zero and one.")

        probabilities = [
            1 - stagnation,
            stagnation * (1 - stagnation),
            stagnation ** 2
        ]

        parent_counts = [1, 2, 3]

        selected_count = np.random.choice(
            parent_counts,
            p=probabilities
        )

        return int(selected_count)

    def get_strategy(self, parent_count: int) -> str:
        strategies = {
            1: "refine",
            2: "cross",
            3: "fuse"
        }

        if parent_count not in strategies:
            raise ValueError("Parent count must be 1, 2, or 3.")

        return strategies[parent_count]

    def select_expansion(self, stagnation: float) -> dict:
        parent_count = self.select_parent_count(stagnation)

        strategy = self.get_strategy(parent_count)

        return {
            "parent_count": parent_count,
            "strategy": strategy
        }


# ---- merged from core\stagnation.py ----


class StagnationTracker:

    def __init__(self, patience_window: int = 10, epsilon: float = 0.01):
        if patience_window <= 0:
            raise ValueError("Patience window must be greater than zero.")

        if not 0 < epsilon < 1:
            raise ValueError("Epsilon must be between zero and one.")

        self.patience_window = patience_window
        self.epsilon = epsilon

        self.best_reward = None
        self.steps_without_improvement = 0

    def update(self, reward: float) -> bool:
        if self.best_reward is None or reward > self.best_reward:
            self.best_reward = reward
            self.steps_without_improvement = 0
            return True

        self.steps_without_improvement += 1
        return False

    def get_stagnation(self) -> float:
        stagnation = self.steps_without_improvement / self.patience_window

        return min(stagnation, 1 - self.epsilon)


# ---- merged from core\parent_selector.py ----


import numpy as np

from engine import Node


class ParentSelector:

    def __init__(self, beta: float = 1.0):
        if beta <= 0:
            raise ValueError("Beta must be greater than zero.")
        self.beta = beta


    def select_first_parent(self, nodes: list[Node]) -> Node:
        candidates = [node for node in nodes if node.reward is not None]

        if not candidates:
            raise ValueError("No nodes with valid rewards are available.")

        rewards = np.array([node.reward for node in candidates], dtype=float)
        scaled_rewards = rewards / self.beta
        probabilities = np.exp(scaled_rewards - np.max(scaled_rewards))
        probabilities = probabilities / np.sum(probabilities)

        selected_index = np.random.choice(len(candidates), p=probabilities)
        return candidates[selected_index]

    def select_additional_parent(
        self,
        nodes: list[Node],
        selected_parents: list[Node],
        stagnation: float,
        distance_fn
    ) -> Node:

        candidates = [
            node for node in nodes
            if node.reward is not None
            and node.node_id not in {
                parent.node_id for parent in selected_parents
            }
        ]

        if not candidates:
            raise ValueError("No additional parent candidates available.")

        if not selected_parents:
            raise ValueError("At least one parent must already be selected.")

        if not 0 <= stagnation < 1:
            raise ValueError("Stagnation must be between zero and one.")

        def score(candidate):
            novelty = min(
                distance_fn(candidate.plan, parent.plan)
                for parent in selected_parents
            )

            return candidate.reward + stagnation * novelty

        return max(candidates, key=score)


# ---- merged from core\semantic_novelty.py ----

"""SentenceTransformer cosine novelty used for selecting diverse DAG parents."""
import os
import numpy as np


class SemanticNovelty:
    def __init__(self, model_name=None):
        self.model_name = model_name or os.getenv(
            "OPTMASTER_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def similarity(self, left, right):
        vectors = np.asarray(self._load().encode([left, right], normalize_embeddings=True))
        return float(np.dot(vectors[0], vectors[1]))

    def distance(self, left, right):
        return 1.0 - self.similarity(left, right)


# ---- merged from core\search_engine.py ----


from collections.abc import Callable

from engine import Node
from engine import ExplorationDAG
from engine import ParentSelector
from engine import StagnationTracker
from engine import ExpansionController


class SearchEngine:

    def __init__(
        self,
        patience_window: int = 10,
        epsilon: float = 0.01,
        beta: float = 1.0,
        distance_fn: Callable[[str, str], float] | None = None
    ):

        self.dag = ExplorationDAG()

        self.stagnation_tracker = StagnationTracker(
            patience_window=patience_window,
            epsilon=epsilon
        )

        self.parent_selector = ParentSelector(beta=beta)
        self.expansion_controller = ExpansionController()

        self.distance_fn = distance_fn
        self._semantic_novelty = None
        self.buggy_nodes: set[str] = set()

    def record_evaluation(self, node: Node) -> None:
        if node.status == "failed":
            self.buggy_nodes.add(node.node_id)
        else:
            self.buggy_nodes.discard(node.node_id)

    def pop_buggy_nodes(self) -> list[Node]:
        nodes = [self.dag.get_node(node_id) for node_id in sorted(self.buggy_nodes)
                 if node_id in self.dag.graph]
        self.buggy_nodes.clear()
        return nodes

    def add_initial_node(self, node: Node):
        self.dag.add_node(node)

    def update_search(self, evaluated_node: Node) -> bool:
        if evaluated_node.node_id not in self.dag.graph:
            raise ValueError("Node must be added to the DAG before updating.")

        if evaluated_node.reward is None:
            raise ValueError("Evaluated node must have a reward.")

        return self.stagnation_tracker.update(evaluated_node.reward)

    def create_next_node(self) -> tuple[Node, dict]:

        nodes = self.dag.get_all_nodes()

        stagnation = self.stagnation_tracker.get_stagnation()

        expansion = self.expansion_controller.select_expansion(
            stagnation
        )

        parent_count = expansion["parent_count"]

        candidates = [
            node for node in nodes
            if node.reward is not None and node.status != "failed"
        ]

        if not candidates:
            raise ValueError("No evaluated nodes are available for expansion.")

        # Early in the search there may be fewer evaluated nodes than the
        # stochastic expansion policy requests. Use the available count.
        parent_count = min(parent_count, len(candidates))
        expansion["parent_count"] = parent_count
        expansion["strategy"] = self.expansion_controller.get_strategy(parent_count)

        selected_parents = [
            self.parent_selector.select_first_parent(candidates)
        ]

        while len(selected_parents) < parent_count:

            distance_fn = self.distance_fn
            if distance_fn is None:
                if self._semantic_novelty is None:
                    try:
                        from engine import SemanticNovelty
                    except ImportError as exc:
                        raise RuntimeError(
                            "Multi-parent search needs the optional SentenceTransformer package. "
                            "Install it with: python -m pip install -r requirements-embeddings.txt"
                        ) from exc
                    self._semantic_novelty = SemanticNovelty()
                distance_fn = self._semantic_novelty.distance

            next_parent = self.parent_selector.select_additional_parent(
                candidates,
                selected_parents,
                stagnation,
                distance_fn
            )

            selected_parents.append(next_parent)

        # add_edge() records parent_ids, so initialize the list only once.
        child = Node()

        self.dag.add_node(child)

        for parent in selected_parents:
            self.dag.add_edge(parent.node_id, child.node_id)

        expansion["parent_ids"] = [
            parent.node_id for parent in selected_parents
        ]

        return child, expansion

    def get_best_node(self) -> Node | None:
        return self.dag.get_best_node()

