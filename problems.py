

# ---- merged from problems\definitions.py ----

from abc import ABC, abstractmethod
from itertools import product

class BaseProblem(ABC):

    name = "base"

    def __init__(self, data=None):
        self.data = data or self.default_data()

    @abstractmethod
    def default_data(self):
        ...

    @property
    @abstractmethod
    def description(self):
        ...

    @property
    @abstractmethod
    def artifact_spec(self):
        ...

    def reference_optimum(self):
        return None

    @property
    def reward_objective(self):
        return "objective"

    def to_dict(self):
        return {"name": self.name, "description": self.description, "artifact_spec": self.artifact_spec}


class UserDefinedProblem(BaseProblem):
    """Natural-language formulation input supplied as a small JSON task file."""
    name = "user_defined"

    def default_data(self):
        return {"description": "", "artifact_spec": ""}

    @property
    def description(self):
        return self.data["description"]

    @property
    def artifact_spec(self):
        return self.data["artifact_spec"]

    @property
    def reward_objective(self):
        return self.data.get("reward_objective")

    def to_dict(self):
        return {"name": self.name, "description": self.description,
                "artifact_spec": self.artifact_spec,
                "reward_objective": self.reward_objective}


class MotorcycleProblem(BaseProblem):
    name = "motorcycle_production"

    @property
    def reward_objective(self):
        return "profit"

    def default_data(self):
        return {"profit_a": 100, "hours_a": 2, "profit_b": 400, "hours_b": 5,
                "max_b": 10, "total_hours": 100, "min_a": 5}

    @property
    def description(self):
        d = self.data
        return f"""A manufacturer produces two motorcycle models in one planning period.
- Type A: built in-house, profit {d['profit_a']} per unit, needs {d['hours_a']} assembly hours per unit.
- Type B: assembled from imports, profit {d['profit_b']} per unit, needs {d['hours_b']} assembly hours per unit, and at most {d['max_b']} units can be imported.
- Total assembly hours available: {d['total_hours']}.
- Idle time = unused assembly hours = {d['total_hours']} - ({d['hours_a']}*A + {d['hours_b']}*B).
- At least {d['min_a']} units of Type A must be produced.
- Production quantities must be non-negative integers.
Goal: maximize profit and minimize idle time."""

    @property
    def artifact_spec(self):
        return """The solver must save a file named 'solution.npz' in the current working directory using numpy.savez with exactly these keys:
- 'production': integer array [units_A, units_B]
- 'claimed_profit': float, total profit the solver believes it achieved
- 'claimed_idle_time': float, idle assembly hours the solver believes it has"""

    def reference_optimum(self):
        d = self.data
        best = None
        for a in range(d["min_a"], d["total_hours"] // d["hours_a"] + 1):
            for b in range(0, d["max_b"] + 1):
                used = d["hours_a"] * a + d["hours_b"] * b
                if used > d["total_hours"]:
                    continue
                key = (d["profit_a"] * a + d["profit_b"] * b, -(d["total_hours"] - used))
                if best is None or key > best[0]:
                    best = (key, (a, b))
        return {"production": best[1], "profit": best[0][0], "idle_time": -best[0][1]}


class MaxCutProblem(BaseProblem):
    name = "max_cut"

    @property
    def reward_objective(self):
        return "cut_value"

    def default_data(self):
        return {"n": 6, "edges": [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 5)]}

    @property
    def description(self):
        d = self.data
        edge_text = ", ".join(f"({u},{v})" for u, v in d["edges"])
        return f"""MAX-CUT on an undirected graph with {d['n']} nodes labeled 0 to {d['n'] - 1}.
Edges (each with weight 1): {edge_text}.
Assign every node to one of two sets (label 0 or label 1).
The cut value is the number of edges whose two endpoints have different labels.
Every node must receive exactly one label.
Goal: maximize the cut value."""

    @property
    def artifact_spec(self):
        return f"""The solver must save a file named 'solution.npz' in the current working directory using numpy.savez with exactly these keys:
- 'partition': integer array of length {self.data['n']}, entry i is the label (0 or 1) of node i
- 'claimed_cut_value': float, the cut value the solver believes it achieved"""

    def reference_optimum(self):
        d = self.data
        best = (-1, None)
        for labels in product([0, 1], repeat=d["n"]):
            cut = sum(1 for u, v in d["edges"] if labels[u] != labels[v])
            if cut > best[0]:
                best = (cut, labels)
        return {"partition": best[1], "cut_value": best[0]}


class CirclePackingProblem(BaseProblem):
    """Pack n circles of independently chosen radii in the unit square."""
    name = "circle_packing"

    @property
    def reward_objective(self):
        return "sum_radii"

    def default_data(self):
        return {"n": 26, "width": 1.0, "height": 1.0}

    @property
    def description(self):
        d = self.data
        return (f"Pack {d['n']} non-overlapping circles inside the unit square "
                f"of width {d['width']} and height {d['height']}. Choose each center "
                "(x_i,y_i) and its own radius r_i >= 0. Maximize the sum of all radii. "
                "Containment constraints require r_i <= x_i <= width-r_i and "
                "r_i <= y_i <= height-r_i. For each distinct pair i,j, enforce "
                "(x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2.")

    @property
    def artifact_spec(self):
        return (f"Save solution.npz with centers (shape ({self.data['n']}, 2)), radii "
                f"(length {self.data['n']}), and claimed_sum_radii (float). Centers "
                "are measured from the lower-left corner.")

    def verify(self, artifact, tolerance=1e-6):
        import numpy as np
        violations, claim_errors = [], []
        if "centers" not in artifact or "radii" not in artifact or "claimed_sum_radii" not in artifact:
            return {"feasible": False, "objectives": {}, "claims_match": False,
                    "violations": ["centers, radii, or claimed_sum_radii missing"], "diagnostics": {"claim_errors": []}}
        centers = np.asarray(artifact["centers"], dtype=float)
        radii = np.asarray(artifact["radii"], dtype=float)
        claimed = float(artifact["claimed_sum_radii"])
        n, w, h = self.data["n"], self.data["width"], self.data["height"]
        if centers.shape != (n, 2) or radii.shape != (n,):
            return {"feasible": False, "objectives": {}, "claims_match": False,
                    "violations": [f"centers/radii must have shapes ({n}, 2) and ({n},)"], "diagnostics": {"claim_errors": []}}
        if not np.isfinite(centers).all() or not np.isfinite(radii).all() or not np.isfinite(claimed):
            return {"feasible": False, "objectives": {}, "claims_match": False,
                    "violations": ["all decisions and claims must be finite"], "diagnostics": {"claim_errors": []}}
        total = float(np.sum(radii))
        objectives = {"sum_radii": total}
        if np.any(radii < -tolerance): violations.append("radii must be nonnegative")
        if np.any(centers[:, 0] - radii < -tolerance) or np.any(centers[:, 0] + radii > w + tolerance):
            violations.append("circle extends beyond rectangle width")
        if np.any(centers[:, 1] - radii < -tolerance) or np.any(centers[:, 1] + radii > h + tolerance):
            violations.append("circle extends beyond rectangle height")
        for i in range(n):
            for j in range(i + 1, n):
                if float(np.dot(centers[i] - centers[j], centers[i] - centers[j])) + tolerance < (radii[i] + radii[j]) ** 2:
                    violations.append(f"circles {i} and {j} overlap")
        if abs(claimed - total) > tolerance:
            claim_errors.append(f"claimed_sum_radii={claimed} but recomputed sum is {total}")
        return {"feasible": not violations, "objectives": objectives,
                "claims_match": not claim_errors, "violations": violations,
                "diagnostics": {"claim_errors": claim_errors}}


PROBLEMS = {"motorcycle_production": MotorcycleProblem, "max_cut": MaxCutProblem,
            "circle_packing": CirclePackingProblem, "user_defined": UserDefinedProblem}


def get_problem(name):
    return PROBLEMS[name]()


# ---- merged from problems\gset.py ----

"""Reader for the unweighted Gset DIMACS/plain edge-list benchmark files."""
from pathlib import Path


def load_gset_instance(path):
    """Read a Gset graph file and return zero-based nodes and unit-weight edges.

    Supports a `p edge N M` header, an `N M` header, and `e U V` or `U V [1]`
    records. The OptMaster paper's Gset70 task uses unit weights.
    """
    path = Path(path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith(("#", "%", "c "))]
    if not lines:
        raise ValueError(f"Gset file is empty: {path}")
    first = lines.pop(0).split()
    if first[0].lower() == "p":
        if len(first) < 4 or first[1].lower() != "edge":
            raise ValueError("Expected DIMACS header: p edge <nodes> <edges>")
        node_count, expected_edges = int(first[2]), int(first[3])
    elif len(first) >= 2 and all(token.lstrip("+-").isdigit() for token in first[:2]):
        node_count, expected_edges = int(first[0]), int(first[1])
    else:
        raise ValueError("Expected an edge-count header (N M or p edge N M)")

    raw_edges = []
    for line_no, line in enumerate(lines, start=2):
        fields = line.split()
        if fields[0].lower() in {"e", "a"}:
            fields = fields[1:]
        if len(fields) not in (2, 3):
            raise ValueError(f"Invalid edge record on line {line_no}: {line}")
        try:
            u, v = int(fields[0]), int(fields[1])
            weight = float(fields[2]) if len(fields) == 3 else 1.0
        except ValueError as exc:
            raise ValueError(f"Invalid numeric edge record on line {line_no}: {line}") from exc
        if weight != 1.0:
            raise ValueError(f"Expected unit edge weights in Gset instance, found {weight} on line {line_no}")
        raw_edges.append((u, v))
    if len(raw_edges) != expected_edges:
        raise ValueError(f"Header says {expected_edges} edges but file contains {len(raw_edges)}")
    if not raw_edges:
        raise ValueError("Gset graph contains no edges")
    min_id = min(min(u, v) for u, v in raw_edges)
    max_id = max(max(u, v) for u, v in raw_edges)
    offset = 1 if min_id >= 1 and max_id <= node_count else 0
    edges = [(u - offset, v - offset) for u, v in raw_edges]
    if min(min(u, v) for u, v in edges) < 0 or max(max(u, v) for u, v in edges) >= node_count:
        raise ValueError("Edge endpoints fall outside the node range described by the header")
    return {"n": node_count, "edges": edges, "source": str(path)}

