from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import csv
import itertools

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector, partial_trace


BELL_PAULI_MAP: Dict[str, str] = {
    "phi_plus": "I",
    "phi_minus": "Z",
    "psi_plus": "X",
    "psi_minus": "Y",
}

BELL_LABELS = ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]
PAULI_AXES = ("X", "Y", "Z")


@dataclass(frozen=True)
class ErrorModelResult:
    edge_id: Tuple[str, str]
    coeffs: Dict[str, float]
    coeffs_norm: Dict[str, float]
    success_probs: Dict[str, float]
    base_threshold: Optional[float]
    difficulty_rating: Optional[float]


@dataclass(frozen=True)
class DejmpsPlan:
    alignment: Tuple[str, str, str]
    alignment_gates: Tuple[str, ...]
    target_label: str
    pre_rotation_pauli: str
    round_rotations: List[str]
    round_bases: List[str]
    num_bell_pairs: int
    rounds: int
    expected_fidelity: float
    expected_success_probability: float
    objective: str


def bell_tomography_circuit(pauli: str) -> QuantumCircuit:
    """Create a 2-qubit circuit that maps a Bell basis element to Phi+ via a local Pauli."""
    qc = QuantumCircuit(2, 1)  # flag_bit=0; no measurement keeps it 0
    if pauli == "X":
        qc.x(0)
    elif pauli == "Y":
        qc.y(0)
    elif pauli == "Z":
        qc.z(0)
    elif pauli != "I":
        raise ValueError(f"Unknown Pauli: {pauli}")
    return qc


def canonical_edge(edge_id: Sequence[str]) -> Tuple[str, str]:
    a, b = edge_id
    return tuple(sorted([a, b]))


def load_existing_edges(path: Path) -> set[Tuple[str, str]]:
    seen: set[Tuple[str, str]] = set()
    if not path.exists():
        return seen
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = row.get("node_a")
            b = row.get("node_b")
            if a and b:
                seen.add(canonical_edge((a, b)))
    return seen


def append_error_row(path: Path, row: Dict[str, object]) -> None:
    fieldnames = [
        "node_a",
        "node_b",
        "p_phi_plus",
        "p_phi_minus",
        "p_psi_plus",
        "p_psi_minus",
        "p_sum",
        "p_phi_plus_norm",
        "p_phi_minus_norm",
        "p_psi_plus_norm",
        "p_psi_minus_norm",
        "base_threshold",
        "difficulty_rating",
        "success_probability",
        "timestamp_utc",
        "note",
    ]
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _timestamp_key(row: Dict[str, object]) -> str:
    ts = row.get("timestamp_utc")
    return str(ts) if ts is not None else ""


def load_error_model_row(
    edge_id: Sequence[str],
    path: Path | str = "edge_error_models.csv",
    prefer_latest: bool = True,
) -> Optional[Dict[str, object]]:
    edge = canonical_edge(edge_id)
    path = Path(path)
    if not path.exists():
        return None
    best_row = None
    best_ts = ""
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = row.get("node_a")
            b = row.get("node_b")
            if not a or not b:
                continue
            if canonical_edge((a, b)) != edge:
                continue
            if not prefer_latest:
                return row
            ts = _timestamp_key(row)
            if ts >= best_ts:
                best_row = row
                best_ts = ts
    return best_row


def load_error_models(path: Path | str = "edge_error_models.csv") -> Dict[Tuple[str, str], Dict[str, object]]:
    path = Path(path)
    table: Dict[Tuple[str, str], Dict[str, object]] = {}
    if not path.exists():
        return table
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = row.get("node_a")
            b = row.get("node_b")
            if not a or not b:
                continue
            edge = canonical_edge((a, b))
            prev = table.get(edge)
            if prev is None or _timestamp_key(row) >= _timestamp_key(prev):
                table[edge] = row
    return table


def bell_vector_from_row(row: Dict[str, object]) -> Optional[np.ndarray]:
    vals: List[float] = []
    for key in ("p_phi_plus_norm", "p_phi_minus_norm", "p_psi_plus_norm", "p_psi_minus_norm"):
        v = row.get(key)
        if v is None:
            vals = []
            break
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            vals = []
            break
    if not vals:
        vals = []
        for key in ("p_phi_plus", "p_phi_minus", "p_psi_plus", "p_psi_minus"):
            v = row.get(key)
            if v is None:
                return None
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                return None
        s = sum(vals)
        if s > 0:
            vals = [v / s for v in vals]
    s = sum(vals)
    if s <= 0:
        return None
    return np.array(vals, dtype=float)


def pauli_rates_from_bell_vector(p: np.ndarray) -> Dict[str, float]:
    # Bell order: phi+, phi-, psi+, psi-
    return {
        "I": float(p[0]),
        "Z": float(p[1]),
        "X": float(p[2]),
        "Y": float(p[3]),
    }


def bell_vector_from_pauli_rates(pI: float, pX: float, pY: float, pZ: float) -> np.ndarray:
    return np.array([pI, pZ, pX, pY], dtype=float)


def dominant_error_summary(p: np.ndarray) -> Dict[str, float]:
    rates = pauli_rates_from_bell_vector(p)
    bit_flip = rates["X"] + rates["Y"]
    phase_flip = rates["Z"] + rates["Y"]
    return {
        "bit_flip": float(bit_flip),
        "phase_flip": float(phase_flip),
        "y_rate": float(rates["Y"]),
    }


_BELL_PERM = {
    "I": [0, 1, 2, 3],
    "X": [2, 3, 0, 1],
    "Z": [1, 0, 3, 2],
    "Y": [3, 2, 1, 0],
}


def apply_bob_pauli_to_bell_vector(p: np.ndarray, rotation_pauli: str) -> np.ndarray:
    perm = _BELL_PERM.get(rotation_pauli, _BELL_PERM["I"])
    out = np.zeros_like(p)
    for in_idx, out_idx in enumerate(perm):
        out[out_idx] = p[in_idx]
    return out


def label_to_bob_pauli(label_idx: int) -> str:
    for pauli, perm in _BELL_PERM.items():
        for in_idx, out_idx in enumerate(perm):
            if in_idx == label_idx and out_idx == 0:
                return pauli
    return "I"


def _axis_map_for_gate(gate: str) -> Dict[str, str]:
    if gate == "H":
        return {"X": "Z", "Y": "Y", "Z": "X"}
    if gate == "S":
        return {"X": "Y", "Y": "X", "Z": "Z"}
    if gate == "I":
        return {"X": "X", "Y": "Y", "Z": "Z"}
    raise ValueError(f"Unknown gate: {gate}")


def _orig_to_work_from_gates(gates: Sequence[str]) -> Dict[str, str]:
    mapping = {"X": "X", "Y": "Y", "Z": "Z"}
    for gate in gates:
        gate_map = _axis_map_for_gate(gate)
        mapping = {axis: gate_map[mapping[axis]] for axis in mapping}
    return mapping


def _invert_axis_map(mapping: Dict[str, str]) -> Dict[str, str]:
    return {v: k for k, v in mapping.items()}


def _alignment_permutation_from_gates(gates: Sequence[str]) -> Tuple[str, str, str]:
    orig_to_work = _orig_to_work_from_gates(gates)
    work_to_orig = _invert_axis_map(orig_to_work)
    return (work_to_orig["X"], work_to_orig["Y"], work_to_orig["Z"])


def generate_alignment_gates(max_len: int = 3) -> Dict[Tuple[str, str, str], Tuple[str, ...]]:
    best: Dict[Tuple[str, str, str], Tuple[str, ...]] = {}
    best[_alignment_permutation_from_gates(())] = ()
    queue = [()]
    for _ in range(max_len):
        next_queue = []
        for seq in queue:
            for gate in ("H", "S"):
                new_seq = seq + (gate,)
                perm = _alignment_permutation_from_gates(new_seq)
                if perm not in best or len(new_seq) < len(best[perm]):
                    best[perm] = new_seq
                next_queue.append(new_seq)
        queue = next_queue
    return best


_ALIGNMENT_GATES = generate_alignment_gates()


def alignment_candidates(mode: str = "all") -> List[Tuple[Tuple[str, str, str], Tuple[str, ...]]]:
    if mode == "heuristic":
        identity = ("X", "Y", "Z")
        swap_xz = ("Z", "Y", "X")
        return [
            (identity, _ALIGNMENT_GATES.get(identity, ())),
            (swap_xz, _ALIGNMENT_GATES.get(swap_xz, ("H",))),
        ]
    if mode != "all":
        raise ValueError("mode must be 'all' or 'heuristic'")
    return [(perm, gates) for perm, gates in _ALIGNMENT_GATES.items()]


def apply_axis_permutation(p_rates: Dict[str, float], perm: Tuple[str, str, str]) -> Dict[str, float]:
    return {
        "X": float(p_rates[perm[0]]),
        "Y": float(p_rates[perm[1]]),
        "Z": float(p_rates[perm[2]]),
    }


_BELL_VECS = None
_DEJMPS_TABLE = None
_TABLE_ID = None


def _bell_pair_circuit(qc: QuantumCircuit, a: int, b: int, label_idx: int) -> None:
    # Prepare phi+ then map with Pauli on Bob
    qc.h(a)
    qc.cx(a, b)
    if label_idx == 1:  # phi-
        qc.z(b)
    elif label_idx == 2:  # psi+
        qc.x(b)
    elif label_idx == 3:  # psi-
        qc.z(b)
        qc.x(b)


def _get_bell_vectors():
    global _BELL_VECS
    if _BELL_VECS is not None:
        return _BELL_VECS
    vecs = []
    for i in range(4):
        qc = QuantumCircuit(2)
        _bell_pair_circuit(qc, 0, 1, i)
        vecs.append(Statevector.from_instruction(qc).data)
    _BELL_VECS = vecs
    return vecs


def _round_unitary(basis: str = "dejmps") -> QuantumCircuit:
    # One round on 4 qubits: target (q0,q3), source (q1,q2)
    qr = QuantumRegister(4, "q")
    qc = QuantumCircuit(qr)
    if basis == "dejmps":
        for q in (0, 1):
            qc.sdg(qr[q])
            qc.h(qr[q])
        for q in (2, 3):
            qc.s(qr[q])
            qc.h(qr[q])
    elif basis != "identity":
        raise ValueError(f"Unknown basis: {basis}")
    qc.cx(qr[1], qr[0])
    qc.cx(qr[2], qr[3])
    return qc


def _project_on_bits(state: Statevector, m0: int, m3: int):
    vec = state.data
    new = np.zeros_like(vec)
    prob = 0.0
    for idx, amp in enumerate(vec):
        b0 = (idx >> 0) & 1
        b3 = (idx >> 3) & 1
        if b0 == m0 and b3 == m3:
            new[idx] = amp
            prob += (amp.conjugate() * amp).real
    if prob == 0:
        return None, 0.0
    new = new / np.sqrt(prob)
    return Statevector(new), prob


def _round_output_for_labels(i: int, j: int, basis: str):
    qc = QuantumCircuit(4)
    _bell_pair_circuit(qc, 0, 3, i)
    _bell_pair_circuit(qc, 1, 2, j)
    sv = Statevector.from_instruction(qc)
    sv = sv.evolve(_round_unitary(basis))

    bell_vecs = _get_bell_vectors()
    total_prob = 0.0
    out_dm = None

    for m0, m3 in [(0, 0), (1, 1)]:
        post_sv, prob = _project_on_bits(sv, m0, m3)
        if prob == 0:
            continue
        # Correction if both were 1: Z on Alice source (q1)
        if m0 == 1:
            corr = QuantumCircuit(4)
            corr.z(1)
            post_sv = post_sv.evolve(corr)
        dm = partial_trace(post_sv, [0, 3])
        out_dm = dm * prob if out_dm is None else out_dm + prob * dm
        total_prob += prob

    if out_dm is None or total_prob == 0:
        return [0.0, 0.0, 0.0, 0.0], 0.0

    out_dm = out_dm / total_prob
    coeffs = [float(np.real(np.vdot(bv, out_dm.data @ bv))) for bv in bell_vecs]
    return coeffs, total_prob


def _get_round_table(basis: str):
    global _DEJMPS_TABLE, _TABLE_ID
    if basis == "dejmps":
        if _DEJMPS_TABLE is not None:
            return _DEJMPS_TABLE
    elif basis == "identity":
        if _TABLE_ID is not None:
            return _TABLE_ID
    else:
        raise ValueError(f"Unknown basis: {basis}")

    table = [[None] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            coeffs, keep = _round_output_for_labels(i, j, basis)
            table[i][j] = (coeffs, keep)

    if basis == "dejmps":
        _DEJMPS_TABLE = table
    else:
        _TABLE_ID = table
    return table


def update_bell_vector(p: np.ndarray, basis: str = "dejmps") -> Tuple[np.ndarray, float]:
    table = _get_round_table(basis)
    p_out = np.zeros(4, dtype=float)
    keep_prob = 0.0
    for i in range(4):
        for j in range(4):
            weight = p[i] * p[j]
            coeffs, keep = table[i][j]
            if weight == 0:
                continue
            p_out += weight * np.array(coeffs)
            keep_prob += weight * keep
    if keep_prob > 0:
        p_out = p_out / keep_prob
    return p_out, keep_prob


def optimize_rotation_sequence(
    p: np.ndarray,
    rounds: int,
    bases: Sequence[str] = ("identity", "dejmps"),
) -> Tuple[List[str], List[str], np.ndarray, float]:
    rotations: List[str] = []
    bases_out: List[str] = []
    keep_prob_total = 1.0
    for _ in range(rounds):
        best_rot = "I"
        best_basis = "dejmps"
        best_p = p
        best_f = -1.0
        best_keep = 0.0
        for basis in bases:
            p_next, keep = update_bell_vector(p, basis)
            best_label = int(np.argmax(p_next))
            rot = label_to_bob_pauli(best_label)
            p_rot = apply_bob_pauli_to_bell_vector(p_next, rot)
            if p_rot[0] > best_f:
                best_f = float(p_rot[0])
                best_rot = rot
                best_basis = basis
                best_p = p_rot
                best_keep = keep
        rotations.append(best_rot)
        bases_out.append(best_basis)
        p = best_p
        keep_prob_total *= best_keep
    return rotations, bases_out, p, keep_prob_total


def _objective_score(
    fidelity: float,
    success_prob: float,
    objective: str,
    success_threshold: Optional[float],
) -> float:
    if success_threshold is not None and success_prob < success_threshold:
        return -1.0
    if objective == "fidelity":
        return fidelity
    if objective == "success":
        return success_prob
    if objective == "fidelity_times_success":
        return fidelity * success_prob
    raise ValueError("objective must be 'fidelity', 'success', or 'fidelity_times_success'")


def plan_dejmps_from_error_row(
    row: Dict[str, object],
    candidate_num_bell_pairs: Sequence[int] = (2, 4, 8),
    alignment_mode: str = "all",
    bases: Sequence[str] = ("identity", "dejmps"),
    objective: str = "fidelity_times_success",
    success_threshold: Optional[float] = None,
) -> Optional[DejmpsPlan]:
    p = bell_vector_from_row(row)
    if p is None:
        return None

    rates = pauli_rates_from_bell_vector(p)
    rate_axes = {"X": rates["X"], "Y": rates["Y"], "Z": rates["Z"]}
    alignments = alignment_candidates(alignment_mode)

    best_plan: Optional[DejmpsPlan] = None
    best_score = -1.0

    for perm, gates in alignments:
        permuted = apply_axis_permutation(rate_axes, perm)
        p_work = bell_vector_from_pauli_rates(
            rates["I"], permuted["X"], permuted["Y"], permuted["Z"]
        )
        target_idx = int(np.argmax(p_work))
        target_label = BELL_LABELS[target_idx]
        pre_rot = label_to_bob_pauli(target_idx)
        p_rot = apply_bob_pauli_to_bell_vector(p_work, pre_rot)

        for num_bell_pairs in candidate_num_bell_pairs:
            rounds = {2: 1, 4: 2, 8: 3}.get(int(num_bell_pairs))
            if rounds is None:
                continue
            rot_seq, base_seq, p_out, keep_prob = optimize_rotation_sequence(
                p_rot, rounds, bases=bases
            )
            fidelity = float(p_out[0])
            score = _objective_score(fidelity, keep_prob, objective, success_threshold)
            if score > best_score:
                best_score = score
                best_plan = DejmpsPlan(
                    alignment=perm,
                    alignment_gates=gates,
                    target_label=target_label,
                    pre_rotation_pauli=pre_rot,
                    round_rotations=rot_seq,
                    round_bases=base_seq,
                    num_bell_pairs=int(num_bell_pairs),
                    rounds=rounds,
                    expected_fidelity=fidelity,
                    expected_success_probability=float(keep_prob),
                    objective=objective,
                )
    return best_plan


def plan_dejmps_for_edge(
    edge_id: Sequence[str],
    csv_path: Path | str = "edge_error_models.csv",
    **kwargs,
) -> Optional[DejmpsPlan]:
    row = load_error_model_row(edge_id, path=csv_path, prefer_latest=True)
    if row is None:
        return None
    return plan_dejmps_from_error_row(row, **kwargs)


@dataclass(frozen=True)
class RecalibrationPolicy:
    max_claims: int = 20
    fidelity_drift_threshold: float = 0.03


def should_recalibrate(
    claims_since_last: int,
    predicted_fidelity: Optional[float],
    observed_fidelity: Optional[float],
    policy: Optional[RecalibrationPolicy] = None,
) -> bool:
    policy = policy or RecalibrationPolicy()
    if claims_since_last >= policy.max_claims:
        return True
    if predicted_fidelity is None or observed_fidelity is None:
        return False
    return abs(observed_fidelity - predicted_fidelity) >= policy.fidelity_drift_threshold


def maybe_recalibrate_edge(
    client,
    edge_id: Sequence[str],
    claims_since_last: int,
    predicted_fidelity: Optional[float],
    observed_fidelity: Optional[float],
    csv_path: Path | str = "edge_error_models.csv",
    num_bell_pairs: int = 1,
    policy: Optional[RecalibrationPolicy] = None,
    note_prefix: str = "recalibration N=1, local Pauli on Alice",
    verbose: bool = True,
) -> Optional[ErrorModelResult]:
    if not should_recalibrate(
        claims_since_last, predicted_fidelity, observed_fidelity, policy=policy
    ):
        return None

    payload, err = estimate_bell_coeffs(client, edge_id, num_bell_pairs=num_bell_pairs)
    if err:
        if verbose:
            print(f"Recalibration error: {err.get('error', {}).get('message')}")
        return None

    coeffs, coeffs_norm, success_probs = payload
    p_sum = sum(coeffs.values())
    avg_success = sum(success_probs.values()) / max(len(success_probs), 1)

    edge_info = None
    if hasattr(client, "get_edge_info"):
        try:
            edge_info = client.get_edge_info(edge_id[0], edge_id[1])
        except Exception:
            edge_info = None

    row = {
        "node_a": canonical_edge(edge_id)[0],
        "node_b": canonical_edge(edge_id)[1],
        "p_phi_plus": coeffs["phi_plus"],
        "p_phi_minus": coeffs["phi_minus"],
        "p_psi_plus": coeffs["psi_plus"],
        "p_psi_minus": coeffs["psi_minus"],
        "p_sum": p_sum,
        "p_phi_plus_norm": coeffs_norm["phi_plus"],
        "p_phi_minus_norm": coeffs_norm["phi_minus"],
        "p_psi_plus_norm": coeffs_norm["psi_plus"],
        "p_psi_minus_norm": coeffs_norm["psi_minus"],
        "base_threshold": (edge_info or {}).get("base_threshold"),
        "difficulty_rating": (edge_info or {}).get("difficulty_rating"),
        "success_probability": avg_success,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "note": note_prefix,
    }
    append_error_row(Path(csv_path), row)

    return ErrorModelResult(
        edge_id=canonical_edge(edge_id),
        coeffs=coeffs,
        coeffs_norm=coeffs_norm,
        success_probs=success_probs,
        base_threshold=(edge_info or {}).get("base_threshold"),
        difficulty_rating=(edge_info or {}).get("difficulty_rating"),
    )


def estimate_bell_coeffs(client, edge_id: Sequence[str], num_bell_pairs: int = 1):
    """Estimate Bell coefficients for an edge using local Paulis on Alice."""
    coeffs: Dict[str, float] = {}
    success_probs: Dict[str, float] = {}
    for label, pauli in BELL_PAULI_MAP.items():
        circuit = bell_tomography_circuit(pauli)
        result = client.claim_edge(tuple(edge_id), circuit, flag_bit=0, num_bell_pairs=num_bell_pairs)
        if not result.get("ok"):
            return None, result
        data = result.get("data", {})
        coeffs[label] = float(data.get("fidelity", 0.0))
        success_probs[label] = float(data.get("success_probability", 0.0))

    p_sum = sum(coeffs.values())
    if p_sum > 0:
        coeffs_norm = {k: v / p_sum for k, v in coeffs.items()}
    else:
        coeffs_norm = {k: 0.0 for k in coeffs}

    return (coeffs, coeffs_norm, success_probs), None


def _difficulty_matches(value: Optional[float], difficulty_level) -> bool:
    if difficulty_level is None:
        return True
    if value is None:
        return False
    if isinstance(difficulty_level, str):
        try:
            difficulty_level = float(difficulty_level)
        except ValueError:
            return False
    if isinstance(difficulty_level, (list, set, tuple)):
        if len(difficulty_level) == 2 and all(isinstance(v, (int, float)) for v in difficulty_level):
            low, high = difficulty_level
            return low <= value <= high
        return value in difficulty_level
    return value == difficulty_level


def update_error_models_for_claimable_edges(
    client,
    difficulty_level,
    csv_path: Path | str = "edge_error_models.csv",
    num_bell_pairs: int = 1,
    note: str = "tomography N=1, local Pauli on Alice",
    verbose: bool = True,
) -> List[ErrorModelResult]:
    """
    For claimable edges with the given difficulty, estimate Bell coefficients and append to CSV.
    Skips edges already present in the CSV.

    difficulty_level can be:
      - a single numeric value (exact match)
      - a (min, max) tuple
      - a list/set of allowed values
    """
    csv_path = Path(csv_path)
    existing = load_existing_edges(csv_path)

    claimable = client.get_claimable_edges()
    filtered = []
    for edge in claimable:
        edge_id = canonical_edge(edge.get("edge_id", ("", "")))
        if edge_id in existing:
            continue
        diff = edge.get("difficulty_rating")
        try:
            diff_val = float(diff)
        except (TypeError, ValueError):
            diff_val = None
        if _difficulty_matches(diff_val, difficulty_level):
            filtered.append(edge)

    if verbose:
        print(
            f"Claimable: {len(claimable)} | New (difficulty={difficulty_level}): {len(filtered)}"
        )

    results: List[ErrorModelResult] = []
    for edge in filtered:
        edge_id = tuple(edge["edge_id"])
        if verbose:
            print(f"Tomography for edge {edge_id}...")
        payload, err = estimate_bell_coeffs(client, edge_id, num_bell_pairs=num_bell_pairs)
        if err:
            if verbose:
                print(f"  Error: {err.get('error', {}).get('message')}")
            continue

        coeffs, coeffs_norm, success_probs = payload
        p_sum = sum(coeffs.values())
        avg_success = sum(success_probs.values()) / max(len(success_probs), 1)

        a, b = canonical_edge(edge_id)
        row = {
            "node_a": a,
            "node_b": b,
            "p_phi_plus": coeffs["phi_plus"],
            "p_phi_minus": coeffs["phi_minus"],
            "p_psi_plus": coeffs["psi_plus"],
            "p_psi_minus": coeffs["psi_minus"],
            "p_sum": p_sum,
            "p_phi_plus_norm": coeffs_norm["phi_plus"],
            "p_phi_minus_norm": coeffs_norm["phi_minus"],
            "p_psi_plus_norm": coeffs_norm["psi_plus"],
            "p_psi_minus_norm": coeffs_norm["psi_minus"],
            "base_threshold": edge.get("base_threshold"),
            "difficulty_rating": edge.get("difficulty_rating"),
            "success_probability": avg_success,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "note": note,
        }
        append_error_row(csv_path, row)

        results.append(
            ErrorModelResult(
                edge_id=canonical_edge(edge_id),
                coeffs=coeffs,
                coeffs_norm=coeffs_norm,
                success_probs=success_probs,
                base_threshold=edge.get("base_threshold"),
                difficulty_rating=edge.get("difficulty_rating"),
            )
        )
        if verbose:
            print("  Saved to CSV.")

    return results
