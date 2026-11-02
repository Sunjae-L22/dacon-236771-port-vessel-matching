"""Normalized 0/1/2-event character channel; syllables are NFC Python characters.

For every event choose delete/dup/swap independently, then choose its position
uniformly. Equal-character swaps are retained. Swap on length <2 is identity;
all operations on the empty string are identity. Different event paths that
produce one observation retain their separate probability mass.

Training reads only English records. No Korean label or proxy result is used.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

OPERATIONS = ("delete", "dup", "swap")


def event_distribution(clean: str, operation: str) -> dict[str, float]:
    """Conditional output distribution given one operation, including no-ops."""
    if operation not in OPERATIONS:
        raise ValueError(operation)
    n = len(clean)
    if not n or (operation == "swap" and n < 2):
        return {clean: 1.0}
    out = defaultdict(float)
    count = n - 1 if operation == "swap" else n
    for i in range(count):
        if operation == "delete":
            query = clean[:i] + clean[i + 1:]
        elif operation == "dup":
            query = clean[:i + 1] + clean[i] + clean[i + 1:]
        else:
            query = clean[:i] + clean[i + 1] + clean[i] + clean[i + 2:]
        out[query] += 1.0 / count
    return dict(out)


def _one_coefficients(clean: str, query: str) -> dict[str, float]:
    """Pair-specific one-event coefficients without materializing all outputs."""
    n = len(clean)
    if not n:
        return {op: 1.0 for op in OPERATIONS} if query == clean else {}
    out = {}
    if len(query) == n - 1:
        v = sum(clean[:i] + clean[i + 1:] == query for i in range(n)) / n
        if v:
            out["delete"] = v
    elif len(query) == n + 1:
        v = sum(clean[:i + 1] + clean[i] + clean[i + 1:] == query for i in range(n)) / n
        if v:
            out["dup"] = v
    elif len(query) == n:
        if n < 2:
            if query == clean:
                out["swap"] = 1.0
        else:
            v = sum(clean[:i] + clean[i + 1] + clean[i] + clean[i + 2:] == query
                    for i in range(n - 1)) / (n - 1)
            if v:
                out["swap"] = v
    return out


def path_coefficients(clean: str, query: str) -> dict[str, float]:
    """JSON-safe coefficients by event sequence ('' denotes zero events).

    Coefficients sum position-path probabilities conditional on the chosen
    operation sequence. Multiply by pi[len(sequence)] and operation priors.
    No OSA restriction: second events act on the actual first-event output.
    """
    if abs(len(clean) - len(query)) > 2:
        return {}
    out = defaultdict(float)
    if clean == query:
        out[""] = 1.0
    out.update(_one_coefficients(clean, query))
    for first in OPERATIONS:
        for middle, first_mass in event_distribution(clean, first).items():
            for second, second_mass in _one_coefficients(middle, query).items():
                out[first + "/" + second] += first_mass * second_mass
    return dict(sorted(out.items()))


def coefficient_score(coefficients: dict[str, float], params: dict) -> float:
    steps = params["step_probabilities"]
    operations = params["operation_probabilities"]
    total = 0.0
    for key, coefficient in coefficients.items():
        seq = key.split("/") if key else []
        total += coefficient * steps[str(len(seq))] * math.prod(operations[op] for op in seq)
    return total


def score(clean: str, query: str, params: dict) -> float:
    """Normalized P(query | clean) for the finite 0/1/2-event model."""
    return coefficient_score(path_coefficients(clean, query), params)


def support(clean: str) -> set[str]:
    one = {q for op in OPERATIONS for q in event_distribution(clean, op)}
    return {clean} | one | {q for middle in one for op in OPERATIONS
                           for q in event_distribution(middle, op)}


def fit(source: Path, output: Path) -> dict:
    raw = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    records = []
    excluded = 0
    for record in raw:
        vessel_id = record["vessel_id"]
        fold = int(hashlib.sha256(("channel-v1:" + vessel_id).encode()).hexdigest()[:8], 16) % 5
        if fold == 0:
            continue
        if any("가" <= c <= "힣" for c in record["clean"] + record["query"]):
            raise ValueError("Non-English training record")
        coeff = path_coefficients(record["clean"], record["query"])
        if not coeff:
            excluded += 1
            continue
        records.append((record, [(key.split("/") if key else [], value)
                                  for key, value in coeff.items()]))
    if not records:
        raise ValueError("No reconstructible fitting records")
    operations = {op: 1 / 3 for op in OPERATIONS}
    steps = {str(k): 1 / 3 for k in range(3)}
    history = []
    for iteration in range(30):
        op_counts = {op: 1.0 for op in OPERATIONS}
        step_counts = {str(k): 1.0 for k in range(3)}
        ll = 0.0
        for _, paths in records:
            weighted = [(seq, value * steps[str(len(seq))] * math.prod(operations[op] for op in seq))
                        for seq, value in paths]
            z = math.fsum(value for _, value in weighted)
            ll += math.log(z)
            for seq, value in weighted:
                responsibility = value / z
                step_counts[str(len(seq))] += responsibility
                for op in seq:
                    op_counts[op] += responsibility
        history.append({"iteration": iteration, "log_likelihood_before_update": ll,
                        "log_posterior_before_update": ll + sum(map(math.log, operations.values()))
                        + sum(map(math.log, steps.values()))})
        op_total, step_total = math.fsum(op_counts.values()), math.fsum(step_counts.values())
        operations = {op: count / op_total for op, count in op_counts.items()}
        steps = {key: count / step_total for key, count in step_counts.items()}
    params = {"operation_probabilities": operations, "step_probabilities": steps}
    final_ll = sum(math.log(math.fsum(value * steps[str(len(seq))]
                    * math.prod(operations[op] for op in seq) for seq, value in paths))
                   for _, paths in records)
    result = {
        **params, "iterations": 30, "pseudocount_each_category": 1.0,
        "initialization": "uniform over three operations and three step counts",
        "fit_unit": "unique English (vessel_id, normalized query) forms from source",
        "fit_fold_rule": "sha256('channel-v1:' + vessel_id)[:8] mod 5 != 0",
        "fit_record_count": len(records),
        "fit_exact_count": sum(record["clean"] == record["query"] for record, _ in records),
        "excluded_nonreconstructible_fit_records": excluded,
        "train_vessel_ids": sorted({record["vessel_id"] for record, _ in records}),
        "train_form_keys": [[record["vessel_id"], record["clean"], record["query"]] for record, _ in records],
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "final_log_likelihood": final_ll,
        "final_log_posterior": final_ll + sum(map(math.log, operations.values())) + sum(map(math.log, steps.values())),
        "em_history": history,
        "event_definition": "independent op priors; uniform deletion/duplication position or adjacent swap pair; equal-symbol swaps retained; swap length<2 identity; empty all operations identity",
        "likelihood_definition": "sum all 0/1/2-event paths, including identity cancellations; additive-one counts correspond to Dirichlet(2) MAP",
        "limitation": "Fit includes only forms reconstructible by this support. English event process is not evidence of the Korean noise distribution. No Korean labels read."
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent.parent / "english_pairs.jsonl")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "NORMALIZED_FIT.json")
    args = parser.parse_args()
    result = fit(args.source, args.output)
    print(json.dumps({key: result[key] for key in ("operation_probabilities", "step_probabilities", "fit_record_count", "fit_exact_count", "excluded_nonreconstructible_fit_records", "final_log_likelihood")}, indent=2))
