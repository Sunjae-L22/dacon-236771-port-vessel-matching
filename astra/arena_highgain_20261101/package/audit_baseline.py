"""Fail-closed correspondence audit for the unchanged C_vote observation pass."""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import math
import unicodedata


def _rows(path, jsonl=False, allow_absent=False):
    path = Path(path)
    if not path.exists():
        if allow_absent:
            return {}
        raise ValueError(f"Required file missing: {path}")
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = ([json.loads(line) for line in source if line.strip()] if jsonl
                else list(csv.DictReader(source)))
    indexed = {}
    for row in rows:
        sid = row.get("sample_id")
        if not isinstance(sid, str) or not sid.strip() or sid in indexed:
            raise ValueError(f"Missing/duplicate sample_id in {path}: {sid!r}")
        indexed[sid] = row
    return indexed


def _exact(actual, expected, context):
    if set(actual) != set(expected):
        raise ValueError(f"{context} IDs differ: missing={sorted(set(expected)-set(actual))[:5]}, "
                         f"extra={sorted(set(actual)-set(expected))[:5]}")


def _number(value, context, nonnegative=False):
    if isinstance(value, bool):
        raise ValueError(f"Boolean score in {context}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid number in {context}") from exc
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise ValueError(f"Nonfinite/negative number in {context}")
    return number


def _vector(value, length, context, nonnegative=False):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"Array shape mismatch in {context}")
    return [_number(v, context, nonnegative) for v in value]


def _candidate_ids(value, catalog_ids, context):
    if (not isinstance(value, list) or not value or
            any(not isinstance(v, str) or not v for v in value) or
            len(set(value)) != len(value) or not set(value) <= catalog_ids):
        raise ValueError(f"Invalid/duplicate/unknown candidate IDs in {context}")
    return value


def _catalog(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    ids = [row.get("vessel_id") for row in rows]
    if not ids or any(not isinstance(v, str) or not v.strip() for v in ids) or len(set(ids)) != len(ids):
        raise ValueError("Invalid catalog vessel IDs")
    return set(ids)


def audit(payload, test, baseline, catalog=None):
    """Validate all Korean traces against actual components; write receipt on pass.

    catalog defaults to vessel_catalog.csv next to test, then payload/data/.
    All-excluded v20 candidates retain original scores, exactly as script.py.
    An absent ko/hs CSV or trace is allowed only when no Korean row was input.
    """
    payload, test, baseline = map(Path, (payload, test, baseline))
    if catalog is None:
        options = [test.parent / "vessel_catalog.csv", payload / "data/vessel_catalog.csv"]
        catalog = next((p for p in options if p.exists()), None)
        if catalog is None:
            raise ValueError("Official catalog required; pass catalog= explicitly")
    catalog = Path(catalog)
    known = _catalog(catalog)
    queries = _rows(test)
    if not queries:
        raise ValueError("Empty audit input")
    korean = {sid for sid, row in queries.items()
              if any("가" <= c <= "힣" for c in unicodedata.normalize("NFC", row.get("vessel_name", "")))}
    component = {}
    for tag in ("en", "ko", "hs"):
        expected = set(queries) if tag == "en" else korean
        component[tag] = _rows(baseline / f"{tag}_pred.csv", allow_absent=not expected)
        _exact(component[tag], expected, f"component {tag}")
        for sid, row in component[tag].items():
            if row.get("target_vessel_id") not in known:
                raise ValueError(f"Invalid {tag} output ID: {sid}")
    traces = {}
    for tag in ("en", "hs", "v20"):
        traces[tag] = _rows(baseline / f"internal/{tag}_candidates.jsonl", jsonl=True,
                            allow_absent=not korean)
        _exact(traces[tag], korean, f"trace {tag}")
    exclude_path = payload / "model/ko/model/train_vessels.json"
    excluded_raw = json.loads(exclude_path.read_text())
    if not isinstance(excluded_raw, list) or any(not isinstance(v, str) for v in excluded_raw):
        raise ValueError("Malformed original train-vessel exclusion list")
    excluded = set(excluded_raw)
    for sid in sorted(korean):
        for tag in ("en", "hs"):
            t = traces[tag][sid]
            ids = _candidate_ids(t.get("candidate_ids"), known, f"{tag}:{sid}")
            _vector(t.get("scores"), len(ids), f"{tag} scores:{sid}")
            top = _candidate_ids(t.get("top20_ids"), known, f"{tag} top20:{sid}")
            if len(top) > 20 or not set(top) <= set(ids):
                raise ValueError(f"Invalid {tag} top20 subset: {sid}")
            if top[0] != component[tag][sid]["target_vessel_id"]:
                raise ValueError(f"{tag} trace winner mismatch: {sid}")
        trace_gap = _number(traces["en"][sid].get("gap110"), f"trace gap:{sid}")
        output_gap = _number(component["en"][sid].get("ko_gap"), f"output gap:{sid}")
        if abs(trace_gap - output_gap) > 1e-9:
            raise ValueError(f"110 gap mismatch: {sid}")
        t = traces["v20"][sid]
        ids = _candidate_ids(t.get("candidate_ids"), known, f"v20:{sid}")
        base = _vector(t.get("base_prob"), len(ids), f"v20 base:{sid}", True)
        final = _vector(t.get("final_scores"), len(ids), f"v20 final:{sid}", True)
        if sum(base) <= 0 or sum(final) <= 0:
            raise ValueError(f"Zero v20 score mass: {sid}")
        expected_top = [ids[i] for i in sorted(range(len(ids)), key=lambda i: -base[i])[:20]]
        if t.get("top20_ids") != expected_top:
            raise ValueError(f"v20 stable top20 mismatch: {sid}")
        allowed = [i for i, v in enumerate(ids) if v not in excluded]
        positions = allowed or list(range(len(ids)))
        winner = ids[max(positions, key=lambda i: final[i])]
        if winner != component["ko"][sid]["target_vessel_id"]:
            raise ValueError(f"v20 post-exclusion winner mismatch: {sid}")
        for view in t.get("views", []):
            view_ids = _candidate_ids(view.get("candidate_ids"), known, f"v20 view:{sid}")
            if view_ids != expected_top:
                raise ValueError(f"v20 view candidate order mismatch: {sid}")
            _vector(view.get("scores"), len(view_ids), f"v20 view scores:{sid}")
    receipt = dict(state="PASS", rows=len(queries), korean_rows=len(korean),
                   component_rows={k: len(v) for k, v in component.items()},
                   trace_rows={k: len(v) for k, v in traces.items()},
                   original_v20_post_exclusion_verified=True,
                   checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   test_sha256=hashlib.sha256(test.read_bytes()).hexdigest(),
                   catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest())
    (baseline / "TRACE_AUDIT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
