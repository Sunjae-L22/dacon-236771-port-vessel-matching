"""Post-hoc error location only; never used to choose evaluation rows or policy."""
from pathlib import Path
import csv, json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def rows(path):
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def mapping(path):
    return {r["sample_id"]: r["target_vessel_id"] for r in rows(path)}

def analyze(work, data):
    truth = {r["sample_id"]: r for r in rows(data / "truth.csv")}
    en = {r["sample_id"]: r for r in rows(data / "en_pred.csv")}
    hs, v20 = mapping(data / "hs_pred.csv"), mapping(work / "v20_hard_pred.csv")
    details = []
    for row in rows(data / "test.csv"):
        sid = row["sample_id"]
        target = truth[sid]["target_vessel_id"]
        a, b = en[sid]["target_vessel_id"], hs[sid]
        protected = not any("가" <= c <= "힣" for c in row["vessel_name"]) or float(en[sid]["ko_gap"]) >= 4
        c = v20.get(sid)
        pred = a if protected or a == b else c
        assert pred is not None
        if pred == target:
            continue
        voted = target in (a, b, c)
        path = work / "capture" / (sid + ".npz")
        contains, top20 = None, None
        if path.exists():
            z = np.load(path)
            ids = z["candidate_ids"].astype(str)
            contains = target in ids
            top20 = target in ids[np.argsort(-z["base"], kind="stable")[:20]]
        details.append({"sample_id": sid, "truth": target, "cvote": pred,
                        "variant": truth[sid]["variant"], "protected": protected,
                        "truth_is_any_voter_top1": voted,
                        "truth_in_v20_candidates": contains,
                        "truth_in_v20_top20": top20})
    out = {"rows": len(truth), "errors": len(details), "posthoc_oracle_not_expected_gain": True,
           "errors_resolvable_by_perfect_three_voter_choice": sum(r["truth_is_any_voter_top1"] for r in details),
           "errors_no_voter_top1": sum(not r["truth_is_any_voter_top1"] for r in details),
           "no_voter_top1_but_truth_in_v20_top20": sum(not r["truth_is_any_voter_top1"] and r["truth_in_v20_top20"] is True for r in details),
           "errors_protected": sum(r["protected"] for r in details), "details": details}
    return out

if __name__ == "__main__":
    out = {"development400": analyze(HERE, ROOT / "astra/new_direction_20261101/local_cvote/eval"),
           "confirmation90": analyze(HERE / "confirm", HERE / "confirm/eval")}
    (HERE / "ORACLE_INVENTORY.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps({k: {a: b for a, b in v.items() if a != "details"} for k, v in out.items()}, indent=2))
