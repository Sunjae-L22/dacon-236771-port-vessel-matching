"""DACON inference entry point: reads data/test.csv, writes output/submission.csv."""
from pathlib import Path
import sys

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "model"))
from matcher import Catalog
from pipeline import predict, read_records

# False(기본값): 모델 예측만 쓴다.
# True: test call_sign이 train에 있으면 train의 정답 선박으로 답한다. 대회 규칙 해석이 확정될 때까지
# 쓰지 않는다. 켜려면 model.pkl도 fit(train, cat, callsign=True)로 다시 만들어야 한다.
USE_CALLSIGN = False


def run(test_path="data/test.csv", catalog_path="data/vessel_catalog.csv",
        model_path="model/model.pkl", output_path="output/submission.csv") -> Path:
    test = read_records(test_path)
    cat = Catalog(pd.read_csv(catalog_path, encoding="utf-8-sig"))
    bundle = joblib.load(model_path)
    preds = predict(bundle, test, cat, use_callsign=USE_CALLSIGN)
    if len(preds) != len(test) or not set(preds) <= set(cat.id_to_index):
        raise ValueError("Invalid predictions")
    out = pd.DataFrame({"sample_id": test["sample_id"], "target_vessel_id": preds})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved {len(out)} predictions to {output_path}")
    return output_path


if __name__ == "__main__":
    run()
