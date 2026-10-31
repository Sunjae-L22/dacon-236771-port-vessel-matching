#!/bin/sh
set -eu
R=/Users/sunjae/Desktop/항만
E=$R/shared/experiments
S=$R/submissions
PY=/Users/sunjae/.pyenv/versions/3.11.9/bin/python3
cd /tmp

echo "[1/4] Launching fast Dedicated Korean-Only training for 81 (SEED_OFFSET=0) and 82 (SEED_OFFSET=5000)..."
for n in 81 82; do
  off=$(( (n - 81) * 5000 ))
  PYTHONHASHSEED=0 KO_DICT=1 LLM_DICT_CSV=llm_korean_scored.csv LLM_TRAIN_CSV=llm_korean_scored.csv SUBMIT_NUM=$n SEEDS=3 \
    SEED_OFFSET=$off OUT_DIR=submit$n CACHE_DIR=$E/cache nohup "$PY" "$E/pinned/build81/make_submit.py" > "$E/make$n.log" 2>&1 &
done
wait

echo "[2/4] Applying post-hoc train-target exclusion & determinism patch..."
for n in 81 82; do
  grep -q "wrote .*${n}_submit.zip" "$E/make$n.log" || { echo "build $n FAILED"; exit 1; }
  "$PY" "$R/shared/src/add_exclusion.py" "$S/${n}_submit.zip" "$S/${n}_submit.zip"
done

echo "[3/4] Building 83_submit.zip (Dedicated Ko [81+82] + Joint Ko [72] + Best En [65])..."
"$PY" "$E/build_83_from_81_82.py"

echo "[4/4] Running local_check.py on 83_submit.zip..."
"$PY" "$R/shared/src/local_check.py" "$S/83_submit.zip" > "$E/check_83.log" 2>&1
echo "83 ready $(date)"
tail -3 "$E/check_83.log"
