#!/bin/sh
set -eu
R=/Users/sunjae/Desktop/항만
E=$R/shared/experiments
S=$R/submissions
PY=/Users/sunjae/.pyenv/versions/3.11.9/bin/python3
cd /tmp

echo "[1/4] Launching parallel training from /tmp for 76 (SEED_OFFSET=0) and 77 (SEED_OFFSET=5000)..."
for n in 76 77; do
  off=$(( (n - 76) * 5000 ))
  PYTHONHASHSEED=0 KO_DICT=1 LLM_DICT_CSV=llm_korean_scored.csv LLM_TRAIN_CSV=llm_korean_scored.csv SUBMIT_NUM=$n SEEDS=3 \
    SEED_OFFSET=$off OUT_DIR=submit$n CACHE_DIR=$E/cache nohup "$PY" "$E/pinned/build67/make_submit.py" > "$E/make$n.log" 2>&1 &
done
wait

echo "[2/4] Applying post-hoc train-target exclusion & determinism patch..."
for n in 76 77; do
  grep -q "wrote .*${n}_submit.zip" "$E/make$n.log" || { echo "build $n FAILED"; exit 1; }
  "$PY" "$R/shared/src/add_exclusion.py" "$S/${n}_submit.zip" "$S/${n}_submit.zip"
done

echo "[3/4] Merging 76 + 77 into 6-model ensemble submissions/78_submit.zip..."
"$PY" "$R/shared/src/merge_rankers.py" "$S/78_submit.zip" "$S/76_submit.zip" "$S/77_submit.zip"

echo "[4/4] Running local_check.py on 78_submit.zip..."
"$PY" "$R/shared/src/local_check.py" "$S/78_submit.zip" > "$E/check_78.log" 2>&1
echo "78 ready $(date)"
tail -3 "$E/check_78.log"
