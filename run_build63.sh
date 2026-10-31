#!/bin/sh
set -e
R=/Users/sunjae/Desktop/항만
P=$R/shared/experiments/pinned/build63
E=$R/shared/experiments

echo "[1/3] Launching build 63 (SEED_OFFSET=0) and 64 (SEED_OFFSET=5000)..."
(cd $P && KO_DICT=1 LLM_DICT_CSV=llm_korean_scored.csv LLM_TRAIN_CSV=llm_korean_scored.csv SUBMIT_NUM=63 SEEDS=3 SEED_OFFSET=0 OUT_DIR=submit63 CACHE_DIR=$E/cache python3 make_submit.py && cd $R && python3 shared/src/add_exclusion.py submissions/63_submit.zip submissions/63_submit.zip) > $E/make63.log 2>&1 &
PID63=$!

(cd $P && KO_DICT=1 LLM_DICT_CSV=llm_korean_scored.csv LLM_TRAIN_CSV=llm_korean_scored.csv SUBMIT_NUM=64 SEEDS=3 SEED_OFFSET=5000 OUT_DIR=submit64 CACHE_DIR=$E/cache python3 make_submit.py && cd $R && python3 shared/src/add_exclusion.py submissions/64_submit.zip submissions/64_submit.zip) > $E/make64.log 2>&1 &
PID64=$!

wait $PID63
wait $PID64

echo "[2/3] Merging 63 + 64 -> 65_submit.zip (6 models)..."
cd $R
python3 shared/src/merge_rankers.py submissions/65_submit.zip submissions/63_submit.zip submissions/64_submit.zip

echo "[3/3] Running local_check on 65_submit.zip..."
python3 shared/src/local_check.py submissions/65_submit.zip | tee $E/check_65.log
echo "Done! Ready to submit: submissions/65_submit.zip"
